
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void carid_encoder_create_picture_list (CaridEncoder *encoder);
static void predict_dc (CaridMotionVector *mv, CaridFrame *frame,
    int x, int y, int w, int h);
static void predict_motion (CaridMotionVector *mv, CaridFrame *frame,
    CaridFrame *reference_frame, int x, int y, int w, int h);

static void carid_encoder_frame_queue_push (CaridEncoder *encoder,
    CaridFrame *frame);
static CaridFrame * carid_encoder_frame_queue_get (CaridEncoder *encoder,
    int frame_number);
static void carid_encoder_frame_queue_remove (CaridEncoder *encoder,
    int frame_number);
static void carid_encoder_reference_add (CaridEncoder *encoder,
    CaridFrame *frame);
static CaridFrame * carid_encoder_reference_get (CaridEncoder *encoder,
    int frame_number);
static void carid_encoder_reference_retire (CaridEncoder *encoder,
    int frame_number);



CaridEncoder *
carid_encoder_new (void)
{
  CaridEncoder *encoder;
  CaridParams *params;
  int base_quant_index;
  int i;

  encoder = malloc(sizeof(CaridEncoder));
  memset (encoder, 0, sizeof(CaridEncoder));

  encoder->tmpbuf = malloc(1024 * 2);
  encoder->tmpbuf2 = malloc(1024 * 2);

  encoder->subband_buffer = carid_buffer_new_and_alloc (100000);

  params = &encoder->params;
  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;
  params->transform_depth = 4;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;

  base_quant_index = 18;
  encoder->encoder_params.quant_index_dc = 6;
  for(i=0;i<params->transform_depth;i++){
    encoder->encoder_params.quant_index[i] =
      base_quant_index - 2 * (params->transform_depth - 1 - i);
  }

  return encoder;
}

void
carid_encoder_free (CaridEncoder *encoder)
{
  if (encoder->frame) {
    carid_frame_free (encoder->frame);
  }

  free (encoder->tmpbuf);
  free (encoder->tmpbuf2);
  free (encoder);
}


void
carid_encoder_set_wavelet_type (CaridEncoder *encoder, int wavelet_type)
{
  encoder->params.wavelet_filter_index = wavelet_type;
  CARID_DEBUG("set wavelet %d", wavelet_type);
}

void
carid_encoder_set_size (CaridEncoder *encoder, int width, int height)
{
  CaridParams *params = &encoder->params;

  if (params->width == width && params->height == height) return;

  params->width = width;
  params->height = height;
  params->chroma_width =
    (width + params->chroma_h_scale - 1) / params->chroma_h_scale;
  params->chroma_height =
    (height + params->chroma_v_scale - 1) / params->chroma_v_scale;

  if (encoder->frame) {
    carid_frame_free (encoder->frame);
    encoder->frame = NULL;
  }

  encoder->need_rap = TRUE;
}

#if 0
static int
round_up_pow2 (int x, int pow)
{
  x += (1<<pow) - 1;
  x &= ~((1<<pow) - 1);
  return x;
}
#endif

void
carid_encoder_push_frame (CaridEncoder *encoder, CaridFrame *frame)
{
  frame->frame_number = encoder->frame_queue_index;
  encoder->frame_queue_index++;

  carid_encoder_frame_queue_push (encoder, frame);
}

CaridBuffer *
carid_encoder_encode (CaridEncoder *encoder)
{
  CaridBuffer *outbuffer;
  CaridBuffer *subbuffer;
  int i;
  
  if (encoder->need_rap) {
    outbuffer = carid_buffer_new_and_alloc (0x100);

    encoder->bits = carid_bits_new ();
    carid_bits_encode_init (encoder->bits, outbuffer);

    carid_encoder_encode_rap (encoder);
    encoder->need_rap = FALSE;

    if (encoder->bits->offset > 0) {
      subbuffer = carid_buffer_new_subbuffer (outbuffer, 0,
          encoder->bits->offset/8);
    } else {
      subbuffer = NULL;
    }
    carid_bits_free (encoder->bits);
    carid_buffer_unref (outbuffer);

    return subbuffer;
  }
 
  if (encoder->picture_index >= encoder->n_pictures) {
    carid_encoder_create_picture_list (encoder);
  }
  encoder->picture = &encoder->picture_list[encoder->picture_index];

  if (encoder->picture->n_refs > 0) {
    encoder->ref_frame0 = carid_encoder_reference_get (encoder,
        encoder->picture->reference_frame_number[0]);
    CARID_ASSERT (encoder->ref_frame0 != NULL);
  }
  if (encoder->picture->n_refs > 1) {
    encoder->ref_frame1 = carid_encoder_reference_get (encoder,
        encoder->picture->reference_frame_number[1]);
    CARID_ASSERT (encoder->ref_frame1 != NULL);
  }

  encoder->encode_frame = carid_encoder_frame_queue_get (encoder,
      encoder->picture->frame_number);
  if (encoder->encode_frame == NULL) return NULL;

  CARID_ERROR("encoding picture frame_number=%d is_ref=%d n_refs=%d",
      encoder->picture->frame_number, encoder->picture->is_ref,
      encoder->picture->n_refs);

  carid_encoder_frame_queue_remove (encoder, encoder->picture->frame_number);

  outbuffer = carid_buffer_new_and_alloc (0x10000);

  encoder->bits = carid_bits_new ();
  carid_bits_encode_init (encoder->bits, outbuffer);

  CARID_DEBUG("frame number %d", encoder->frame_number);

  if (encoder->picture->n_refs > 0) {
    carid_encoder_encode_inter (encoder);
  } else {
    carid_encoder_encode_intra (encoder);
  }

  for(i=0;i<encoder->picture->n_retire;i++){
    carid_encoder_reference_retire (encoder, encoder->picture->retire[i]);
  }

  encoder->picture_index++;

  CARID_ERROR("encoded %d bits", encoder->bits->offset);

  if (encoder->bits->offset > 0) {
    subbuffer = carid_buffer_new_subbuffer (outbuffer, 0,
        encoder->bits->offset/8);
  } else {
    subbuffer = NULL;
  }
  carid_bits_free (encoder->bits);
  carid_buffer_unref (outbuffer);

  return subbuffer;
}

static void
carid_encoder_create_picture_list (CaridEncoder *encoder)
{
  int type = 1;
  int i;

  switch(type) {
    case 0:
      /* intra only */
      encoder->n_pictures = 1;
      encoder->picture_list[0].is_ref = 0;
      encoder->picture_list[0].n_refs = 0;
      encoder->picture_list[0].frame_number = encoder->frame_number;
      encoder->frame_number++;
      break;
    case 1:
      /* */
      encoder->n_pictures = 8;
      encoder->picture_list[0].is_ref = 1;
      encoder->picture_list[0].n_refs = 0;
      encoder->picture_list[0].frame_number = encoder->frame_number;
      if (encoder->frame_number != 0) {
        encoder->picture_list[0].n_retire = 1;
        encoder->picture_list[0].retire[0] = encoder->frame_number - 8;
      }
      for(i=1;i<8;i++){
        encoder->picture_list[i].is_ref = 0;
        encoder->picture_list[i].n_refs = 1;
        encoder->picture_list[i].frame_number = encoder->frame_number + i;
        encoder->picture_list[i].reference_frame_number[0] =
          encoder->frame_number;
        encoder->picture_list[i].n_retire = 0;
      }
      encoder->frame_number+=8;
      break;
    case 2:
      /* */
      i = 0;
      if (encoder->frame_number == 0) {
        encoder->n_pictures = 1;
        encoder->picture_list[0].is_ref = 1;
        encoder->picture_list[0].n_refs = 0;
        encoder->picture_list[0].frame_number = encoder->frame_number;
      } else {
        encoder->picture_list[0].is_ref = 1;
        encoder->picture_list[0].n_refs = 0;
        encoder->picture_list[0].frame_number = encoder->frame_number + 7;
        for(i=1;i<8;i++){
          encoder->picture_list[i].is_ref = 0;
          encoder->picture_list[i].n_refs = 2;
          encoder->picture_list[i].frame_number = encoder->frame_number + i - 1;
          encoder->picture_list[i].reference_frame_number[0] =
            encoder->frame_number - 1;
          encoder->picture_list[i].reference_frame_number[0] =
            encoder->frame_number + 7;
          encoder->picture_list[i].n_retire = 0;
        }
        encoder->picture_list[7].n_retire = 1;
        encoder->picture_list[7].retire[0] = encoder->frame_number - 1;
      }
      encoder->frame_number+=encoder->n_pictures;
      break;
    default:
      CARID_ASSERT(0);
      break;
  }
  encoder->picture_index = 0;
}

void
carid_encoder_iwt_transform (CaridEncoder *encoder, int component)
{
  int16_t *frame_data;
  CaridParams *params = &encoder->params;
  int16_t *tmp = encoder->tmpbuf;
  int width;
  int height;
  int level;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }
  
  frame_data = (int16_t *)encoder->frame->components[component].data;
  for(level=0;level<params->transform_depth;level++) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = width << level;

    CARID_DEBUG("wavelet transform %dx%d stride %d", w, h, stride);
    carid_wavelet_transform_2d (params->wavelet_filter_index,
        frame_data, stride*2, w, h, tmp);
  }
}

void
carid_encoder_inverse_iwt_transform (CaridEncoder *encoder, int component)
{
  int16_t *frame_data;
  CaridParams *params = &encoder->params;
  int16_t *tmp = encoder->tmpbuf;
  int width;
  int height;
  int level;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }
  
  frame_data = (int16_t *)encoder->frame->components[component].data;
  for(level=params->transform_depth-1; level >=0;level--) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = width << level;

    carid_wavelet_inverse_transform_2d (params->wavelet_filter_index,
        frame_data, stride*2, w, h, tmp);
  }
}

void
carid_encoder_encode_intra (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int is_ref = encoder->picture->is_ref;

  carid_params_calculate_iwt_sizes (params);

  if (encoder->frame == NULL) {
    encoder->frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  carid_encoder_encode_frame_header (encoder, CARID_PARSE_CODE_INTRA_REF);

  carid_frame_convert (encoder->frame, encoder->encode_frame);

  carid_frame_free (encoder->encode_frame);

  carid_encoder_encode_transform_parameters (encoder);

  carid_encoder_iwt_transform (encoder, 0);
  carid_encoder_encode_transform_data (encoder, 0);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 0);
  }

  carid_encoder_iwt_transform (encoder, 1);
  carid_encoder_encode_transform_data (encoder, 1);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 1);
  }

  carid_encoder_iwt_transform (encoder, 2);
  carid_encoder_encode_transform_data (encoder, 2);
  if (is_ref) {
    CaridFrame *ref_frame;

    carid_encoder_inverse_iwt_transform (encoder, 2);

    ref_frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    carid_frame_convert (ref_frame, encoder->frame);

    carid_encoder_reference_add (encoder, ref_frame);
  }

}

void
carid_encoder_encode_inter (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  //int is_ref = 0;

  carid_params_calculate_mc_sizes (params);
  carid_params_calculate_iwt_sizes (params);

  if (encoder->frame == NULL) {
    encoder->frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  carid_encoder_encode_frame_header (encoder, CARID_PARSE_CODE_INTER_NON_REF);

  carid_encoder_motion_predict (encoder);

  carid_encoder_encode_frame_prediction (encoder);

  carid_frame_free (encoder->encode_frame);
#if 0
  carid_frame_convert (encoder->frame, encoder->encode_frame);
  carid_frame_free (encoder->encode_frame);

  carid_encoder_encode_transform_parameters (encoder);

  carid_encoder_iwt_transform (encoder, 0);
  carid_encoder_encode_transform_data (encoder, 0);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 0);
  }

  carid_encoder_iwt_transform (encoder, 1);
  carid_encoder_encode_transform_data (encoder, 1);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 1);
  }

  carid_encoder_iwt_transform (encoder, 2);
  carid_encoder_encode_transform_data (encoder, 2);
  if (is_ref) {
    CaridFrame *ref_frame;

    carid_encoder_inverse_iwt_transform (encoder, 2);

    ref_frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    carid_frame_convert (ref_frame, encoder->frame);
  }
#endif
}

void
carid_encoder_encode_frame_prediction (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i,j;
  int global_motion;

  /* block params flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* mv precision flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* global motion flag */
  global_motion = FALSE;
  carid_bits_encode_bit (encoder->bits, global_motion);

  /* block data length */
  carid_bits_encode_uegol (encoder->bits, 100);

  carid_bits_sync (encoder->bits);

  for(j=0;j<4*params->y_num_mb;j+=4){
    for(i=0;i<4*params->x_num_mb;i+=4){
      int k,l;
      int mb_using_global = FALSE;
      int mb_split = 2;
      int mb_common = FALSE;

      if (global_motion) {
        carid_bits_encode_bit (encoder->bits, mb_using_global);
      } else {
        CARID_ASSERT(mb_using_global == FALSE);
      }
      if (!mb_using_global) {
        CARID_ASSERT(mb_split < 3);
        carid_bits_encode_bits (encoder->bits, 2, mb_split);
      } else {
        CARID_ASSERT(mb_split == 2);
      }
      if (mb_split != 0) {
        carid_bits_encode_bit (encoder->bits, mb_common);
      }

      for(k=0;k<4;k+=(4>>mb_split)) {
        for(l=0;l<4;l+=(4>>mb_split)) {
          CaridMotionVector *mv =
            &encoder->motion_vectors[(j+l)*(4*params->x_num_mb) + i + k];

          carid_bits_encode_bits(encoder->bits, 2, mv->pred_mode);
          if (mv->pred_mode == 0) {
            /* FIXME not defined in spec */
            carid_bits_encode_uegol(encoder->bits, mv->dc[0]);
            carid_bits_encode_uegol(encoder->bits, mv->dc[1]);
            carid_bits_encode_uegol(encoder->bits, mv->dc[2]);
          } else {
            carid_bits_encode_segol(encoder->bits, mv->x);
            carid_bits_encode_segol(encoder->bits, mv->y);
          }
        }
      }
    }
  }

  carid_bits_sync (encoder->bits);
}


void
carid_encoder_encode_rap (CaridEncoder *encoder)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 8, 'B');
  carid_bits_encode_bits (encoder->bits, 8, 'B');
  carid_bits_encode_bits (encoder->bits, 8, 'C');
  carid_bits_encode_bits (encoder->bits, 8, 'D');

  carid_bits_encode_bits (encoder->bits, 8, CARID_PARSE_CODE_RAP);

  /* offsets */
  /* FIXME */
  carid_bits_encode_bits (encoder->bits, 24, 0);
  carid_bits_encode_bits (encoder->bits, 24, 0);

  /* rap frame number */
  /* FIXME */
  carid_bits_encode_ue2gol (encoder->bits, 0);

  /* major/minor version */
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 0);

  /* profile */
  carid_bits_encode_uegol (encoder->bits, 0);
  /* level */
  carid_bits_encode_uegol (encoder->bits, 0);


  /* sequence parameters */
  /* video format */
  carid_bits_encode_uegol (encoder->bits, 5);
  /* custom dimensions */
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, encoder->params.width);
  carid_bits_encode_uegol (encoder->bits, encoder->params.height);

  /* chroma format */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* signal range */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* display parameters */
  /* interlace */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* frame rate */
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 24);
  carid_bits_encode_uegol (encoder->bits, 1);

  /* pixel aspect ratio */
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 1);
  carid_bits_encode_uegol (encoder->bits, 1);

  /* clean area flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* colour matrix flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* signal_range flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* colour spec flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* transfer characteristic flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  carid_bits_sync (encoder->bits);
}

void
carid_encoder_encode_frame_header (CaridEncoder *encoder,
    int parse_code)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 8, 'B');
  carid_bits_encode_bits (encoder->bits, 8, 'B');
  carid_bits_encode_bits (encoder->bits, 8, 'C');
  carid_bits_encode_bits (encoder->bits, 8, 'D');
  carid_bits_encode_bits (encoder->bits, 8, parse_code);

  /* offsets */
  /* FIXME */
  carid_bits_encode_bits (encoder->bits, 24, 0);
  carid_bits_encode_bits (encoder->bits, 24, 0);

  /* frame number offset */
  /* FIXME */
  carid_bits_encode_se2gol (encoder->bits, 1);

  /* list */
  carid_bits_encode_uegol (encoder->bits, 0);
  //carid_bits_encode_se2gol (encoder->bits, -1);

  carid_bits_sync (encoder->bits);
}


void
carid_encoder_encode_transform_parameters (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;

  /* transform */
  if (params->wavelet_filter_index == CARID_WAVELET_DAUB97) {
    carid_bits_encode_bit (encoder->bits, 0);
  } else {
    carid_bits_encode_bit (encoder->bits, 1);
    carid_bits_encode_uegol (encoder->bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    carid_bits_encode_bit (encoder->bits, 0);
  } else {
    carid_bits_encode_bit (encoder->bits, 1);
    carid_bits_encode_uegol (encoder->bits, params->transform_depth);
  }

  /* spatial partitioning */
  carid_bits_encode_bit (encoder->bits, 0);

  carid_bits_sync(encoder->bits);
}

void
carid_encoder_init_subbands (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i;
  int w;
  int h;
  int stride;
  int chroma_w;
  int chroma_h;
  int chroma_stride;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = sizeof(int16_t)*(params->iwt_luma_width << params->transform_depth);
  chroma_w = params->iwt_chroma_width >> params->transform_depth;
  chroma_h = params->iwt_chroma_height >> params->transform_depth;
  chroma_stride = sizeof(int16_t)*(params->iwt_chroma_width << params->transform_depth);

  encoder->subbands[0].x = 0;
  encoder->subbands[0].y = 0;
  encoder->subbands[0].w = w;
  encoder->subbands[0].h = h;
  encoder->subbands[0].offset = 0;
  encoder->subbands[0].stride = stride;
  encoder->subbands[0].chroma_w = chroma_w;
  encoder->subbands[0].chroma_h = chroma_h;
  encoder->subbands[0].chroma_offset = 0;
  encoder->subbands[0].chroma_stride = chroma_stride;
  encoder->subbands[0].has_parent = 0;
  encoder->subbands[0].scale_factor_shift = 0;
  encoder->subbands[0].horizontally_oriented = 0;
  encoder->subbands[0].vertically_oriented = 0;
  encoder->subbands[0].quant_index = encoder->encoder_params.quant_index_dc;

  for(i=0; i<params->transform_depth; i++) {
    encoder->subbands[1+3*i].x = 1;
    encoder->subbands[1+3*i].y = 1;
    encoder->subbands[1+3*i].w = w;
    encoder->subbands[1+3*i].h = h;
    encoder->subbands[1+3*i].offset = w + (stride/2/sizeof(int16_t));
    encoder->subbands[1+3*i].stride = stride;
    encoder->subbands[1+3*i].chroma_w = chroma_w;
    encoder->subbands[1+3*i].chroma_h = chroma_h;
    encoder->subbands[1+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    encoder->subbands[1+3*i].chroma_stride = chroma_stride;
    encoder->subbands[1+3*i].has_parent = (i>0);
    encoder->subbands[1+3*i].scale_factor_shift = i;
    encoder->subbands[1+3*i].horizontally_oriented = 0;
    encoder->subbands[1+3*i].vertically_oriented = 0;
    encoder->subbands[1+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    encoder->subbands[2+3*i].x = 0;
    encoder->subbands[2+3*i].y = 1;
    encoder->subbands[2+3*i].w = w;
    encoder->subbands[2+3*i].h = h;
    encoder->subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    encoder->subbands[2+3*i].stride = stride;
    encoder->subbands[2+3*i].chroma_w = chroma_w;
    encoder->subbands[2+3*i].chroma_h = chroma_h;
    encoder->subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    encoder->subbands[2+3*i].chroma_stride = chroma_stride;
    encoder->subbands[2+3*i].has_parent = (i>0);
    encoder->subbands[2+3*i].scale_factor_shift = i;
    encoder->subbands[2+3*i].horizontally_oriented = 0;
    encoder->subbands[2+3*i].vertically_oriented = 1;
    encoder->subbands[2+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    encoder->subbands[3+3*i].x = 1;
    encoder->subbands[3+3*i].y = 0;
    encoder->subbands[3+3*i].w = w;
    encoder->subbands[3+3*i].h = h;
    encoder->subbands[3+3*i].offset = w;
    encoder->subbands[3+3*i].stride = stride;
    encoder->subbands[3+3*i].chroma_w = chroma_w;
    encoder->subbands[3+3*i].chroma_h = chroma_h;
    encoder->subbands[3+3*i].chroma_offset = chroma_w;
    encoder->subbands[3+3*i].chroma_stride = chroma_stride;
    encoder->subbands[3+3*i].has_parent = (i>0);
    encoder->subbands[3+3*i].scale_factor_shift = i;
    encoder->subbands[3+3*i].horizontally_oriented = 1;
    encoder->subbands[3+3*i].vertically_oriented = 0;
    encoder->subbands[3+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
carid_encoder_encode_transform_data (CaridEncoder *encoder, int component)
{
  int i;
  CaridParams *params = &encoder->params;

  carid_encoder_init_subbands (encoder);

  for (i=0;i < 1 + 3*params->transform_depth; i++) {
    carid_encoder_encode_subband (encoder, component, i);
  }
}


void
carid_encoder_encode_subband (CaridEncoder *encoder, int component, int index)
{
  CaridParams *params = &encoder->params;
  CaridSubband *subband = encoder->subbands + index;
  CaridSubband *parent_subband = NULL;
  CaridArith *arith;
  CaridBits *bits;
  int16_t *data;
  int16_t *parent_data = NULL;
  int i,j;
  int quant_factor;
  int quant_offset;
  int scale_factor;
  int subband_zero_flag;
  int ntop;
  int stride;
  int width;
  int height;
  int offset;

  if (component == 0) {
    stride = subband->stride >> 1;
    width = subband->w;
    height = subband->h;
    offset = subband->offset;
  } else {
    stride = subband->chroma_stride >> 1;
    width = subband->chroma_w;
    height = subband->chroma_h;
    offset = subband->chroma_offset;
  }

  CARID_DEBUG("subband index=%d %d x %d at offset %d with stride %d", index,
      width, height, offset, stride);

  data = (int16_t *)encoder->frame->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)encoder->frame->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)encoder->frame->components[component].data +
        parent_subband->chroma_offset;
    }
  }
  quant_factor = carid_table_quant[subband->quant_index];
  quant_offset = carid_table_offset[subband->quant_index];

  scale_factor = 1<<(params->transform_depth - subband->quant_index);
  ntop = (scale_factor>>1) * quant_factor;

  bits = carid_bits_new ();
  carid_bits_encode_init (bits, encoder->subband_buffer);
  arith = carid_arith_new ();
  carid_arith_encode_init (arith, bits);
  carid_arith_init_contexts (arith);

  subband_zero_flag = 1;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      int v = data[j*stride + i];
      int sign;
      int parent_zero;
      int context;
      int context2;
      int nhood_sum;
      int previous_value;
      int sign_context;
      int pred_value;

      nhood_sum = 0;
      if (j>0) {
        nhood_sum += abs(data[(j-1)*stride + i]);
      }
      if (i>0) {
        nhood_sum += abs(data[j*stride + i - 1]);
      }
      if (i>0 && j>0) {
        nhood_sum += abs(data[(j-1)*stride + i - 1]);
      }
//nhood_sum = 0;
      
      if (index == 0) {
        if (j>0) {
          if (i>0) {
            pred_value = (data[j*stride + i - 1] +
                data[(j-1)*stride + i] + data[(j-1)*stride + i - 1])/3;
          } else {
            pred_value = data[(j-1)*stride + i];
          }
        } else {
          if (i>0) {
            pred_value = data[j*stride + i - 1];
          } else {
            pred_value = 0;
          }
        }
      } else {
        pred_value = 0;
      }
//pred_value = 0;

      if (subband->has_parent) {
        if (parent_data[(j>>1)*(stride<<1) + (i>>1)]==0) {
          parent_zero = 1;
        } else {
          parent_zero = 0;
        }
      } else {
        if (subband->x == 0 && subband->y == 0) {
          parent_zero = 0;
        } else {
          parent_zero = 1;
        }
      }
//parent_zero = 0;

      previous_value = 0;
      if (subband->horizontally_oriented) {
        if (i > 0) {
          previous_value = data[j*stride + i - 1];
        }
      } else if (subband->vertically_oriented) {
        if (j > 0) {
          previous_value = data[(j-1)*stride + i];
        }
      }
//previous_value = 0;

      if (parent_zero) {
        if (nhood_sum == 0) {
          context = CARID_CTX_Z_BIN1z;
        } else {
          context = CARID_CTX_Z_BIN1nz;
        }
        context2 = CARID_CTX_Z_BIN2;
      } else {
        if (nhood_sum == 0) {
          context = CARID_CTX_NZ_BIN1z;
        } else {
          if (nhood_sum <= ntop) {
            context = CARID_CTX_NZ_BIN1a;
          } else {
            context = CARID_CTX_NZ_BIN1b;
          }
        }
        context2 = CARID_CTX_NZ_BIN2;
      }

      if (previous_value > 0) {
        sign_context = CARID_CTX_SIGN_POS;
      } else if (previous_value < 0) {
        sign_context = CARID_CTX_SIGN_NEG;
      } else {
        sign_context = CARID_CTX_SIGN_ZERO;
      }
      
      v -= pred_value;
      if (v < 0) {
        sign = 0;
        v = -v;
      } else {
        sign = 1;
      }
      v += quant_factor/2 - quant_offset;
      v /= quant_factor;
      if (v != 0) {
        subband_zero_flag = 0;
      }

      carid_arith_context_encode_uu (arith, context, context2, v);
      if (v) {
        carid_arith_context_encode_bit (arith, sign_context, sign);
      }

      if (v) {
        if (sign) {
          data[j*stride + i] = pred_value + quant_offset + quant_factor * v;
        } else {
          data[j*stride + i] = pred_value - (quant_offset + quant_factor * v);
        }
      } else {
        data[j*stride + i] = pred_value;
      }
      
    }
  }

  carid_arith_flush (arith);
  CARID_DEBUG("encoded %d bits", bits->offset);
  carid_arith_free (arith);
  carid_bits_sync (bits);

  carid_bits_encode_bit (encoder->bits, subband_zero_flag);
  if (!subband_zero_flag) {
    carid_bits_encode_uegol (encoder->bits, subband->quant_index);
    carid_bits_encode_uegol (encoder->bits, bits->offset/8);

    carid_bits_sync (encoder->bits);

    carid_bits_append (encoder->bits, bits);
  } else {
    CARID_DEBUG ("subband is zero");
    carid_bits_sync (encoder->bits);
  }

  carid_bits_free (bits);
}


void
carid_encoder_motion_predict (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i;
  int j;
  CaridFrame *ref_frame;
  CaridFrame *frame;
  int sum_pred_x;
  int sum_pred_y;
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
  double sum_x, sum_y;

#define DIVIDE_ROUND_UP(a,b) (((a) + (b) - 1)/(b))
  params->x_num_mb =
    DIVIDE_ROUND_UP(encoder->params.width, 4*params->xbsep_luma);
  params->y_num_mb =
    DIVIDE_ROUND_UP(encoder->params.height, 4*params->ybsep_luma);

  if (encoder->motion_vectors == NULL) {
    encoder->motion_vectors = malloc(sizeof(CaridMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }

  ref_frame = encoder->ref_frame0;
  if (!ref_frame) {
    CARID_ERROR("no reference frame");
  }
  frame = encoder->encode_frame;

  sum_pred_x = 0;
  sum_pred_y = 0;
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      CaridMotionVector *mv =
        &encoder->motion_vectors[j*(4*params->x_num_mb) + i];

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      predict_dc (mv, frame, x, y, params->xbsep_luma, params->ybsep_luma);

      predict_motion (mv, frame, ref_frame, x, y,
          params->xbsep_luma, params->ybsep_luma);

      if (mv->pred_mode != 0) {
        sum_pred_x += mv->x;
        sum_pred_y += mv->y;
      }
    }
  }

  pan_x = ((double)sum_pred_x)/(16*params->x_num_mb*params->y_num_mb);
  pan_y = ((double)sum_pred_y)/(16*params->x_num_mb*params->y_num_mb);

  mag_x = 0;
  mag_y = 0;
  skew_x = 0;
  skew_y = 0;
  sum_x = 0;
  sum_y = 0;
  for(j=0;j<4*params->y_num_mb;j++) {
    for(i=0;i<4*params->x_num_mb;i++) {
      double x;
      double y;

      x = i*params->xbsep_luma - (2*params->x_num_mb - 0.5);
      y = j*params->ybsep_luma - (2*params->y_num_mb - 0.5);

      mag_x += encoder->motion_vectors[j*(4*params->x_num_mb) + i].x * x;
      mag_y += encoder->motion_vectors[j*(4*params->x_num_mb) + i].y * y;

      skew_x += encoder->motion_vectors[j*(4*params->x_num_mb) + i].x * y;
      skew_y += encoder->motion_vectors[j*(4*params->x_num_mb) + i].y * x;

      sum_x += x * x;
      sum_y += y * y;
    }
  }
  mag_x = mag_x/sum_x;
  mag_y = mag_y/sum_y;
  skew_x = skew_x/sum_x;
  skew_y = skew_y/sum_y;

  CARID_ERROR("pan %6.3f %6.3f mag %6.3f %6.3f skew %6.3f %6.3f",
      pan_x, pan_y, mag_x, mag_y, skew_x, skew_y);

}

#if 0
static int
calculate_metric (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  return metric;
}
#endif

static int
calculate_metric2 (CaridFrame *frame1, int x1, int y1,
    CaridFrame *frame2, int x2, int y2, int width, int height)
{
  int i;
  int j;
  int metric = 0;
  uint8_t *a;
  int a_stride;
  uint8_t *b;
  int b_stride;

  a_stride = frame1->components[0].stride;
  a = frame1->components[0].data + x1 + y1 * a_stride;
  b_stride = frame2->components[0].stride;
  b = frame2->components[0].data + x2 + y2 * b_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  width/=2;
  height/=2;
  x1/=2;
  y1/=2;
  x2/=2;
  y2/=2;

  a_stride = frame1->components[1].stride;
  a = frame1->components[1].data + x1 + y1 * a_stride;
  b_stride = frame2->components[1].stride;
  b = frame2->components[1].data + x2 + y2 * b_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  a_stride = frame1->components[2].stride;
  a = frame1->components[2].data + x1 + y1 * a_stride;
  b_stride = frame2->components[2].stride;
  b = frame2->components[2].data + x2 + y2 * b_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  return metric;
}

static void
predict_motion_search (CaridMotionVector *mv, CaridFrame *frame,
    CaridFrame *reference_frame, int x, int y, int w, int h)
{
  int dx, dy;
  //uint8_t *data = frame->components[0].data;
  //int stride = frame->components[0].stride;
  //uint8_t *ref_data = reference_frame->components[0].data;
  //int ref_stride = reference_frame->components[0].stride;
  int metric;
  int min_metric;
  int step_size;

#if 0
  min_metric = calculate_metric (data + y * stride + x, stride,
      ref_data + y * ref_stride + x, ref_stride, w, h);
  *pred_x = 0;
  *pred_y = 0;

  printf("mp %d %d metric %d\n", x, y, min_metric);
#endif

  dx = 0;
  dy = 0;
  step_size = 4;
  while (step_size > 0) {
    static const int hx[5] = { 0, 0, -1, 0, 1 };
    static const int hy[5] = { 0, -1, 0, 1, 0 };
    int px, py;
    int min_index;
    int i;

    min_index = 0;
    min_metric = calculate_metric2 (frame, x, y, reference_frame, x+dx, y+dy,
        w, h);
    for(i=1;i<5;i++){
      px = x + dx + hx[i] * step_size;
      py = y + dy + hy[i] * step_size;
      if (px < 0 || py < 0 || 
          px + w > reference_frame->components[0].width ||
          py + h > reference_frame->components[0].height) {
        continue;
      }

      metric = calculate_metric2 (frame, x, y, reference_frame, px, py,
          w, h);

      if (metric < min_metric) {
        min_metric = metric;
        min_index = i;
      }
    }

    if (min_index == 0) {
      step_size >>= 1;
    } else {
      dx += hx[min_index] * step_size;
      dy += hy[min_index] * step_size;
    }
  }
  if (min_metric < mv->metric) {
    mv->x = dx;
    mv->y = dy;
    mv->metric = min_metric;
    mv->pred_mode = 1;
  }
}

static void
predict_motion_scan (CaridMotionVector *mv, CaridFrame *frame,
    CaridFrame *reference_frame, int x, int y, int w, int h)
{
  int dx,dy;
  int metric;

  for(dy = -4; dy <= 4; dy++) {
    for(dx = -4; dx <= 4; dx++) {
      if (y + dy < 0) continue;
      if (x + dx < 0) continue;
      if (y + dy + h > reference_frame->components[0].height) continue;
      if (x + dx + w > reference_frame->components[0].width) continue;

      metric = calculate_metric2 (frame, x, y, reference_frame,
          x + dx, y + dy, w, h);

      if (metric < mv->metric) {
        mv->metric = metric;
        mv->x = dx;
        mv->y = dy;
        mv->pred_mode = 1;
      }

    }
  }
}

static void
predict_motion_none (CaridMotionVector *mv, CaridFrame *frame,
    CaridFrame *reference_frame, int x, int y, int w, int h)
{
  int metric;

  metric = calculate_metric2 (frame, x, y, reference_frame, x, y, w, h);
  if (metric < mv->metric) {
    mv->x = 0;
    mv->y = 0;
    mv->metric = metric;
    mv->pred_mode = 1;
  }
}

static void
predict_motion (CaridMotionVector *mv, CaridFrame *frame,
    CaridFrame *reference_frame, int x, int y, int w, int h)
{
  int how = 1;

  switch(how) {
    case 0:
      predict_motion_scan (mv, frame, reference_frame, x, y, w, h);
      break;
    case 1:
      predict_motion_search (mv, frame, reference_frame, x, y, w, h);
      break;
    case 2:
      predict_motion_none (mv, frame, reference_frame, x, y, w, h);
      break;
  }
}

static void
predict_dc (CaridMotionVector *mv, CaridFrame *frame, int x, int y,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;
  uint8_t *a;
  int a_stride;
  int sum;

  a_stride = frame->components[0].stride;
  a = frame->components[0].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[0] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[0]);
    }
  }

  width/=2;
  height/=2;
  x/=2;
  y/=2;

  a_stride = frame->components[1].stride;
  a = frame->components[1].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[1] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[1]);
    }
  }

  a_stride = frame->components[2].stride;
  a = frame->components[2].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[2] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[2]);
    }
  }

  mv->pred_mode = 0;
  mv->metric = metric;
}


/* frame queue */

static void
carid_encoder_frame_queue_push (CaridEncoder *encoder, CaridFrame *frame)
{
  encoder->frame_queue[encoder->frame_queue_length] = frame;
  encoder->frame_queue_length++;
  CARID_ASSERT(encoder->frame_queue_length < 10);
}

static CaridFrame *
carid_encoder_frame_queue_get (CaridEncoder *encoder, int frame_index)
{
  int i;
  for(i=0;i<encoder->frame_queue_length;i++){
    if (encoder->frame_queue[i]->frame_number == frame_index) {
      return encoder->frame_queue[i];
    }
  }
  return NULL;
}

static void
carid_encoder_frame_queue_remove (CaridEncoder *encoder, int frame_index)
{
  int i;
  for(i=0;i<encoder->frame_queue_length;i++){
    if (encoder->frame_queue[i]->frame_number == frame_index) {
      memmove (encoder->frame_queue + i, encoder->frame_queue + i + 1,
          sizeof(CaridFrame *)*(encoder->frame_queue_length - i - 1));
      encoder->frame_queue_length--;
      return;
    }
  }
}


/* reference pool */

static void
carid_encoder_reference_add (CaridEncoder *encoder, CaridFrame *frame)
{
  CARID_DEBUG("adding %d", frame->frame_number);
  encoder->reference_frames[encoder->n_reference_frames] = frame;
  encoder->n_reference_frames++;
  CARID_ASSERT(encoder->n_reference_frames < 10);
}

static CaridFrame *
carid_encoder_reference_get (CaridEncoder *encoder, int frame_number)
{
  int i;
  CARID_DEBUG("getting %d", frame_number);
  for(i=0;i<encoder->n_reference_frames;i++){
    if (encoder->reference_frames[i]->frame_number == frame_number) {
      return encoder->reference_frames[i];
    }
  }
  return NULL;

}

static void
carid_encoder_reference_retire (CaridEncoder *encoder, int frame_number)
{
  int i;
  CARID_DEBUG("retiring %d", frame_number);
  for(i=0;i<encoder->n_reference_frames;i++){
    if (encoder->reference_frames[i]->frame_number == frame_number) {
      carid_frame_free (encoder->reference_frames[i]);
      memmove (encoder->reference_frames + i, encoder->reference_frames + i + 1,
          sizeof(CaridFrame *)*(encoder->n_reference_frames - i - 1));
      encoder->n_reference_frames--;
      return;
    }
  }
}

