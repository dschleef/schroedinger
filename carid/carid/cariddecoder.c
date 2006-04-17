
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static void carid_decoder_decode_macroblock(CaridDecoder *decoder, int i,
    int j);
static void carid_decoder_decode_prediction_unit(CaridDecoder *decoder,
    CaridMotionVector *mv);
static void carid_decoder_predict (CaridDecoder *decoder);


CaridDecoder *
carid_decoder_new (void)
{
  CaridDecoder *decoder;
  CaridParams *params;

  decoder = malloc(sizeof(CaridDecoder));
  memset (decoder, 0, sizeof(CaridDecoder));

  decoder->tmpbuf = malloc(1024 * 2);
  decoder->tmpbuf2 = malloc(1024 * 2);

  params = &decoder->params;

  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;

  return decoder;
}

void
carid_decoder_free (CaridDecoder *decoder)
{
  if (decoder->frame) {
    carid_frame_free (decoder->frame);
  }
  if (decoder->output_frame) {
    carid_frame_free (decoder->output_frame);
  }

  free (decoder->tmpbuf);
  free (decoder->tmpbuf2);
  free (decoder);
}

void
carid_decoder_set_output_frame (CaridDecoder *decoder, CaridFrame *frame)
{
  if (decoder->output_frame) {
    carid_frame_free (decoder->output_frame);
  }
  decoder->output_frame = frame;
}

int
carid_decoder_is_parse_header (CaridBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  return 1;
}

int
carid_decoder_is_rap (CaridBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (data[4] == CARID_PARSE_CODE_RAP) return 1;

  return 0;
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
carid_decoder_decode (CaridDecoder *decoder, CaridBuffer *buffer)
{
  CaridParams *params = &decoder->params;
  
  decoder->bits = carid_bits_new ();
  carid_bits_decode_init (decoder->bits, buffer);

  if (carid_decoder_is_rap (buffer)) {
    carid_decoder_decode_parse_header(decoder);
    carid_decoder_decode_rap(decoder);

    carid_buffer_unref (buffer);
    carid_bits_free (decoder->bits);
    return;
  }

  carid_decoder_decode_parse_header(decoder);
  carid_decoder_decode_frame_header(decoder);

  if (decoder->code == CARID_PARSE_CODE_INTRA_REF) {
    CARID_ERROR("intra ref");
    carid_decoder_decode_transform_parameters (decoder);

    if (decoder->frame == NULL) {
      decoder->frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_S16,
          params->iwt_luma_width, params->iwt_luma_height, 2, 2);
    }

    carid_decoder_decode_transform_data (decoder, 0);
    carid_decoder_iwt_transform (decoder, 0);

    carid_decoder_decode_transform_data (decoder, 1);
    carid_decoder_iwt_transform (decoder, 1);

    carid_decoder_decode_transform_data (decoder, 2);
    carid_decoder_iwt_transform (decoder, 2);

    carid_frame_convert (decoder->output_frame, decoder->frame);

    if (decoder->reference_frames[0] == NULL) {
      decoder->reference_frames[0] = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_U8,
          params->width, params->height, 2, 2);
    }

    carid_frame_convert (decoder->reference_frames[0], decoder->frame);
  } else if (decoder->code == CARID_PARSE_CODE_INTER_NON_REF) {
    CARID_ERROR("inter non-ref");
    carid_decoder_decode_frame_prediction (decoder);

    /* FIXME */
    CARID_ASSERT(params->xbsep_luma == 8);
    CARID_ASSERT(params->ybsep_luma == 8);

    params->x_num_mb = decoder->params.width / (4*params->xbsep_luma);
    params->y_num_mb = decoder->params.height / (4*params->xbsep_luma);

    /* FIXME: This should be kept from frame to frame */
    decoder->motion_vectors = malloc(sizeof(CaridMotionVector) * 16 *
        params->x_num_mb * params->y_num_mb);

    carid_decoder_decode_prediction_data (decoder);
    carid_decoder_predict (decoder);

    free(decoder->motion_vectors);
  }

  carid_buffer_unref (buffer);
  carid_bits_free (decoder->bits);
}

#if 0
void
carid_decoder_copy_from_frame_buffer (CaridDecoder *decoder, CaridBuffer *buffer)
{
  CaridParams *params = &decoder->params;
  int i;
  uint8_t *data;
  int16_t *frame_data;

  data = buffer->data;

  frame_data = (int16_t *)decoder->frame_buffer[0]->data;
  for(i=0;i<params->height;i++){
    oil_convert_u8_s16 (data, frame_data, params->width);
    data += params->width;
    frame_data += params->iwt_luma_width;
  }

  frame_data = (int16_t *)decoder->frame_buffer[1]->data;
  for(i=0;i<params->height/2;i++){
    oil_convert_u8_s16 (data, frame_data, params->width/2);
    data += params->width/2;
    frame_data += params->iwt_chroma_width;
  }

  frame_data = (int16_t *)decoder->frame_buffer[2]->data;
  for(i=0;i<params->height/2;i++){
    oil_convert_u8_s16 (data, frame_data, params->width/2);
    data += params->width/2;
    frame_data += params->iwt_chroma_width;
  }
}
#endif

void
carid_decoder_iwt_transform (CaridDecoder *decoder, int component)
{
  CaridParams *params = &decoder->params;
  int16_t *frame_data;
  int height;
  int width;
  int level;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }

  frame_data = (int16_t *)decoder->frame->components[component].data;
  for(level=params->transform_depth-1;level>=0;level--) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = 2*(width << level);

    carid_wavelet_inverse_transform_2d (params->wavelet_filter_index,
        frame_data, stride, w, h, decoder->tmpbuf);
  }

}

void
carid_decoder_decode_parse_header (CaridDecoder *decoder)
{
  int v1, v2, v3, v4;
  
  v1 = carid_bits_decode_bits (decoder->bits, 8);
  v2 = carid_bits_decode_bits (decoder->bits, 8);
  v3 = carid_bits_decode_bits (decoder->bits, 8);
  v4 = carid_bits_decode_bits (decoder->bits, 8);
  CARID_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    CARID_ERROR ("expected parse header");
    return;
  }


  decoder->code = carid_bits_decode_bits (decoder->bits, 8);
  CARID_DEBUG ("parse code %02x", decoder->code);

  decoder->next_parse_offset = carid_bits_decode_bits (decoder->bits, 24);
  CARID_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = carid_bits_decode_bits (decoder->bits, 24);
  CARID_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

void
carid_decoder_decode_rap (CaridDecoder *decoder)
{
  int bit;
  int index;

  CARID_DEBUG("decoding RAP");
  /* parse parameters */
  decoder->rap_frame_number = carid_bits_decode_ue2gol (decoder->bits);
  CARID_DEBUG("rap frame number = %d", decoder->rap_frame_number);
  decoder->params.major_version = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("major_version = %d", decoder->params.major_version);
  decoder->params.minor_version = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("minor_version = %d", decoder->params.minor_version);
  decoder->params.profile = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("profile = %d", decoder->params.profile);
  decoder->params.level = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("level = %d", decoder->params.level);

  /* sequence parameters */
  index = carid_bits_decode_uegol (decoder->bits);
  carid_params_set_video_format (&decoder->params, index);

  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.width = carid_bits_decode_uegol (decoder->bits);
    decoder->params.height = carid_bits_decode_uegol (decoder->bits);
  }
  CARID_DEBUG("size = %d x %d", decoder->params.width,
      decoder->params.height);

  /* chroma format */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.chroma_format_index =
      carid_bits_decode_bits (decoder->bits, 3);
  } else {
    decoder->params.chroma_format_index = 0;
  }
  CARID_DEBUG("chroma_format_index %d",
      decoder->params.chroma_format_index);
  decoder->params.chroma_width = (decoder->params.width + 1)/2;
  decoder->params.chroma_height = (decoder->params.height + 1)/2;

  /* signal range */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    index = carid_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.luma_offset = carid_bits_decode_uegol (decoder->bits);
      decoder->params.luma_excursion = carid_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_offset = carid_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_excursion =
        carid_bits_decode_uegol (decoder->bits);
    } else {
      /* FIXME */
      //carid_params_set_excursion (&decoder->params, index);
    }
  }
  CARID_DEBUG("luma offset %d excursion %d",
      decoder->params.luma_offset, decoder->params.luma_excursion);

  /* display parameters */
  /* interlace */
  decoder->params.interlace = carid_bits_decode_bit (decoder->bits);
  if (decoder->params.interlace) {
    decoder->params.top_field_first = carid_bits_decode_bit (decoder->bits);
  }
  CARID_DEBUG("interlace %d top_field_first %d",
      decoder->params.interlace, decoder->params.top_field_first);

  /* frame rate */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    index = carid_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.frame_rate_numerator = carid_bits_decode_uegol (decoder->bits);
      decoder->params.frame_rate_denominator = carid_bits_decode_uegol (decoder->bits);
    } else {
      carid_params_set_frame_rate (&decoder->params, index);
    }
  }
  CARID_DEBUG("frame rate %d/%d", decoder->params.frame_rate_numerator,
      decoder->params.frame_rate_denominator);

  /* pixel aspect ratio */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    index = carid_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.pixel_aspect_ratio_numerator =
        carid_bits_decode_uegol (decoder->bits);
      decoder->params.pixel_aspect_ratio_denominator =
        carid_bits_decode_uegol (decoder->bits);
    } else {
      carid_params_set_pixel_aspect_ratio (&decoder->params, index);
    }
  }
  CARID_DEBUG("aspect ratio %d/%d",
      decoder->params.pixel_aspect_ratio_numerator,
      decoder->params.pixel_aspect_ratio_denominator);

  /* clean area */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.clean_tl_x = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_tl_y = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_width = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_height = carid_bits_decode_uegol (decoder->bits);
  }
  CARID_DEBUG("clean offset %d %d", decoder->params.clean_tl_x,
      decoder->params.clean_tl_y);
  CARID_DEBUG("clean size %d %d", decoder->params.clean_width,
      decoder->params.clean_height);

  /* colur matrix */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }

  /* signal range */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }

  /* colour spec */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }
  CARID_DEBUG("bit %d",bit);

  /* transfer characteristic */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }
  CARID_DEBUG("bit %d",bit);

  carid_bits_sync (decoder->bits);
}

void
carid_decoder_decode_frame_header (CaridDecoder *decoder)
{
  int n;
  int i;

  decoder->frame_number_offset = carid_bits_decode_se2gol (decoder->bits);
  CARID_DEBUG("frame number offset %d", decoder->frame_number_offset);

  n = carid_bits_decode_uegol (decoder->bits);
  for(i=0;i<n;i++){
    carid_bits_decode_se2gol (decoder->bits);
    CARID_DEBUG("retire %d", decoder->frame_number_offset);
  }

  carid_bits_sync (decoder->bits);
}

void
carid_decoder_decode_frame_prediction (CaridDecoder *decoder)
{
  CaridParams *params = &decoder->params;
  int bit;
  int length;
  int index;

  /* block params flag */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    index = carid_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      params->xblen_luma = carid_bits_decode_uegol (decoder->bits);
      params->yblen_luma = carid_bits_decode_uegol (decoder->bits);
      params->xbsep_luma = carid_bits_decode_uegol (decoder->bits);
      params->ybsep_luma = carid_bits_decode_uegol (decoder->bits);
    } else {
      carid_params_set_block_params (params, index);
    }
  }
  CARID_ERROR("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  /* mv precision flag */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    params->mv_precision = carid_bits_decode_uegol (decoder->bits);
  }
  CARID_ERROR("mv_precision %d", params->mv_precision);

  /* global motion flag */
  params->global_motion = carid_bits_decode_bit (decoder->bits);
  if (params->global_motion) {
    int i;

    params->global_only_flag = carid_bits_decode_bit (decoder->bits);
    params->global_prec_bits = carid_bits_decode_uegol (decoder->bits);

    for (i=0;i<params->num_refs;i++) {
      /* pan */
      bit = carid_bits_decode_bit (decoder->bits);
      if (bit) {
        params->b_1[i] = carid_bits_decode_segol (decoder->bits);
        params->b_2[i] = carid_bits_decode_segol (decoder->bits);
      } else {
        params->b_1[i] = 0;
        params->b_2[i] = 0;
      }

      /* matrix */
      bit = carid_bits_decode_bit (decoder->bits);
      if (bit) {
        params->a_11[i] = carid_bits_decode_segol (decoder->bits);
        params->a_12[i] = carid_bits_decode_segol (decoder->bits);
        params->a_21[i] = carid_bits_decode_segol (decoder->bits);
        params->a_22[i] = carid_bits_decode_segol (decoder->bits);
      } else {
        params->a_11[i] = 1;
        params->a_12[i] = 0;
        params->a_21[i] = 0;
        params->a_22[i] = 1;
      }

      /* perspective */
      bit = carid_bits_decode_bit (decoder->bits);
      if (bit) {
        params->c_1[i] = carid_bits_decode_segol (decoder->bits);
        params->c_2[i] = carid_bits_decode_segol (decoder->bits);
      } else {
        params->c_1[i] = 0;
        params->c_2[i] = 0;
      }

      CARID_ERROR("ref %d pan %d %d matrix %d %d %d %d perspective %d %d",
          i, params->b_1[i], params->b_2[i],
          params->a_11[i], params->a_12[i],
          params->a_21[i], params->a_22[i],
          params->c_1[i], params->c_2[i]);
    }
  }

  /* block data length */
  length = carid_bits_decode_uegol (decoder->bits);
  CARID_ERROR("length %d", length);

  carid_bits_sync (decoder->bits);
}

static void
copy_block_4x4 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<4;j++){
    *(uint32_t *)(dest + dstr*j) = *(uint32_t *)(src + sstr*j);
  }
}

static void
copy_block_8x8 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<8;j++){
    *(uint64_t *)(dest + dstr*j) = *(uint64_t *)(src + sstr*j);
  }
}

void
copy_block (uint8_t *dest, int dstr, uint8_t *src, int sstr, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = src[sstr*j+i];
    }
  }
}

void
carid_decoder_decode_prediction_data (CaridDecoder *decoder)
{
  CaridParams *params = &decoder->params;
  int i, j;

  for(j=0;j<4*params->y_num_mb;j+=4){
    for(i=0;i<4*params->x_num_mb;i+=4){
      carid_decoder_decode_macroblock(decoder, i, j);
    }
  }
}

static void
carid_decoder_predict (CaridDecoder *decoder)
{
  CaridParams *params = &decoder->params;
  CaridFrame *frame = decoder->output_frame;
  CaridFrame *reference_frame = decoder->reference_frames[0];
  int i, j;
  int dx, dy;
  int x, y;
  uint8_t *data;
  int stride;
  uint8_t *ref_data;
  int ref_stride;

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      CaridMotionVector *mv = &decoder->motion_vectors[j*4*params->x_num_mb + i];

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      dx = mv->x;
      dy = mv->y;

      data = frame->components[0].data;
      stride = frame->components[0].stride;
      ref_data = reference_frame->components[0].data;
      ref_stride = reference_frame->components[0].stride;
      copy_block_8x8 (data + y * stride + x, stride,
          ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

      x /= 2;
      dx /= 2;
      y /= 2;
      dy /= 2;

      data = frame->components[1].data;
      stride = frame->components[1].stride;
      ref_data = reference_frame->components[1].data;
      ref_stride = reference_frame->components[1].stride;
      copy_block_4x4 (data + y * stride + x, stride,
          ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

      data = frame->components[2].data;
      stride = frame->components[2].stride;
      ref_data = reference_frame->components[2].data;
      ref_stride = reference_frame->components[2].stride;
      copy_block_4x4 (data + y * stride + x, stride,
          ref_data + (y+dy) * ref_stride + x + dx, ref_stride);
    }
  }
}

static void
carid_decoder_decode_macroblock(CaridDecoder *decoder, int i, int j)
{
  CaridParams *params = &decoder->params;
  CaridMotionVector *mv = &decoder->motion_vectors[j*4*params->x_num_mb + i];
  int k,l;
  int mask;

  //CARID_ERROR("global motion %d", params->global_motion);
  if (params->global_motion) {
    mv->mb_using_global = carid_bits_decode_bit (decoder->bits);
  } else {
    mv->mb_using_global = FALSE;
  }
  if (!mv->mb_using_global) {
    mv->mb_split = carid_bits_decode_bits (decoder->bits, 2);
    CARID_ASSERT(mv->mb_split != 3);
  } else {
    mv->mb_split = 2;
  }
  if (mv->mb_split != 0) {
    mv->mb_common = carid_bits_decode_bit (decoder->bits);
  } else {
    mv->mb_common = FALSE;
  }
  //CARID_ERROR("mb_using_global=%d mb_split=%d mb_common=%d",
  //    mv->mb_using_global, mv->mb_split, mv->mb_common);

  mask = 3 >> mv->mb_split;
  for (k=0;k<4;k++) {
    for (l=0;l<4;l++) {
      CaridMotionVector *bv =
        &decoder->motion_vectors[(j+l)*4*params->x_num_mb + (i+k)];

      if ((k&mask) == 0 && (l&mask) == 0) {
        //CARID_ERROR("decoding PU %d %d", j+l, i+k);

        carid_decoder_decode_prediction_unit (decoder, bv);
      } else {
        CaridMotionVector *bv1;
       
        bv1 = &decoder->motion_vectors[(j+(l&(~mask)))*4*params->x_num_mb +
          (i+(k&(~mask)))];

        *bv = *bv1;
      }
    }
  }
}

static void
carid_decoder_decode_prediction_unit(CaridDecoder *decoder,
    CaridMotionVector *mv)
{
  /* FIXME */

  //int pred_mode = 1;

  mv->x = carid_bits_decode_segol (decoder->bits);
  mv->y = carid_bits_decode_segol (decoder->bits);
}

void
carid_decoder_decode_transform_parameters (CaridDecoder *decoder)
{
  int bit;
  CaridParams *params = &decoder->params;

  /* transform */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    params->wavelet_filter_index = carid_bits_decode_uegol (decoder->bits);
    if (params->wavelet_filter_index > 4) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->wavelet_filter_index = CARID_WAVELET_DAUB97;
  }
  CARID_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    params->transform_depth = carid_bits_decode_uegol (decoder->bits);
    if (params->transform_depth > 6) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->transform_depth = 4;
  }
  CARID_DEBUG ("transform depth %d", params->transform_depth);

  /* spatial partitioning */
  params->spatial_partition = carid_bits_decode_bit (decoder->bits);
  CARID_DEBUG ("spatial_partitioning %d", params->spatial_partition);
  if (params->spatial_partition) {
    params->partition_index = carid_bits_decode_uegol (decoder->bits);
    if (params->partition_index > 1) {
      /* FIXME: ? */
      params->non_spec_input = TRUE;
    }
    if (params->partition_index == 0) {
      params->max_xblocks = carid_bits_decode_uegol (decoder->bits);
      params->max_yblocks = carid_bits_decode_uegol (decoder->bits);
    }
    params->multi_quant = carid_bits_decode_bit (decoder->bits);
  }

  carid_params_calculate_iwt_sizes (params);

  carid_bits_sync(decoder->bits);
}

void
carid_decoder_init_subbands (CaridDecoder *decoder)
{
  CaridParams *params = &decoder->params;
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

  decoder->subbands[0].x = 0;
  decoder->subbands[0].y = 0;
  decoder->subbands[0].w = w;
  decoder->subbands[0].h = h;
  decoder->subbands[0].offset = 0;
  decoder->subbands[0].stride = stride;
  decoder->subbands[0].chroma_w = chroma_w;
  decoder->subbands[0].chroma_h = chroma_h;
  decoder->subbands[0].chroma_offset = 0;
  decoder->subbands[0].chroma_stride = chroma_stride;
  decoder->subbands[0].has_parent = 0;
  decoder->subbands[0].scale_factor_shift = 0;
  decoder->subbands[0].horizontally_oriented = 0;
  decoder->subbands[0].vertically_oriented = 0;

  for(i=0; i<params->transform_depth; i++) {
    decoder->subbands[1+3*i].x = 1;
    decoder->subbands[1+3*i].y = 1;
    decoder->subbands[1+3*i].w = w;
    decoder->subbands[1+3*i].h = h;
    decoder->subbands[1+3*i].offset = w + (stride/2/sizeof(int16_t));
    decoder->subbands[1+3*i].stride = stride;
    decoder->subbands[1+3*i].chroma_w = chroma_w;
    decoder->subbands[1+3*i].chroma_h = chroma_h;
    decoder->subbands[1+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    decoder->subbands[1+3*i].chroma_stride = chroma_stride;
    decoder->subbands[1+3*i].has_parent = (i>0);
    decoder->subbands[1+3*i].scale_factor_shift = i;
    decoder->subbands[1+3*i].horizontally_oriented = 0;
    decoder->subbands[1+3*i].vertically_oriented = 0;

    decoder->subbands[2+3*i].x = 0;
    decoder->subbands[2+3*i].y = 1;
    decoder->subbands[2+3*i].w = w;
    decoder->subbands[2+3*i].h = h;
    decoder->subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    decoder->subbands[2+3*i].stride = stride;
    decoder->subbands[2+3*i].chroma_w = chroma_w;
    decoder->subbands[2+3*i].chroma_h = chroma_h;
    decoder->subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    decoder->subbands[2+3*i].chroma_stride = chroma_stride;
    decoder->subbands[2+3*i].has_parent = (i>0);
    decoder->subbands[2+3*i].scale_factor_shift = i;
    decoder->subbands[2+3*i].horizontally_oriented = 0;
    decoder->subbands[2+3*i].vertically_oriented = 1;

    decoder->subbands[3+3*i].x = 1;
    decoder->subbands[3+3*i].y = 0;
    decoder->subbands[3+3*i].w = w;
    decoder->subbands[3+3*i].h = h;
    decoder->subbands[3+3*i].offset = w;
    decoder->subbands[3+3*i].stride = stride;
    decoder->subbands[3+3*i].chroma_w = chroma_w;
    decoder->subbands[3+3*i].chroma_h = chroma_h;
    decoder->subbands[3+3*i].chroma_offset = chroma_w;
    decoder->subbands[3+3*i].chroma_stride = chroma_stride;
    decoder->subbands[3+3*i].has_parent = (i>0);
    decoder->subbands[3+3*i].scale_factor_shift = i;
    decoder->subbands[3+3*i].horizontally_oriented = 1;
    decoder->subbands[3+3*i].vertically_oriented = 0;

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
carid_decoder_decode_transform_data (CaridDecoder *decoder, int component)
{
  int i;
  CaridParams *params = &decoder->params;

#if 1
  for (i=0;i<3;i++){
    uint8_t value = 0;
    oil_splat_u8_ns (decoder->frame->components[component].data, &value,
        decoder->frame->components[component].length);
  }
#endif

  carid_decoder_init_subbands (decoder);

  for(i=0;i<1+3*params->transform_depth;i++) {
    carid_decoder_decode_subband (decoder, component, i);
  }
}

void
carid_decoder_decode_subband (CaridDecoder *decoder, int component, int index)
{
  CaridParams *params = &decoder->params;
  CaridSubband *subband = decoder->subbands + index;
  CaridSubband *parent_subband = NULL;
  int subband_zero_flag;
  int16_t *data;
  int16_t *parent_data = NULL;
  int quant_index = 0;
  int quant_factor = 0;
  int quant_offset = 0;
  int subband_length;
  int scale_factor;
  int ntop;
  int height;
  int width;
  int stride;
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

  CARID_DEBUG("subband index=%d %d x %d at offset %d stride=%d",
      index, width, height, offset, stride);

  data = (int16_t *)decoder->frame->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband-3;
    if (component == 0) {
      parent_data = (int16_t *)decoder->frame->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)decoder->frame->components[component].data +
        parent_subband->chroma_offset;
    }
  }

  scale_factor = 1<<(params->transform_depth - quant_index);
  ntop = (scale_factor>>1) * quant_factor;

  subband_zero_flag = carid_bits_decode_bit (decoder->bits);
  if (!subband_zero_flag) {
    int i,j;
    CaridBuffer *buffer;
    CaridBits *bits;
    CaridArith *arith;

    quant_index = carid_bits_decode_uegol (decoder->bits);
    CARID_DEBUG("quant index %d", quant_index);
    if ((unsigned int)quant_index > 60) {
      CARID_ERROR("quant_index too big (%u > 60)", quant_index);
      params->non_spec_input = TRUE;
      return;
    }
    quant_factor = carid_table_quant[quant_index];
    quant_offset = carid_table_offset[quant_index];
    CARID_DEBUG("quant factor %d offset %d", quant_factor, quant_offset);

    subband_length = carid_bits_decode_uegol (decoder->bits);
    CARID_DEBUG("subband length %d", subband_length);

    carid_bits_sync (decoder->bits);

    buffer = carid_buffer_new_subbuffer (decoder->bits->buffer,
        decoder->bits->offset>>3, subband_length);
    bits = carid_bits_new ();
    carid_bits_decode_init (bits, buffer);

    arith = carid_arith_new ();
    carid_arith_decode_init (arith, bits);
    carid_arith_init_contexts (arith);

    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int sign = 0;
        int v;
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

        v = carid_arith_context_decode_uu (arith, context, context2);
        if (v) {
          sign = carid_arith_context_decode_bit (arith, sign_context);
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
    carid_arith_free (arith);
    CARID_DEBUG("decoded %d bits", bits->offset);
    carid_bits_free (bits);
    carid_buffer_unref (buffer);

    decoder->bits->offset += subband_length * 8;
  } else {
    int i,j;

    CARID_DEBUG("subband is zero");
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        data[j*stride + i] = 0;
      }
    }

    carid_bits_sync (decoder->bits);
  }
}



