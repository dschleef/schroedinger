
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void schro_encoder_create_picture_list (SchroEncoder *encoder);

static void schro_encoder_frame_queue_push (SchroEncoder *encoder,
    SchroFrame *frame);
static SchroFrame * schro_encoder_frame_queue_get (SchroEncoder *encoder,
    int frame_number);
static void schro_encoder_frame_queue_remove (SchroEncoder *encoder,
    int frame_number);
static void schro_encoder_reference_add (SchroEncoder *encoder,
    SchroFrame *frame);
static SchroFrame * schro_encoder_reference_get (SchroEncoder *encoder,
    int frame_number);
static void schro_encoder_reference_retire (SchroEncoder *encoder,
    int frame_number);



SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;
  SchroParams *params;

  encoder = malloc(sizeof(SchroEncoder));
  memset (encoder, 0, sizeof(SchroEncoder));

  encoder->tmpbuf = malloc(1024 * 2);
  encoder->tmpbuf2 = malloc(1024 * 2);

  encoder->subband_buffer = schro_buffer_new_and_alloc (1000000);

  params = &encoder->params;
  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;
  params->transform_depth = 4;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->wavelet_filter_index = SCHRO_WAVELET_5_3;

  schro_params_set_default_codeblock (params);

  encoder->base_quant = 20;

  encoder->encoder_params.quant_index_dc = 4;
  encoder->encoder_params.quant_index[0] = 4;
  encoder->encoder_params.quant_index[1] = 4;
  encoder->encoder_params.quant_index[2] = 6;
  encoder->encoder_params.quant_index[3] = 8;
  encoder->encoder_params.quant_index[4] = 10;
  encoder->encoder_params.quant_index[5] = 12;

  return encoder;
}

void
schro_encoder_free (SchroEncoder *encoder)
{
  int i;

  if (encoder->tmp_frame0) {
    schro_frame_free (encoder->tmp_frame0);
  }
  if (encoder->tmp_frame1) {
    schro_frame_free (encoder->tmp_frame1);
  }
  if (encoder->motion_vectors_dc) {
    free (encoder->motion_vectors);
  }
  if (encoder->motion_vectors_dc) {
    free (encoder->motion_vectors_dc);
  }
#if 0
  if (encoder->motion_vectors_none) {
    free (encoder->motion_vectors_none);
  }
  if (encoder->motion_vectors_scan) {
    free (encoder->motion_vectors_scan);
  }
#endif
  for(i=0;i<encoder->n_reference_frames; i++) {
    schro_frame_free(encoder->reference_frames[i]);
  }
  for(i=0;i<encoder->frame_queue_length; i++) {
    schro_frame_free(encoder->frame_queue[i]);
  }
  if (encoder->subband_buffer) {
    schro_buffer_unref (encoder->subband_buffer);
  }

  free (encoder->tmpbuf);
  free (encoder->tmpbuf2);
  free (encoder);
}

void
schro_encoder_set_size (SchroEncoder *encoder, int width, int height)
{
  SchroParams *params = &encoder->params;

  if (params->width == width && params->height == height) return;

  params->width = width;
  params->height = height;
  params->chroma_width =
    (width + params->chroma_h_scale - 1) / params->chroma_h_scale;
  params->chroma_height =
    (height + params->chroma_v_scale - 1) / params->chroma_v_scale;

  if (encoder->tmp_frame0) {
    schro_frame_free (encoder->tmp_frame0);
    encoder->tmp_frame0 = NULL;
  }
  if (encoder->tmp_frame1) {
    schro_frame_free (encoder->tmp_frame1);
    encoder->tmp_frame1 = NULL;
  }

  encoder->need_rap = TRUE;
}

void
schro_encoder_set_framerate (SchroEncoder *encoder, int numerator,
    int denominator)
{
  encoder->params.frame_rate_numerator = numerator;
  encoder->params.frame_rate_denominator = denominator;
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
schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame)
{
  frame->frame_number = encoder->frame_queue_index;
  encoder->frame_queue_index++;

  schro_encoder_frame_queue_push (encoder, frame);
}

void
schro_encoder_end_of_stream (SchroEncoder *encoder)
{
  encoder->end_of_stream = TRUE;
}

#define CLAMP(x,a,b) ((x)<(a) ? (a) : ((x)>(b) ? (b) : (x)))
static void
schro_encoder_choose_quantisers (SchroEncoder *encoder)
{
  if (encoder->picture->n_refs > 0) {
    encoder->base_quant = 26;
  } else {
    encoder->base_quant = 24;
  }
#if 0
  if ((encoder->picture->frame_number & 0x1f) > 0x10) {
    encoder->base_quant = 28;
  } else {
    encoder->base_quant = 24;
  }
#endif
  encoder->encoder_params.quant_index_dc = 20;
  encoder->encoder_params.quant_index[0] = CLAMP(encoder->base_quant, 0, 63);
  encoder->encoder_params.quant_index[1] = CLAMP(encoder->base_quant, 0, 63);
  encoder->encoder_params.quant_index[2] = CLAMP(encoder->base_quant, 0, 63);
  encoder->encoder_params.quant_index[3] = CLAMP(encoder->base_quant, 0, 63);
  encoder->encoder_params.quant_index[4] = CLAMP(encoder->base_quant, 0, 63);
  encoder->encoder_params.quant_index[5] = CLAMP(encoder->base_quant, 0, 63);
#if 0
  encoder->encoder_params.quant_index_dc = CLAMP(encoder->base_quant - 8, 0, 63);
  encoder->encoder_params.quant_index[0] = CLAMP(encoder->base_quant - 8, 0, 63);
  encoder->encoder_params.quant_index[1] = CLAMP(encoder->base_quant - 8, 0, 63);
  encoder->encoder_params.quant_index[2] = CLAMP(encoder->base_quant - 6, 0, 63);
  encoder->encoder_params.quant_index[3] = CLAMP(encoder->base_quant - 4, 0, 63);
  encoder->encoder_params.quant_index[4] = CLAMP(encoder->base_quant - 2, 0, 63);
  encoder->encoder_params.quant_index[5] = CLAMP(encoder->base_quant, 0, 63);
#endif
}

SchroBuffer *
schro_encoder_encode (SchroEncoder *encoder)
{
  SchroBuffer *outbuffer;
  SchroBuffer *subbuffer;
  int i;
  
  if (encoder->need_rap) {
    outbuffer = schro_buffer_new_and_alloc (0x100);

    encoder->bits = schro_bits_new ();
    schro_bits_encode_init (encoder->bits, outbuffer);

    schro_encoder_encode_rap (encoder);
    encoder->need_rap = FALSE;

    if (encoder->bits->offset > 0) {
      subbuffer = schro_buffer_new_subbuffer (outbuffer, 0,
          encoder->bits->offset/8);
    } else {
      subbuffer = NULL;
    }
    schro_bits_free (encoder->bits);
    schro_buffer_unref (outbuffer);

    return subbuffer;
  }
 
  if (encoder->picture_index >= encoder->n_pictures) {
    schro_encoder_create_picture_list (encoder);
  }
  encoder->picture = &encoder->picture_list[encoder->picture_index];

  schro_encoder_choose_quantisers (encoder);

  encoder->encode_frame = schro_encoder_frame_queue_get (encoder,
      encoder->picture->frame_number);
  if (encoder->encode_frame == NULL) return NULL;

  if (encoder->picture->n_refs > 0) {
    encoder->ref_frame0 = schro_encoder_reference_get (encoder,
        encoder->picture->reference_frame_number[0]);
    SCHRO_ASSERT (encoder->ref_frame0 != NULL);
  }
  if (encoder->picture->n_refs > 1) {
    encoder->ref_frame1 = schro_encoder_reference_get (encoder,
        encoder->picture->reference_frame_number[1]);
    SCHRO_ASSERT (encoder->ref_frame1 != NULL);
  }

  SCHRO_DEBUG("encoding picture frame_number=%d is_ref=%d n_refs=%d",
      encoder->picture->frame_number, encoder->picture->is_ref,
      encoder->picture->n_refs);

  schro_encoder_frame_queue_remove (encoder, encoder->picture->frame_number);

  outbuffer = schro_buffer_new_and_alloc (0x40000);

  encoder->bits = schro_bits_new ();
  schro_bits_encode_init (encoder->bits, outbuffer);

  SCHRO_DEBUG("frame number %d", encoder->frame_number);

  encoder->params.num_refs = encoder->picture->n_refs;
  if (encoder->picture->n_refs > 0) {
    schro_encoder_encode_inter (encoder);
  } else {
    schro_encoder_encode_intra (encoder);
  }

  for(i=0;i<encoder->picture->n_retire;i++){
    schro_encoder_reference_retire (encoder, encoder->picture->retire[i]);
  }

  encoder->picture_index++;

  SCHRO_ERROR("frame %d encoded %d bits (q=%d)", encoder->picture->frame_number,
      encoder->bits->offset, encoder->base_quant);

#if 0
  if (encoder->picture->n_refs == 0) {
    if (encoder->bits->offset > 200000) {
      encoder->base_quant+=4;
    } else if (encoder->bits->offset > 100000) {
      encoder->base_quant++;
    }
    if (encoder->bits->offset < 100000) {
      encoder->base_quant--;
    }
    encoder->base_quant = CLAMP(encoder->base_quant, 0, 70);
  }
  encoder->base_quant = 35;
#endif

  if (encoder->bits->offset > 0) {
    subbuffer = schro_buffer_new_subbuffer (outbuffer, 0,
        encoder->bits->offset/8);
  } else {
    subbuffer = NULL;
  }
  schro_bits_free (encoder->bits);
  schro_buffer_unref (outbuffer);

  return subbuffer;
}

static void
schro_encoder_create_picture_list (SchroEncoder *encoder)
{
  int type = 2;
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
        encoder->n_pictures = 8;
        encoder->picture_list[0].is_ref = 1;
        encoder->picture_list[0].n_refs = 0;
        encoder->picture_list[0].frame_number = encoder->frame_number + 7;
        for(i=1;i<8;i++){
          encoder->picture_list[i].is_ref = 0;
          encoder->picture_list[i].n_refs = 2;
          encoder->picture_list[i].frame_number = encoder->frame_number + i - 1;
          encoder->picture_list[i].reference_frame_number[0] =
            encoder->frame_number - 1;
          encoder->picture_list[i].reference_frame_number[1] =
            encoder->frame_number + 7;
          encoder->picture_list[i].n_retire = 0;
        }
        encoder->picture_list[7].n_retire = 1;
        encoder->picture_list[7].retire[0] = encoder->frame_number - 1;
      }
      encoder->frame_number+=encoder->n_pictures;
      break;
    default:
      SCHRO_ASSERT(0);
      break;
  }
  encoder->picture_index = 0;
}

void
schro_encoder_encode_intra (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int is_ref = encoder->picture->is_ref;

  schro_params_calculate_iwt_sizes (params);

  if (encoder->tmp_frame0 == NULL) {
    encoder->tmp_frame0 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  schro_encoder_encode_frame_header (encoder, SCHRO_PARSE_CODE_INTRA_REF);

  schro_frame_convert (encoder->tmp_frame0, encoder->encode_frame);
  schro_frame_shift_left (encoder->tmp_frame0, 4);

  schro_frame_free (encoder->encode_frame);

  schro_encoder_encode_transform_parameters (encoder);

  schro_frame_iwt_transform (encoder->tmp_frame0, &encoder->params,
      encoder->tmpbuf);

  schro_encoder_encode_transform_data (encoder, 0);
  schro_encoder_encode_transform_data (encoder, 1);
  schro_encoder_encode_transform_data (encoder, 2);

  if (is_ref) {
    SchroFrame *ref_frame;

    schro_frame_inverse_iwt_transform (encoder->tmp_frame0, &encoder->params,
        encoder->tmpbuf);

    ref_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    schro_frame_shift_right (encoder->tmp_frame0, 4);
    schro_frame_convert (ref_frame, encoder->tmp_frame0);

    schro_encoder_reference_add (encoder, ref_frame);
  }

}

void
schro_encoder_encode_inter (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int is_ref = 0;
  int residue_bits_start;

  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  if (encoder->tmp_frame0 == NULL) {
    encoder->tmp_frame0 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }
  if (encoder->tmp_frame1 == NULL) {
    encoder->tmp_frame1 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
        params->mc_luma_width, params->mc_luma_height, 2, 2);
  }

  schro_encoder_encode_frame_header (encoder, SCHRO_PARSE_CODE_INTER_NON_REF);

  schro_encoder_motion_predict (encoder);

  schro_encoder_encode_frame_prediction (encoder);

  schro_frame_convert (encoder->tmp_frame0, encoder->encode_frame);
  schro_frame_free (encoder->encode_frame);

  schro_frame_copy_with_motion (encoder->tmp_frame1,
      encoder->ref_frame0, encoder->ref_frame1, encoder->motion_vectors,
      &encoder->params);

  schro_frame_subtract (encoder->tmp_frame0, encoder->tmp_frame1);

  schro_frame_zero_extend (encoder->tmp_frame0, params->width, params->height);

  schro_encoder_encode_transform_parameters (encoder);

  schro_frame_shift_left (encoder->tmp_frame0, 4);
  schro_frame_iwt_transform (encoder->tmp_frame0, &encoder->params,
      encoder->tmpbuf);
  residue_bits_start = encoder->bits->offset;
  schro_encoder_encode_transform_data (encoder, 0);
  schro_encoder_encode_transform_data (encoder, 1);
  schro_encoder_encode_transform_data (encoder, 2);

  encoder->metric_to_cost =
    (double)(encoder->bits->offset - residue_bits_start) /
    encoder->stats_metric;
  SCHRO_ERROR("pred bits %d, residue bits %d, stats_metric %d, m_to_c = %g, dc_blocks %d, scan blocks %d",
      residue_bits_start, encoder->bits->offset - residue_bits_start,
      encoder->stats_metric, encoder->metric_to_cost,
      encoder->stats_dc_blocks, encoder->stats_scan_blocks);

  if (is_ref) {
    SchroFrame *ref_frame;

    schro_frame_inverse_iwt_transform (encoder->tmp_frame0,
        &encoder->params, encoder->tmpbuf);
    schro_frame_shift_right (encoder->tmp_frame0, 4);
    schro_frame_add (encoder->tmp_frame0, encoder->tmp_frame1);

    ref_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    schro_frame_convert (ref_frame, encoder->tmp_frame0);
  }
}

void
schro_encoder_encode_frame_prediction (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i,j;
  int global_motion;
  SchroArith *arith;
  int superblock_count = 0;

  /* block params flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* mv precision flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* global motion flag */
  global_motion = FALSE;
  schro_bits_encode_bit (encoder->bits, global_motion);

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, encoder->subband_buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<4*params->y_num_mb;j+=4){
    for(i=0;i<4*params->x_num_mb;i+=4){
      int k,l;
      int mb_using_global = FALSE;
      int mb_common = FALSE;
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
        &encoder->motion_vectors[j*(4*params->x_num_mb) + i];

      split_prediction = schro_motion_split_prediction (
          encoder->motion_vectors, params, i, j);
      split_residual = (mv->split - split_prediction)%3;
      schro_arith_encode_mode (arith, SCHRO_CTX_SPLIT_0, SCHRO_CTX_SPLIT_1,
          split_residual);

#if 0
      common_prediction = schro_motion_common_prediction (
          ncoder->motion_vectors, params, i, j);
#endif

      if (global_motion) {
        schro_arith_context_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
            mb_using_global);
      } else {
        SCHRO_ASSERT(mb_using_global == FALSE);
      }
      if (!mb_using_global) {
        SCHRO_ASSERT(mv->split < 3);
        //schro_bits_encode_bits (encoder->bits, 2, mv->split);
      } else {
        SCHRO_ASSERT(mv->split == 2);
      }
      if (mv->split != 0) {
        schro_arith_context_encode_bit (arith, SCHRO_CTX_COMMON,
            mb_common);
      }

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &encoder->motion_vectors[(j+l)*(4*params->x_num_mb) + i + k];

          schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
              mv->pred_mode & 1);
          schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
              mv->pred_mode >> 1);
          if (mv->pred_mode == 0) {
            int pred[3];

            schro_motion_dc_prediction (encoder->motion_vectors,
                params, i+k, j+l, pred);

            schro_arith_context_encode_sint (arith,
                SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                SCHRO_CTX_LUMA_DC_SIGN,
                mv->dc[0] - pred[0]);
            schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                SCHRO_CTX_CHROMA1_DC_SIGN,
                mv->dc[1] - pred[1]);
            schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
                SCHRO_CTX_CHROMA2_DC_SIGN,
                mv->dc[2] - pred[2]);
          } else {
            int pred_x, pred_y;

            schro_motion_vector_prediction (encoder->motion_vectors,
                params, i+k, j+l, &pred_x, &pred_y);

            schro_arith_context_encode_sint(arith,
                SCHRO_CTX_MV_REF1_H_CONT_BIN1,
                SCHRO_CTX_MV_REF1_H_VALUE,
                SCHRO_CTX_MV_REF1_H_SIGN,
                mv->x - pred_x);
            schro_arith_context_encode_sint(arith,
                SCHRO_CTX_MV_REF1_V_CONT_BIN1,
                SCHRO_CTX_MV_REF1_V_VALUE,
                SCHRO_CTX_MV_REF1_V_SIGN,
                mv->y - pred_y);
          }
        }
      }

      superblock_count++;
      if (superblock_count == 32) {
        schro_arith_halve_all_counts (arith);
        superblock_count = 0;
      }
    }
  }

  schro_arith_flush (arith);

  schro_bits_encode_uegol(encoder->bits, arith->offset);

  schro_bits_sync (encoder->bits);
  schro_bits_append (encoder->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}


void
schro_encoder_encode_rap (SchroEncoder *encoder)
{
  
  /* parse parameters */
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'C');
  schro_bits_encode_bits (encoder->bits, 8, 'D');

  schro_bits_encode_bits (encoder->bits, 8, SCHRO_PARSE_CODE_RAP);

  /* offsets */
  /* FIXME */
  schro_bits_encode_bits (encoder->bits, 24, 0);
  schro_bits_encode_bits (encoder->bits, 24, 0);

  /* rap frame number */
  /* FIXME */
  schro_bits_encode_ue2gol (encoder->bits, 0);

  /* major/minor version */
  schro_bits_encode_uegol (encoder->bits, 0);
  schro_bits_encode_uegol (encoder->bits, 0);

  /* profile */
  schro_bits_encode_uegol (encoder->bits, 0);
  /* level */
  schro_bits_encode_uegol (encoder->bits, 0);


  /* sequence parameters */
  /* video format */
  schro_bits_encode_uegol (encoder->bits, 5);
  /* custom dimensions */
  schro_bits_encode_bit (encoder->bits, TRUE);
  schro_bits_encode_uegol (encoder->bits, encoder->params.width);
  schro_bits_encode_uegol (encoder->bits, encoder->params.height);

  /* chroma format */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* signal range */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* display parameters */
  /* interlace */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* frame rate */
  schro_bits_encode_bit (encoder->bits, TRUE);
  schro_bits_encode_uegol (encoder->bits, 0);
  schro_bits_encode_uegol (encoder->bits, encoder->params.frame_rate_numerator);
  schro_bits_encode_uegol (encoder->bits, encoder->params.frame_rate_denominator);

  /* pixel aspect ratio */
  schro_bits_encode_bit (encoder->bits, TRUE);
  schro_bits_encode_uegol (encoder->bits, 0);
  schro_bits_encode_uegol (encoder->bits, 1);
  schro_bits_encode_uegol (encoder->bits, 1);

  /* clean area flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* colour matrix flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* signal_range flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* colour spec flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* transfer characteristic flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  schro_bits_sync (encoder->bits);
}

void
schro_encoder_encode_frame_header (SchroEncoder *encoder,
    int parse_code)
{
  int i;
  
  /* parse parameters */
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'C');
  schro_bits_encode_bits (encoder->bits, 8, 'D');
  schro_bits_encode_bits (encoder->bits, 8, parse_code);

  /* offsets */
  /* FIXME */
  schro_bits_encode_bits (encoder->bits, 24, 0);
  schro_bits_encode_bits (encoder->bits, 24, 0);

  /* frame number offset */
  schro_bits_encode_se2gol (encoder->bits, encoder->picture->frame_number);

  schro_bits_encode_uegol (encoder->bits, encoder->picture->n_refs);
  for(i=0;i<encoder->picture->n_refs;i++){
    schro_bits_encode_se2gol (encoder->bits,
        encoder->picture->reference_frame_number[i] -
        encoder->picture->frame_number);
  }

  /* list */
  schro_bits_encode_uegol (encoder->bits, encoder->picture->n_retire);
  for(i=0;i<encoder->picture->n_retire;i++){
    schro_bits_encode_se2gol (encoder->bits,
        encoder->picture->retire[i] - encoder->picture->frame_number);
  }

  schro_bits_sync (encoder->bits);
}


void
schro_encoder_encode_transform_parameters (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;

  /* transform */
  if (params->wavelet_filter_index == SCHRO_WAVELET_APPROX97) {
    schro_bits_encode_bit (encoder->bits, 0);
  } else {
    schro_bits_encode_bit (encoder->bits, 1);
    schro_bits_encode_uegol (encoder->bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    schro_bits_encode_bit (encoder->bits, 0);
  } else {
    schro_bits_encode_bit (encoder->bits, 1);
    schro_bits_encode_uegol (encoder->bits, params->transform_depth);
  }

  /* spatial partitioning */
  schro_bits_encode_bit (encoder->bits, params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    int i;

    schro_bits_encode_bit (encoder->bits, params->nondefault_partition_flag);
    for(i=0;i<params->transform_depth;i++){
      schro_bits_encode_uegol (encoder->bits, params->codeblock_width[i]);
      schro_bits_encode_uegol (encoder->bits, params->codeblock_height[i]);
    }
    schro_bits_encode_uegol (encoder->bits, params->codeblock_mode_index);
  }

  schro_bits_sync(encoder->bits);
}

void
schro_encoder_init_subbands (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
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

//#define ROUND_UP_SHIFT(x,y) (((x) + (1<<(y)) - 1)>>(y))
#define ROUND_UP_SHIFT(x,y) (((x))>>(y))
void
schro_encoder_clean_up_transform (SchroEncoder *encoder, int component,
    int index)
{
  SchroSubband *subband = encoder->subbands + index;
  SchroParams *params = &encoder->params;
  int stride;
  int width;
  int height;
  int offset;
  int w;
  int h;
  int shift;
  int16_t *data;
  int i,j;

  shift = params->transform_depth - subband->scale_factor_shift;

  if (component == 0) {
    stride = subband->stride >> 1;
    width = subband->w;
    w = ROUND_UP_SHIFT(params->width, shift);
    height = subband->h;
    h = ROUND_UP_SHIFT(params->height, shift);
    offset = subband->offset;
  } else {
    stride = subband->chroma_stride >> 1;
    width = subband->chroma_w;
    w = ROUND_UP_SHIFT(params->width/2, shift);
    height = subband->chroma_h;
    h = ROUND_UP_SHIFT(params->height/2, shift);
    offset = subband->chroma_offset;
  }

  data = (int16_t *)encoder->tmp_frame0->components[component].data + offset;

  SCHRO_DEBUG("subband index=%d %d x %d at offset %d with stride %d; clean area %d %d", index,
      width, height, offset, stride, w, h);

  /* FIXME this is dependent on the particular wavelet transform */
  h+=1;
  w+=1;

  for(j=0;j<h;j++){
    for(i=w;i<width;i++){
      data[j*stride + i] = 0;
    }
  }
  for(j=h;j<height;j++){
    for(i=0;i<width;i++){
      data[j*stride + i] = 0;
    }
  }
}

void
schro_encoder_encode_transform_data (SchroEncoder *encoder, int component)
{
  int i;
  SchroParams *params = &encoder->params;

  schro_encoder_init_subbands (encoder);

  for (i=0;i < 1 + 3*params->transform_depth; i++) {
    schro_encoder_clean_up_transform (encoder, component, i);
    schro_encoder_encode_subband (encoder, component, i);
  }
}

static int
dequantize (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return q * quant_factor - quant_offset;
  } else {
    return q * quant_factor + quant_offset;
  }
}

static int
quantize (int value, int quant_factor, int quant_offset)
{
  if (value == 0) return 0;
  if (value < 0) {
    value = - (-value - quant_offset + quant_factor/2)/quant_factor;
  } else {
    value = (value - quant_offset + quant_factor/2)/quant_factor;
  }
  return value;
}

static int
schro_encoder_quantize_subband (SchroEncoder *encoder, int component, int index,
    int16_t *quant_data)
{
  SchroSubband *subband = encoder->subbands + index;
  int pred_value;
  int quant_factor;
  int quant_offset;
  int stride;
  int width;
  int height;
  int offset;
  int i,j;
  int16_t *data;
  int subband_zero_flag;

  subband_zero_flag = 1;

  quant_factor = schro_table_quant[subband->quant_index];
  quant_offset = schro_table_offset[subband->quant_index];

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

  data = (int16_t *)encoder->tmp_frame0->components[component].data + offset;

  if (index == 0) {
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int q;

        if (j>0) {
          if (i>0) {
            pred_value = (data[j*stride + i - 1] +
                data[(j-1)*stride + i] + data[(j-1)*stride + i - 1] + 1)/3;
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

        q = quantize(data[j*stride + i] - pred_value, quant_factor, quant_offset);
        data[j*stride + i] = dequantize(q, quant_factor, quant_offset) +
          pred_value;
        quant_data[j*width + i] = q;
        if (data[j*stride + i] != 0) {
          subband_zero_flag = 0;
        }

      }
    }
  } else {
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int q;

        q = quantize(data[j*stride + i], quant_factor, quant_offset);
        data[j*stride + i] = dequantize(q, quant_factor, quant_offset);
        quant_data[j*width + i] = q;
        if (data[j*stride + i] != 0) {
          subband_zero_flag = 0;
        }

      }
    }
  }

  return subband_zero_flag;
}

void
schro_encoder_encode_subband (SchroEncoder *encoder, int component, int index)
{
  SchroParams *params = &encoder->params;
  SchroSubband *subband = encoder->subbands + index;
  SchroSubband *parent_subband = NULL;
  SchroArith *arith;
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
  int16_t *quant_data;
  int x,y;

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

  SCHRO_DEBUG("subband index=%d %d x %d at offset %d with stride %d", index,
      width, height, offset, stride);

  data = (int16_t *)encoder->tmp_frame0->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)encoder->tmp_frame0->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)encoder->tmp_frame0->components[component].data +
        parent_subband->chroma_offset;
    }
  }
  quant_factor = schro_table_quant[subband->quant_index];
  quant_offset = schro_table_offset[subband->quant_index];

  scale_factor = 1<<(params->transform_depth - subband->scale_factor_shift);
  ntop = (scale_factor>>1) * quant_factor;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, encoder->subband_buffer);
  schro_arith_init_contexts (arith);

  quant_data = malloc (sizeof(int16_t) * height * width);
  subband_zero_flag = schro_encoder_quantize_subband (encoder, component,
      index, quant_data);
  schro_bits_encode_bit (encoder->bits, subband_zero_flag);
  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_bits_sync (encoder->bits);
    schro_arith_free (arith);
    free (quant_data);
    return;
  }

#define MIN(a,b) ((a)<(b) ? (a) : (b))
  for(y=0;y<height;y+=params->codeblock_height[subband->scale_factor_shift]) {
    int ymax = MIN(y + params->codeblock_height[subband->scale_factor_shift],
        height);

    for(x=0;x<width;x+=params->codeblock_width[subband->scale_factor_shift]) {
      int xmax = MIN(x + params->codeblock_width[subband->scale_factor_shift],
          width);
      int zero_codeblock = 1;

  if (index == 0) zero_codeblock = 0;
  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      if (quant_data[j*width + i] != 0) {
        zero_codeblock = 0;
      }
    }
  }
  schro_arith_context_encode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK, zero_codeblock);
  if (zero_codeblock) {
    continue;
  }

  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      int parent_zero;
      int cont_context;
      int value_context;
      int nhood_sum;
      int previous_value;
      int sign_context;

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
          cont_context = SCHRO_CTX_Z_BIN1_0;
        } else {
          cont_context = SCHRO_CTX_Z_BIN1_1;
        }
        value_context = SCHRO_CTX_Z_VALUE;
        if (previous_value > 0) {
          sign_context = SCHRO_CTX_Z_SIGN_0;
        } else if (previous_value < 0) {
          sign_context = SCHRO_CTX_Z_SIGN_1;
        } else {
          sign_context = SCHRO_CTX_Z_SIGN_2;
        }
      } else {
        if (nhood_sum == 0) {
          cont_context = SCHRO_CTX_NZ_BIN1_0;
        } else {
          if (nhood_sum <= ntop) {
            cont_context = SCHRO_CTX_NZ_BIN1_1;
          } else {
            cont_context = SCHRO_CTX_NZ_BIN1_2;
          }
        }
        value_context = SCHRO_CTX_NZ_VALUE;
        if (previous_value > 0) {
          sign_context = SCHRO_CTX_NZ_SIGN_0;
        } else if (previous_value < 0) {
          sign_context = SCHRO_CTX_NZ_SIGN_1;
        } else {
          sign_context = SCHRO_CTX_NZ_SIGN_2;
        }
      }

      schro_arith_context_encode_sint (arith, cont_context, value_context,
          sign_context, quant_data[j*width + i]);
    }
  }
    }
  }
  free (quant_data);

  schro_arith_flush (arith);

  schro_bits_encode_uegol (encoder->bits, subband->quant_index);
  schro_bits_encode_uegol (encoder->bits, arith->offset);

  schro_bits_sync (encoder->bits);

  schro_bits_append (encoder->bits, arith->buffer->data, arith->offset);
  schro_arith_free (arith);
}

static int table[32][3] = {
  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },

  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },

  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },

  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },

  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
};

void
codeblock_line_encode (SchroSubband *subband, int16_t *data,
    int16_t *quant_data, int x, int y, int n,
    int16_t *tmp, SchroArith *arith, int stride, int ntop,
    int16_t *parent_data)
{
  int16_t *nhood_sum = tmp;
  int16_t *previous_value = tmp+n;
  int16_t *parent_zero = tmp + 2*n;
  int i;

  if (x>0) {
    for(i=0;i<n;i++){
      nhood_sum[i] = abs(data[i-1]);
    }
  } else {
    nhood_sum[0] = 0;
    for(i=1;i<n;i++){
      nhood_sum[i] = abs(data[i-1]);
    }
  }
  if (y>0) {
    if (x>0) {
      for(i=0;i<n;i++){
        nhood_sum[i] += abs(data[-stride + i]) + abs(data[-stride-1+i]);
      }
    } else {
      nhood_sum[i] += abs(data[-stride + i]);
      for(i=1;i<n;i++){
        nhood_sum[i] += abs(data[-stride + i]) + abs(data[-stride-1+i]);
      }
    }
  }

  if (subband->has_parent) {
    for(i=0;i<n;i++){
      if (parent_data[(y>>1)*(stride<<1) + ((x+i)>>1)]==0) {
        parent_zero[i] = 1;
      } else {
        parent_zero[i] = 0;
      }
    }
  } else {
    if (subband->x == 0 && subband->y == 0) {
      for(i=0;i<n;i++){
        parent_zero[i] = 0;
      }
    } else {
      for(i=0;i<n;i++){
        parent_zero[i] = 1;
      }
    }
  }

  if (subband->vertically_oriented && y>0) {
    for(i=0;i<n;i++){
      previous_value[i] = data[-stride+i];
    }
  } else {
    for(i=0;i<n;i++){
      previous_value[i] = 0;
    }
  }


  if (subband->horizontally_oriented) {
    int prev_value;

    if (x>0) {
      prev_value = data[-1];
    } else {
      prev_value = 0;
    }
    for(i=0;i<n;i++){
      int table_index;

      table_index = (parent_zero[i] == 0)<<4;
      table_index |= (prev_value < 0)<<3;
      table_index |= (prev_value > 0)<<2;
      table_index |= (nhood_sum[i] > ntop)<<1;
      table_index |= (nhood_sum[i] > 0)<<0;

      schro_arith_context_encode_sint (arith, table[table_index][0],
          table[table_index][1], table[table_index][2], quant_data[i]);
    }
  } else {
    for(i=0;i<n;i++){
      int table_index;

      table_index = (parent_zero[i] == 0)<<4;
      table_index |= (previous_value[i] < 0)<<3;
      table_index |= (previous_value[i] > 0)<<2;
      table_index |= (nhood_sum[i] > ntop)<<1;
      table_index |= (nhood_sum[i] > 0)<<0;

      schro_arith_context_encode_sint (arith, table[table_index][0],
          table[table_index][1], table[table_index][2], quant_data[i]);
    }
  }
}

#if 0
static void
schro_encoder_choose_split (SchroEncoder *encoder, int x, int y)
{
  SchroParams *params = &encoder->params;
  SchroMotionVector *mv0;
  SchroMotionVector *mv;
  int i,j;
 
  mv0 = &encoder->motion_vectors[y*(4*params->x_num_mb) + x];

  if (mv0->pred_mode == 0) {
    for(j=0;j<4;j++){
      for(i=0;i<4;i++){
        mv = &encoder->motion_vectors[(y+j)*(4*params->x_num_mb) + (x+j)];
        if (mv0->dc[0] != mv->dc[0] ||
            mv0->dc[1] != mv->dc[1] ||
            mv0->dc[2] != mv->dc[2]) {
          mv->split = 2;
          return;
        }
      }
    }
    mv->split = 0;
  } else {
    for(j=0;j<4;j++){
      for(i=0;i<4;i++){
        mv = &encoder->motion_vectors[(y+j)*(4*params->x_num_mb) + (x+j)];
        if (mv0->pred_mode != mv->pred_mode ||
            mv0->x != mv->x ||
            mv0->y != mv->y) {
          mv->split = 2;
          return;
        }
      }
    }
    mv->split = 0;
  }
}

int cost (int value)
{
  int n;
  if (value == 0) return 1;
  if (value < 0) value = -value;
  value++;
  n = 0;
  while (value) {
    n+=2;
    value>>=1;
  }
  return n;
}

void
schro_encoder_motion_predict (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i;
  int j;
  SchroFrame *ref_frame;
  SchroFrame *frame;
  int sum_pred_x;
  int sum_pred_y;
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
  double sum_x, sum_y;

  SCHRO_ASSERT(params->x_num_mb != 0);
  SCHRO_ASSERT(params->y_num_mb != 0);

  if (encoder->motion_vectors == NULL) {
    encoder->motion_vectors = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
  if (encoder->motion_vectors_dc == NULL) {
    encoder->motion_vectors_dc = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
  if (encoder->motion_vectors_none == NULL) {
    encoder->motion_vectors_none = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
  if (encoder->motion_vectors_scan == NULL) {
    encoder->motion_vectors_scan = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }

  ref_frame = encoder->ref_frame0;
  if (!ref_frame) {
    SCHRO_ERROR("no reference frame");
  }
  frame = encoder->encode_frame;

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      int w,h;

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;
      
      w = CLAMP(params->width - x, 0, params->xbsep_luma);
      h = CLAMP(params->height - y, 0, params->ybsep_luma);

      predict_dc (&encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i],
          frame, x, y, w, h);

      predict_motion_none (&encoder->motion_vectors_none[j*(4*params->x_num_mb) + i],
          frame, ref_frame, x, y, w, h);

      predict_motion_search (&encoder->motion_vectors_scan[j*(4*params->x_num_mb) + i],
          frame, ref_frame, x, y, w, h);

      (void)&predict_motion;
    }
  }

  encoder->stats_metric = 0;
  encoder->stats_dc_blocks = 0;
  encoder->stats_none_blocks = 0;
  encoder->stats_scan_blocks = 0;
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int cost_dc;
      int cost_none;
      int cost_scan;
      int pred[3];
      int pred_x, pred_y;
      SchroMotionVector *mv;

      schro_motion_dc_prediction (encoder->motion_vectors,
          params, i, j, pred);
      mv = &encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i];
      cost_dc = cost(mv->dc[0] - pred[0]) + cost(mv->dc[1] - pred[1]) +
        cost(mv->dc[2] - pred[2]);
      cost_dc += encoder->metric_to_cost * mv->metric;

      mv = &encoder->motion_vectors_none[j*(4*params->x_num_mb) + i];
      schro_motion_vector_prediction (encoder->motion_vectors,
          params, i, j, &pred_x, &pred_y);
      cost_none = cost(mv->x - pred_x) + cost(mv->y - pred_y);
      cost_none += encoder->metric_to_cost * mv->metric;

      mv = &encoder->motion_vectors_scan[j*(4*params->x_num_mb) + i];
      cost_scan = cost(mv->x - pred_x) + cost(mv->y - pred_y);
      cost_scan += encoder->metric_to_cost * mv->metric;

      if (cost_none < cost_dc && cost_none < cost_scan) {
        memcpy (&encoder->motion_vectors[j*(4*params->x_num_mb) + i],
            &encoder->motion_vectors_none[j*(4*params->x_num_mb) + i],
            sizeof(SchroMotionVector));
        encoder->stats_none_blocks++;
      } else if (cost_scan < cost_dc) {
        memcpy (&encoder->motion_vectors[j*(4*params->x_num_mb) + i],
            &encoder->motion_vectors_scan[j*(4*params->x_num_mb) + i],
            sizeof(SchroMotionVector));
        encoder->stats_scan_blocks++;
      } else {
        memcpy (&encoder->motion_vectors[j*(4*params->x_num_mb) + i],
            &encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i],
            sizeof(SchroMotionVector));
        encoder->stats_dc_blocks++;
      }
      encoder->stats_metric += 
          encoder->motion_vectors[j*(4*params->x_num_mb) + i].metric;

      encoder->motion_vectors[j*(4*params->x_num_mb) + i].split = 2;
    }
  }
  for(j=0;j<4*params->y_num_mb;j+=4) {
    for(i=0;i<4*params->x_num_mb;i+=4) {
      schro_encoder_choose_split (encoder, i, j);
    }
  }

  sum_pred_x = 0;
  sum_pred_y = 0;
#if 0
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      SchroMotionVector *mv =
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
#endif

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
  if (sum_x != 0) {
    mag_x = mag_x/sum_x;
    skew_x = skew_x/sum_x;
  } else {
    mag_x = 0;
    skew_x = 0;
  }
  if (sum_y != 0) {
    mag_y = mag_y/sum_y;
    skew_y = skew_y/sum_y;
  } else {
    mag_y = 0;
    skew_y = 0;
  }

  encoder->pan_x = pan_x;
  encoder->pan_y = pan_y;
  encoder->mag_x = mag_x;
  encoder->mag_y = mag_y;
  encoder->skew_x = skew_x;
  encoder->skew_y = skew_y;

  SCHRO_DEBUG("pan %g %g mag %g %g skew %g %g",
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
calculate_metric2 (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2, int width, int height)
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
predict_motion_search (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
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
  mv->x = dx;
  mv->y = dy;
  mv->metric = min_metric;
  mv->pred_mode = 1;
}

static void
predict_motion_scan (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
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
predict_motion_none (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int metric;

  metric = calculate_metric2 (frame, x, y, reference_frame, x, y, w, h);
  mv->x = 0;
  mv->y = 0;
  mv->metric = metric;
  mv->pred_mode = 1;
}

static void
predict_motion (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int how = 2;

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
predict_dc (SchroMotionVector *mv, SchroFrame *frame, int x, int y,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;
  uint8_t *a;
  int a_stride;
  int sum;

  if (height == 0 || width == 0) {
    mv->pred_mode = 0;
    mv->metric = 1000000;
    return;
  }

  SCHRO_ASSERT(x + width <= frame->components[0].width);
  SCHRO_ASSERT(y + height <= frame->components[0].height);

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
#endif


/* frame queue */

static void
schro_encoder_frame_queue_push (SchroEncoder *encoder, SchroFrame *frame)
{
  encoder->frame_queue[encoder->frame_queue_length] = frame;
  encoder->frame_queue_length++;
  SCHRO_ASSERT(encoder->frame_queue_length < 10);
}

static SchroFrame *
schro_encoder_frame_queue_get (SchroEncoder *encoder, int frame_index)
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
schro_encoder_frame_queue_remove (SchroEncoder *encoder, int frame_index)
{
  int i;
  for(i=0;i<encoder->frame_queue_length;i++){
    if (encoder->frame_queue[i]->frame_number == frame_index) {
      memmove (encoder->frame_queue + i, encoder->frame_queue + i + 1,
          sizeof(SchroFrame *)*(encoder->frame_queue_length - i - 1));
      encoder->frame_queue_length--;
      return;
    }
  }
}


/* reference pool */

static void
schro_encoder_reference_add (SchroEncoder *encoder, SchroFrame *frame)
{
  SCHRO_DEBUG("adding %d", frame->frame_number);
  encoder->reference_frames[encoder->n_reference_frames] = frame;
  encoder->n_reference_frames++;
  SCHRO_ASSERT(encoder->n_reference_frames < 10);
}

static SchroFrame *
schro_encoder_reference_get (SchroEncoder *encoder, int frame_number)
{
  int i;
  SCHRO_DEBUG("getting %d", frame_number);
  for(i=0;i<encoder->n_reference_frames;i++){
    if (encoder->reference_frames[i]->frame_number == frame_number) {
      return encoder->reference_frames[i];
    }
  }
  return NULL;

}

static void
schro_encoder_reference_retire (SchroEncoder *encoder, int frame_number)
{
  int i;
  SCHRO_DEBUG("retiring %d", frame_number);
  for(i=0;i<encoder->n_reference_frames;i++){
    if (encoder->reference_frames[i]->frame_number == frame_number) {
      schro_frame_free (encoder->reference_frames[i]);
      memmove (encoder->reference_frames + i, encoder->reference_frames + i + 1,
          sizeof(SchroFrame *)*(encoder->n_reference_frames - i - 1));
      encoder->n_reference_frames--;
      return;
    }
  }
}

