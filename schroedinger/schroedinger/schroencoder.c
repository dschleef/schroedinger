
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
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
void schro_encoder_encode_picture_header (SchroEncoder *encoder);

#define DIRAC_COMPAT 1


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

  encoder->version_major = 0;
  encoder->version_minor = 0;
  encoder->profile = 0;
  encoder->level = 0;
  encoder->need_rap = TRUE;

  params = &encoder->params;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;
  params->transform_depth = 4;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->wavelet_filter_index = SCHRO_WAVELET_5_3;

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

SchroVideoFormat *
schro_encoder_get_video_format (SchroEncoder *encoder)
{
  SchroVideoFormat *format;

  format = malloc(sizeof(SchroVideoFormat));
  memcpy (format, &encoder->video_format, sizeof(SchroVideoFormat));

  return format;
}

void
schro_encoder_set_video_format (SchroEncoder *encoder,
    SchroVideoFormat *format)
{
  /* FIXME check that we're in the right state to do this */
  memcpy (&encoder->video_format, format, sizeof(SchroVideoFormat));

  SCHRO_ERROR("wxh %d %d", format->width, format->height);
  encoder->video_format_index =
    schro_params_get_video_format (&encoder->video_format);

}

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

static void
schro_encoder_choose_quantisers (SchroEncoder *encoder)
{
  if (encoder->picture->is_ref) {
    encoder->base_quant = 24;
  } else {
    encoder->base_quant = 28;
  }
#if 0
  if ((encoder->picture->frame_number & 0x1f) > 0x10) {
    encoder->base_quant = 28;
  } else {
    encoder->base_quant = 24;
  }
#endif
  if (encoder->picture->is_ref) {
    encoder->encoder_params.quant_index_dc = 0;
    encoder->encoder_params.quant_index[0] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[1] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[2] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[3] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[4] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[5] = CLAMP(encoder->base_quant, 0, 63);
  } else {
    encoder->encoder_params.quant_index_dc = 0;
    encoder->encoder_params.quant_index[0] = CLAMP(encoder->base_quant, 0, 63);
    encoder->encoder_params.quant_index[1] = CLAMP(encoder->base_quant + 2, 0, 63);
    encoder->encoder_params.quant_index[2] = CLAMP(encoder->base_quant + 4, 0, 63);
    encoder->encoder_params.quant_index[3] = CLAMP(encoder->base_quant + 6, 0, 63);
    encoder->encoder_params.quant_index[4] = CLAMP(encoder->base_quant + 8, 0, 63);
    encoder->encoder_params.quant_index[5] = CLAMP(encoder->base_quant + 10, 0, 63);
    //encoder->encoder_params.quant_index_dc = CLAMP(encoder->base_quant, 0, 63);
  }
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

static void
schro_decoder_fixup_offsets (SchroEncoder *encoder, SchroBuffer *buffer)
{
  uint8_t *data = buffer->data;

  if (buffer->length < 11) {
    SCHRO_ERROR("packet too short");
  }

  data[5] = (buffer->length >> 16) & 0xff;
  data[6] = (buffer->length >> 8) & 0xff;
  data[7] = (buffer->length >> 0) & 0xff;
  data[8] = (encoder->prev_offset >> 16) & 0xff;
  data[9] = (encoder->prev_offset >> 8) & 0xff;
  data[10] = (encoder->prev_offset >> 0) & 0xff;

  encoder->prev_offset = buffer->length;
}

SchroBuffer *
schro_encoder_encode (SchroEncoder *encoder)
{
  SchroBuffer *outbuffer;
  SchroBuffer *subbuffer;
  int i;
  
  if (encoder->need_rap) {
SCHRO_ERROR("encoding rap");
    outbuffer = schro_buffer_new_and_alloc (0x100);

    encoder->bits = schro_bits_new ();
    schro_bits_encode_init (encoder->bits, outbuffer);

    schro_encoder_encode_access_unit_header (encoder);
    encoder->need_rap = FALSE;

    if (encoder->bits->offset > 0) {
      subbuffer = schro_buffer_new_subbuffer (outbuffer, 0,
          encoder->bits->offset/8);
    } else {
      subbuffer = NULL;
    }
    schro_bits_free (encoder->bits);
    schro_buffer_unref (outbuffer);

    schro_decoder_fixup_offsets (encoder, subbuffer);

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
  schro_params_set_default_codeblock (&encoder->params);
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

  if (encoder->bits->offset > 0) {
    subbuffer = schro_buffer_new_subbuffer (outbuffer, 0,
        encoder->bits->offset/8);
  } else {
    subbuffer = NULL;
  }
  schro_bits_free (encoder->bits);
  schro_buffer_unref (outbuffer);

  schro_decoder_fixup_offsets (encoder, subbuffer);

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
  SchroVideoFormat *format = &encoder->video_format;

  params->width = format->width;
  params->height = format->height;
  params->chroma_format = format->chroma_format;

  schro_params_calculate_iwt_sizes (params);

  if (encoder->tmp_frame0 == NULL) {
    encoder->tmp_frame0 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  schro_encoder_encode_parse_info (encoder,
      SCHRO_PARSE_CODE_PICTURE(encoder->picture->is_ref, encoder->picture->n_refs));
  schro_encoder_encode_picture_header (encoder);

  schro_frame_convert (encoder->tmp_frame0, encoder->encode_frame);
  schro_frame_shift_left (encoder->tmp_frame0, 4);

  schro_frame_free (encoder->encode_frame);

  schro_encoder_encode_transform_parameters (encoder);

  schro_frame_iwt_transform (encoder->tmp_frame0, &encoder->params,
      encoder->tmpbuf);

  schro_encoder_encode_transform_data (encoder, 0);
  schro_encoder_encode_transform_data (encoder, 1);
  schro_encoder_encode_transform_data (encoder, 2);

  schro_bits_sync (encoder->bits);

  if (encoder->picture->is_ref) {
    SchroFrame *ref_frame;

    schro_frame_inverse_iwt_transform (encoder->tmp_frame0, &encoder->params,
        encoder->tmpbuf);

    ref_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        format->width, format->height, 2, 2);
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

  schro_encoder_encode_parse_info (encoder,
      SCHRO_PARSE_CODE_PICTURE(encoder->picture->is_ref, encoder->picture->n_refs));
  schro_encoder_encode_picture_header (encoder);

  schro_encoder_motion_predict (encoder);

  schro_encoder_encode_frame_prediction (encoder);

  schro_frame_convert (encoder->tmp_frame0, encoder->encode_frame);
  schro_frame_free (encoder->encode_frame);

  schro_frame_copy_with_motion (encoder->tmp_frame1,
      encoder->ref_frame0, encoder->ref_frame1, encoder->motion_vectors,
      &encoder->params);

  schro_frame_subtract (encoder->tmp_frame0, encoder->tmp_frame1);

  schro_frame_zero_extend (encoder->tmp_frame0, encoder->video_format.width,
      encoder->video_format.height);

  schro_encoder_encode_transform_parameters (encoder);

  schro_frame_shift_left (encoder->tmp_frame0, 4);
  schro_frame_iwt_transform (encoder->tmp_frame0, &encoder->params,
      encoder->tmpbuf);
  residue_bits_start = encoder->bits->offset;
  schro_encoder_encode_transform_data (encoder, 0);
  schro_encoder_encode_transform_data (encoder, 1);
  schro_encoder_encode_transform_data (encoder, 2);

  schro_bits_sync (encoder->bits);

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
        encoder->video_format.width, encoder->video_format.height, 2, 2);
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

  schro_bits_sync(encoder->bits);

  /* block params flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* mv precision flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* global motion flag */
  global_motion = FALSE;
  schro_bits_encode_bit (encoder->bits, global_motion);

  /* picture prediction mode flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

  /* non-default weights flag */
  schro_bits_encode_bit (encoder->bits, FALSE);

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
      _schro_arith_encode_mode (arith, SCHRO_CTX_SPLIT_0, SCHRO_CTX_SPLIT_1,
          split_residual);

      if (global_motion) {
        _schro_arith_context_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
            mb_using_global);
      } else {
        SCHRO_ASSERT(mb_using_global == FALSE);
      }
      if (!mb_using_global) {
        SCHRO_ASSERT(mv->split < 3);
      } else {
        SCHRO_ASSERT(mv->split == 2);
      }
      if (mv->split != 0) {
        _schro_arith_context_encode_bit (arith, SCHRO_CTX_COMMON,
            mb_common);
      }

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &encoder->motion_vectors[(j+l)*(4*params->x_num_mb) + i + k];

          _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
              mv->pred_mode & 1);
          _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
              mv->pred_mode >> 1);
          if (mv->pred_mode == 0) {
            int pred[3];

            schro_motion_dc_prediction (encoder->motion_vectors,
                params, i+k, j+l, pred);

            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                SCHRO_CTX_LUMA_DC_SIGN,
                mv->dc[0] - pred[0]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                SCHRO_CTX_CHROMA1_DC_SIGN,
                mv->dc[1] - pred[1]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
                SCHRO_CTX_CHROMA2_DC_SIGN,
                mv->dc[2] - pred[2]);
          } else {
            int pred_x, pred_y;

            schro_motion_vector_prediction (encoder->motion_vectors,
                params, i+k, j+l, &pred_x, &pred_y);

            _schro_arith_context_encode_sint(arith,
                SCHRO_CTX_MV_REF1_H_CONT_BIN1,
                SCHRO_CTX_MV_REF1_H_VALUE,
                SCHRO_CTX_MV_REF1_H_SIGN,
                mv->x - pred_x);
            _schro_arith_context_encode_sint(arith,
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

  schro_bits_sync (encoder->bits);
  schro_bits_encode_uint(encoder->bits, arith->offset);

  schro_bits_sync (encoder->bits);
  schro_bits_append (encoder->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}


void
schro_encoder_encode_access_unit_header (SchroEncoder *encoder)
{
  SchroVideoFormat *format = &encoder->video_format;
  SchroVideoFormat _std_format;
  SchroVideoFormat *std_format = &_std_format;
  int i;

  schro_encoder_encode_parse_info (encoder, SCHRO_PARSE_CODE_ACCESS_UNIT);
  
  /* parse parameters */
  /* FIXME au picture number */
  schro_bits_encode_bits (encoder->bits, 32, 0);

  schro_bits_encode_uint (encoder->bits, encoder->version_major);
  schro_bits_encode_uint (encoder->bits, encoder->version_minor);
  schro_bits_encode_uint (encoder->bits, encoder->profile);
  schro_bits_encode_uint (encoder->bits, encoder->level);

  /* sequence parameters */
  schro_bits_encode_uint (encoder->bits, encoder->video_format_index);
  schro_params_set_video_format (std_format, encoder->video_format_index);

  if (std_format->width == format->width &&
      std_format->height == format->height) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_uint (encoder->bits, format->width);
    schro_bits_encode_uint (encoder->bits, format->height);
  }

  if (std_format->chroma_format == format->chroma_format) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_uint (encoder->bits, format->chroma_format);
  }

  if (std_format->video_depth == format->video_depth) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_uint (encoder->bits, format->video_depth);
  }

  /* source parameters */
  /* rather than figure out all the logic to encode this optimally, punt. */
  schro_bits_encode_bit (encoder->bits, TRUE);
  schro_bits_encode_bit (encoder->bits, format->interlaced_source);
  if(format->interlaced_source) {
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_bit (encoder->bits, format->top_field_first);
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_bit (encoder->bits, format->sequential_fields);
  }

  /* frame rate */
  if (std_format->frame_rate_numerator == format->frame_rate_numerator &&
      std_format->frame_rate_denominator == format->frame_rate_denominator) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    i = schro_params_get_frame_rate (format);
    schro_bits_encode_uint (encoder->bits, i);
    if (i==0) {
      schro_bits_encode_uint (encoder->bits, format->frame_rate_numerator);
      schro_bits_encode_uint (encoder->bits, format->frame_rate_denominator);
    }
  }

  /* pixel aspect ratio */
  if (std_format->aspect_ratio_numerator == format->aspect_ratio_numerator &&
      std_format->aspect_ratio_denominator == format->aspect_ratio_denominator) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    i = schro_params_get_aspect_ratio (format);
    schro_bits_encode_uint (encoder->bits, i);
    if (i==0) {
      schro_bits_encode_uint (encoder->bits, format->aspect_ratio_numerator);
      schro_bits_encode_uint (encoder->bits, format->aspect_ratio_denominator);
    }
  }

  /* clean area */
  if (std_format->clean_width == format->clean_width &&
      std_format->clean_height == format->clean_height &&
      std_format->left_offset == format->left_offset &&
      std_format->top_offset == format->top_offset) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    schro_bits_encode_uint (encoder->bits, format->clean_width);
    schro_bits_encode_uint (encoder->bits, format->clean_height);
    schro_bits_encode_uint (encoder->bits, format->left_offset);
    schro_bits_encode_uint (encoder->bits, format->top_offset);
  }

  /* signal range */
  if (std_format->luma_offset == format->luma_offset &&
      std_format->luma_excursion == format->luma_excursion &&
      std_format->chroma_offset == format->chroma_offset &&
      std_format->chroma_excursion == format->chroma_excursion) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    i = schro_params_get_signal_range (format);
    schro_bits_encode_uint (encoder->bits, i);
    if (i == 0) {
      schro_bits_encode_uint (encoder->bits, format->luma_offset);
      schro_bits_encode_uint (encoder->bits, format->luma_excursion);
      schro_bits_encode_uint (encoder->bits, format->chroma_offset);
      schro_bits_encode_uint (encoder->bits, format->chroma_excursion);
    }
  }

  /* colour spec */
  if (std_format->colour_primaries == format->colour_primaries &&
      std_format->colour_matrix == format->colour_matrix &&
      std_format->transfer_function == format->transfer_function) {
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
    schro_bits_encode_bit (encoder->bits, TRUE);
    i = schro_params_get_colour_spec (format);
    schro_bits_encode_uint (encoder->bits, i);
    if (i == 0) {
      schro_bits_encode_bit (encoder->bits, TRUE);
      schro_bits_encode_uint (encoder->bits, format->colour_primaries);
      schro_bits_encode_bit (encoder->bits, TRUE);
      schro_bits_encode_uint (encoder->bits, format->colour_matrix);
      schro_bits_encode_bit (encoder->bits, TRUE);
      schro_bits_encode_uint (encoder->bits, format->transfer_function);
    }
  }

  schro_bits_sync (encoder->bits);
}

void
schro_encoder_encode_parse_info (SchroEncoder *encoder,
    int parse_code)
{
  /* parse parameters */
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'B');
  schro_bits_encode_bits (encoder->bits, 8, 'C');
  schro_bits_encode_bits (encoder->bits, 8, 'D');
  schro_bits_encode_bits (encoder->bits, 8, parse_code);

  /* offsets */
  schro_bits_encode_bits (encoder->bits, 24, 0);
  schro_bits_encode_bits (encoder->bits, 24, 0);
}

void
schro_encoder_encode_picture_header (SchroEncoder *encoder)
{
  int i;

  schro_bits_sync(encoder->bits);
  schro_bits_encode_bits (encoder->bits, 32, encoder->picture->frame_number);

  for(i=0;i<encoder->picture->n_refs;i++){
    schro_bits_encode_sint (encoder->bits,
        encoder->picture->reference_frame_number[i] -
        encoder->picture->frame_number);
  }

  /* retire list */
  schro_bits_encode_uint (encoder->bits, encoder->picture->n_retire);
  for(i=0;i<encoder->picture->n_retire;i++){
    schro_bits_encode_sint (encoder->bits,
        encoder->picture->retire[i] - encoder->picture->frame_number);
  }
}


void
schro_encoder_encode_transform_parameters (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;

  if (params->num_refs > 0) {
    /* zero residual */
    schro_bits_encode_bit (encoder->bits, FALSE);
  } else {
#ifdef DIRAC_COMPAT
    schro_bits_sync (encoder->bits);
#endif
  }

  /* transform */
  if (params->wavelet_filter_index == SCHRO_WAVELET_DESL_9_3) {
    schro_bits_encode_bit (encoder->bits, 0);
  } else {
    schro_bits_encode_bit (encoder->bits, 1);
    schro_bits_encode_uint (encoder->bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    schro_bits_encode_bit (encoder->bits, 0);
  } else {
    schro_bits_encode_bit (encoder->bits, 1);
    schro_bits_encode_uint (encoder->bits, params->transform_depth);
  }

  /* spatial partitioning */
  schro_bits_encode_bit (encoder->bits, params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    schro_bits_encode_bit (encoder->bits, params->nondefault_partition_flag);
    if (params->nondefault_partition_flag) {
      int i;

      for(i=0;i<params->transform_depth+1;i++){
        schro_bits_encode_uint (encoder->bits, params->horiz_codeblocks[i]);
        schro_bits_encode_uint (encoder->bits, params->vert_codeblocks[i]);
      }
    }
    schro_bits_encode_uint (encoder->bits, params->codeblock_mode_index);
  }
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
    w = ROUND_UP_SHIFT(encoder->video_format.width, shift);
    height = subband->h;
    h = ROUND_UP_SHIFT(encoder->video_format.height, shift);
    offset = subband->offset;
  } else {
    stride = subband->chroma_stride >> 1;
    width = subband->chroma_w;
    w = ROUND_UP_SHIFT(encoder->video_format.width/2, shift);
    height = subband->chroma_h;
    h = ROUND_UP_SHIFT(encoder->video_format.height/2, shift);
    offset = subband->chroma_offset;
  }

  data = (int16_t *)encoder->tmp_frame0->components[component].data + offset;

  SCHRO_DEBUG("subband index=%d %d x %d at offset %d with stride %d; clean area %d %d", index,
      width, height, offset, stride, w, h);

  /* FIXME this is dependent on the particular wavelet transform */
  if (h < height) h+=1;
  if (w < width) w+=1;

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
  int horiz_codeblocks;
  int vert_codeblocks;
  int have_zero_flags;

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
  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_bits_sync (encoder->bits);
    schro_bits_encode_uint (encoder->bits, 0);
    schro_bits_sync (encoder->bits);
    schro_arith_free (arith);
    free (quant_data);
    return;
  }

  if (params->spatial_partition_flag) {
    horiz_codeblocks = params->horiz_codeblocks[subband->scale_factor_shift];
    vert_codeblocks = params->vert_codeblocks[subband->scale_factor_shift];
  } else {
    horiz_codeblocks = 1;
    vert_codeblocks = 1;
  }
  if (horiz_codeblocks > 1 && vert_codeblocks > 1 && index > 0) {
    have_zero_flags = TRUE;
  } else {
    have_zero_flags = FALSE;
  }
  for(y=0;y<vert_codeblocks;y++){
    int ymin = (height*y)/vert_codeblocks;
    int ymax = (height*(y+1))/vert_codeblocks;

    for(x=0;x<horiz_codeblocks;x++){
      int xmin = (width*x)/horiz_codeblocks;
      int xmax = (width*(x+1))/horiz_codeblocks;

  if (have_zero_flags) {
    int zero_codeblock = 1;
    for(j=ymin;j<ymax;j++){
      for(i=xmin;i<xmax;i++){
        if (quant_data[j*width + i] != 0) {
          zero_codeblock = 0;
        }
      }
    }
    _schro_arith_context_encode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK, zero_codeblock);
    if (zero_codeblock) {
      continue;
    }
  }

  for(j=ymin;j<ymax;j++){
    for(i=xmin;i<xmax;i++){
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

      _schro_arith_context_encode_sint (arith, cont_context, value_context,
          sign_context, quant_data[j*width + i]);
    }
  }
    }
  }
  free (quant_data);

  schro_arith_flush (arith);

#ifdef DIRAC_COMPAT
  schro_bits_sync (encoder->bits);
#endif
  schro_bits_encode_uint (encoder->bits, arith->offset);
  if (arith->offset > 0) {
    schro_bits_encode_uint (encoder->bits, subband->quant_index);

    schro_bits_sync (encoder->bits);

    schro_bits_append (encoder->bits, arith->buffer->data, arith->offset);
  }
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

      _schro_arith_context_encode_sint (arith, table[table_index][0],
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

      _schro_arith_context_encode_sint (arith, table[table_index][0],
          table[table_index][1], table[table_index][2], quant_data[i]);
    }
  }
}


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

