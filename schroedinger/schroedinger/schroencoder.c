
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void schro_encoder_frame_queue_push (SchroEncoder *encoder,
    SchroFrame *frame);
static void schro_encoder_reference_analyse (SchroEncoderFrame *ref);
static void schro_encoder_reference_retire (SchroEncoder *encoder,
    int frame_number);
static void schro_encoder_engine_init (SchroEncoder *encoder);
static void schro_encoder_encode_frame_prediction (SchroEncoderTask *task);
static void schro_encoder_encode_transform_parameters (SchroEncoderTask *task);
static void schro_encoder_encode_transform_data (SchroEncoderTask *task, int component);
static int schro_encoder_pull_is_ready (SchroEncoder *encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder *encoder);

static void
schro_encoder_frame_free (SchroEncoderFrame *encoder_frame);


SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;

  encoder = malloc(sizeof(SchroEncoder));
  memset (encoder, 0, sizeof(SchroEncoder));

  encoder->version_major = 0;
  encoder->version_minor = 0;
  encoder->profile = 0;
  encoder->level = 0;

  encoder->au_frame = -1;
  encoder->au_distance = 24;

  encoder->last_ref = -1;
  encoder->next_ref = -1;
  encoder->mid1_ref = -1;
  encoder->mid2_ref = -1;

  encoder->prefs[SCHRO_PREF_ENGINE] = 2;
  encoder->prefs[SCHRO_PREF_REF_DISTANCE] = 4;
  encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 4;
  encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_DESL_9_3;
  encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_5_3;
  encoder->prefs[SCHRO_PREF_QUANT_BASE] = 4;
  encoder->prefs[SCHRO_PREF_QUANT_OFFSET_NONREF] = 8;
  encoder->prefs[SCHRO_PREF_QUANT_OFFSET_SUBBAND] = -1;
  encoder->prefs[SCHRO_PREF_QUANT_DC] = 0;
  encoder->prefs[SCHRO_PREF_QUANT_DC_OFFSET_NONREF] = 0;

  schro_params_set_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_SD576);

  schro_encoder_encode_codec_comment (encoder);

  /* FIXME */
  encoder->queue_depth = 4;

  return encoder;
}

void
schro_encoder_free (SchroEncoder *encoder)
{
  int i;
  int j;

  if (encoder->async) {
    schro_async_free(encoder->async);
  }

  for(i=0;i<encoder->n_reference_frames; i++) {
    SchroEncoderFrame *ref = encoder->reference_frames + i;
    if (ref->valid) {
      for(j=0;j<5;j++){
        schro_frame_free (ref->frames[j]);
      }
    }
  }
  for(i=0;i<encoder->frame_queue_length; i++) {
    schro_encoder_frame_free(encoder->frame_queue[i]);
  }

  if (encoder->inserted_buffer) {
    schro_buffer_unref (encoder->inserted_buffer);
  }

  free (encoder);
}

SchroEncoderTask *
schro_encoder_task_new (SchroEncoder *encoder)
{
  SchroEncoderTask *task;
  SchroParams *params;

  task = malloc(sizeof(SchroEncoderTask));
  memset (task, 0, sizeof(*task));

  task->encoder = encoder;

  task->tmpbuf = malloc(SCHRO_LIMIT_WIDTH * 2);
  task->tmpbuf2 = malloc(SCHRO_LIMIT_WIDTH * 2);

  task->subband_size = encoder->video_format.width *
    encoder->video_format.height / 4;
  task->subband_buffer = schro_buffer_new_and_alloc (task->subband_size);

  /* FIXME settings */
  params = &task->params;
  params->video_format = &encoder->video_format;
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;

  /* calculate sizes */
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  /* FIXME these should allocate based on the max allowable for the
   * given video format and profile */
  if (task->tmp_frame0 == NULL) {
    task->tmp_frame0 = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height,
        params->iwt_chroma_width, params->iwt_chroma_height);
  }
  if (task->tmp_frame1 == NULL) {
    task->tmp_frame1 = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_S16,
        params->mc_luma_width, params->mc_luma_height,
        params->mc_chroma_width, params->mc_chroma_height);
  }
  if (task->quant_data == NULL) {
    task->quant_data = malloc (sizeof(int16_t) *
        (params->iwt_luma_width/2) * (params->iwt_luma_height/2));
  }

  return task;
}

void
schro_encoder_task_free (SchroEncoderTask *task)
{
  if (task->tmp_frame0) {
    schro_frame_free (task->tmp_frame0);
  }
  if (task->tmp_frame1) {
    schro_frame_free (task->tmp_frame1);
  }
  if (task->motion_field) {
    schro_motion_field_free (task->motion_field);
  }
  if (task->subband_buffer) {
    schro_buffer_unref (task->subband_buffer);
  }
  if (task->quant_data) {
    free (task->quant_data);
  }

  free (task->tmpbuf);
  free (task->tmpbuf2);
  free (task);
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

  schro_params_validate (&encoder->video_format);

  SCHRO_DEBUG("wxh %d %d", format->width, format->height);
  encoder->video_format_index =
    schro_params_get_video_format (&encoder->video_format);
}

int
schro_encoder_push_ready (SchroEncoder *encoder)
{
  if (!encoder->end_of_stream &&
      encoder->frame_queue_length < encoder->queue_depth) {
    return TRUE;
  }
  return FALSE;
}

void
schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame)
{
  frame->frame_number = encoder->frame_queue_index;
  encoder->frame_queue_index++;

  schro_encoder_frame_queue_push (encoder, frame);

  encoder->queue_changed = TRUE;
}

void
schro_encoder_end_of_stream (SchroEncoder *encoder)
{
  encoder->end_of_stream = TRUE;
  if (encoder->frame_queue_length > 0) {
    encoder->frame_queue[encoder->frame_queue_length-1]->last_frame = TRUE;
  }
}

static int
schro_gain_to_index (int value)
{
  value = (value + 8)>>4;
  return CLAMP(value, 0, 63);
}

static void
schro_encoder_choose_quantisers (SchroEncoderTask *task)
{
  /* This is 64*log2 of the gain of the DC part of the wavelet transform */
  static const int wavelet_gain[] = { 64, 64, 64, 0, 64, 128, 128, 103 };
  /* horizontal/vertical part */
  static const int wavelet_gain_hv[] = { 64, 64, 64, 0, 64, 128, 0, 65 };
  /* diagonal part */
  static const int wavelet_gain_diag[] = { 128, 128, 128, 64, 128, 256, -64, 90 };
  SchroSubband *subbands = task->subbands;
  int base;
  int gain;
  int gain_hv;
  int gain_diag;
  int percep;
  int depth;
  int band;
  int dc;
  int i;

  depth = task->params.transform_depth;
  gain = wavelet_gain[task->params.wavelet_filter_index];
  gain_hv = wavelet_gain_hv[task->params.wavelet_filter_index];
  gain_diag = wavelet_gain_diag[task->params.wavelet_filter_index];

  base = task->encoder->prefs[SCHRO_PREF_QUANT_BASE]<<4;
  dc = task->encoder->prefs[SCHRO_PREF_QUANT_DC]<<4;
  percep = task->encoder->prefs[SCHRO_PREF_QUANT_OFFSET_SUBBAND]<<4;
  if (!task->is_ref) {
    base += task->encoder->prefs[SCHRO_PREF_QUANT_OFFSET_NONREF]<<4;
    dc += task->encoder->prefs[SCHRO_PREF_QUANT_DC_OFFSET_NONREF]<<4;
  }

  subbands[0].quant_index = schro_gain_to_index (dc);
  for(i=0; i<depth; i++) {
    band = depth - 1 - i;
    subbands[1+3*i].quant_index =
      schro_gain_to_index (base + (percep + gain)*band + gain_diag);
    subbands[2+3*i].quant_index =
      schro_gain_to_index (base + (percep + gain)*band + gain_hv);
    subbands[3+3*i].quant_index =
      schro_gain_to_index (base + (percep + gain)*band + gain_hv);
  }
}

static void
schro_encoder_fixup_offsets (SchroEncoder *encoder, SchroBuffer *buffer)
{
  uint8_t *data = buffer->data;

  if (buffer->length < 11) {
    SCHRO_ERROR("packet too short (%d < 11)", buffer->length);
  }

  data[5] = (buffer->length >> 16) & 0xff;
  data[6] = (buffer->length >> 8) & 0xff;
  data[7] = (buffer->length >> 0) & 0xff;
  data[8] = (encoder->prev_offset >> 16) & 0xff;
  data[9] = (encoder->prev_offset >> 8) & 0xff;
  data[10] = (encoder->prev_offset >> 0) & 0xff;

  encoder->prev_offset = buffer->length;
}

static void
schro_encoder_encode_codec_comment (SchroEncoder *encoder)
{
  char *s = "bbcdencoSchrodinger " VERSION;
  SchroBuffer *buffer;

  buffer = schro_encoder_encode_auxiliary_data (encoder, s, strlen(s));
  
  schro_encoder_insert_buffer (encoder, buffer);
}

void
schro_encoder_insert_buffer (SchroEncoder *encoder, SchroBuffer *buffer)
{
  if (encoder->inserted_buffer) {
    SCHRO_ERROR("dropping previously inserted buffer");
    schro_buffer_unref (encoder->inserted_buffer);
  }
  encoder->inserted_buffer = buffer;
}

SchroBuffer *
schro_encoder_encode_auxiliary_data (SchroEncoder *encoder, void *data,
    int size)
{
  SchroBits *bits;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (size + SCHRO_PARSE_HEADER_SIZE);

  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);

  schro_encoder_encode_parse_info (bits, SCHRO_PARSE_CODE_AUXILIARY_DATA);
  schro_bits_append (bits, data, size);

  return buffer;
}

SchroBuffer *
schro_encoder_encode_access_unit (SchroEncoder *encoder)
{
  SchroBits *bits;
  SchroBuffer *buffer;
  SchroBuffer *subbuffer;

  buffer = schro_buffer_new_and_alloc (0x100);

  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);

  schro_encoder_encode_access_unit_header (encoder, bits);

  subbuffer = schro_buffer_new_subbuffer (buffer, 0, bits->offset/8);
  schro_bits_free (bits);
  schro_buffer_unref (buffer);

  return subbuffer;
}

SchroBuffer *
schro_encoder_encode_end_of_stream (SchroEncoder *encoder)
{
  SchroBits *bits;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (11);

  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);

  schro_encoder_encode_parse_info (bits, SCHRO_PARSE_CODE_END_SEQUENCE);

  return buffer;
}

static void
schro_encoder_task_complete (SchroEncoderTask *task)
{
  int i;
  SchroEncoderFrame *frame;

  SCHRO_DEBUG("completing picture %d", task->frame_number);

  frame = task->encoder_frame;

  frame->state = SCHRO_ENCODER_FRAME_STATE_DONE;

  task->encoder->queue_changed = TRUE;

  frame->output_buffer = task->outbuffer;
  frame->presentation_frame = task->presentation_frame;

  SCHRO_INFO("PICTURE: %d %d %d %d",
      task->frame_number, task->is_ref, task->params.num_refs, task->bits->offset);

  if (frame->start_access_unit) {
    frame->access_unit_buffer = schro_encoder_encode_access_unit (task->encoder);
  }
  if (frame->last_frame) {
    /* FIXME push an EOS */
    task->encoder->completed_eos = TRUE;
  }

  for(i=0;i<task->n_retire;i++){
    schro_encoder_reference_retire (task->encoder, task->retire[i]);
  }
}

int
schro_encoder_iterate (SchroEncoder *encoder)
{
  int ret = FALSE;

  SCHRO_DEBUG("iterate");

  if (encoder->end_of_stream_pulled) {
    return SCHRO_STATE_END_OF_STREAM;
  }

  if (!encoder->engine_init) {
    schro_encoder_engine_init (encoder);
  }

  if (schro_async_get_num_completed (encoder->async) > 0) {
    SchroEncoderTask *task;

    task = schro_async_pull (encoder->async);
    SCHRO_ASSERT(task != NULL);

    schro_encoder_task_complete (task);
    schro_encoder_task_free (task);
  }

  SCHRO_DEBUG("iterate %d %d %d",
      encoder->queue_changed, schro_encoder_push_ready(encoder),
      encoder->completed_eos);
  while (!encoder->queue_changed && !schro_encoder_push_ready(encoder) &&
      !encoder->completed_eos) {
    SchroEncoderTask *task;

    schro_async_wait_one (encoder->async);

    task = schro_async_pull (encoder->async);
    SCHRO_ASSERT(task != NULL);

    schro_encoder_task_complete (task);
    schro_encoder_task_free (task);
  }

  if (encoder->queue_changed) {
    switch (encoder->engine) {
      case 0:
        ret = schro_encoder_engine_intra_only (encoder);
        break;
      case 1:
        ret = schro_encoder_engine_backref (encoder);
        break;
      case 2:
        ret = schro_encoder_engine_backref2 (encoder);
        break;
      default:
        ret = FALSE;
        break;
    }
    if (!ret) {
      encoder->queue_changed = FALSE;
    }
  }

  if (schro_encoder_pull_is_ready (encoder)) {
    return SCHRO_STATE_HAVE_BUFFER;
  }
  if (schro_encoder_push_ready (encoder)) {
    return SCHRO_STATE_NEED_FRAME;
  }

  return SCHRO_STATE_AGAIN;
}

SchroBuffer *
schro_encoder_encode (SchroEncoder *encoder)
{
  SchroBuffer *buffer;

  schro_encoder_iterate (encoder);

  buffer = schro_encoder_pull (encoder, NULL);
  return buffer;
}

static void
schro_encoder_engine_init (SchroEncoder *encoder)
{
  int i;
  int j;

  encoder->engine_init = 1;

  encoder->engine = encoder->prefs[SCHRO_PREF_ENGINE];
  encoder->ref_distance = encoder->prefs[SCHRO_PREF_REF_DISTANCE];

  encoder->n_reference_frames = 4;
  for(i=0;i<encoder->n_reference_frames;i++){
    SchroEncoderFrame *ref = encoder->reference_frames + i;

    memset(ref, 0, sizeof(*ref));
    for(j=0;j<5;j++){
      ref->frames[j] = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8,
          ROUND_UP_SHIFT(encoder->video_format.width, j),
          ROUND_UP_SHIFT(encoder->video_format.height, j),
          ROUND_UP_SHIFT(encoder->video_format.width, j + 1),
          ROUND_UP_SHIFT(encoder->video_format.height, j + 1));
    }
  }

  encoder->async = schro_async_new (0);
}

void
schro_encoder_encode_picture (SchroEncoderTask *task)
{
  int residue_bits_start;
  SchroBuffer *subbuffer;

  task->bits = schro_bits_new ();
  schro_bits_encode_init (task->bits, task->outbuffer);

  /* encode header */
  schro_encoder_encode_parse_info (task->bits,
      SCHRO_PARSE_CODE_PICTURE(task->is_ref, task->params.num_refs));
  schro_encoder_encode_picture_header (task);

  if (task->params.num_refs > 0) {
    schro_encoder_motion_predict (task);

    schro_encoder_encode_frame_prediction (task);

    schro_frame_convert (task->tmp_frame0, task->encode_frame);

    if (task->params.num_refs == 2) {
      schro_frame_copy_with_motion (task->tmp_frame1,
          task->ref_frame0->frames[0], task->ref_frame1->frames[0],
          task->motion_field->motion_vectors, &task->params);
    } else {
      schro_frame_copy_with_motion (task->tmp_frame1,
          task->ref_frame0->frames[0], NULL,
          task->motion_field->motion_vectors, &task->params);
    }

    schro_frame_subtract (task->tmp_frame0, task->tmp_frame1);

    schro_frame_zero_extend (task->tmp_frame0,
        task->params.video_format->width,
        task->params.video_format->height);
  } else {
    schro_frame_convert (task->tmp_frame0, task->encode_frame);
  }
  schro_frame_free (task->encode_frame);

  schro_encoder_encode_transform_parameters (task);

  schro_frame_iwt_transform (task->tmp_frame0, &task->params,
      task->tmpbuf);
  residue_bits_start = task->bits->offset;
  schro_encoder_encode_transform_data (task, 0);
  schro_encoder_encode_transform_data (task, 1);
  schro_encoder_encode_transform_data (task, 2);

  schro_bits_sync (task->bits);

  subbuffer = schro_buffer_new_subbuffer (task->outbuffer, 0,
      task->bits->offset/8);
  schro_buffer_unref (task->outbuffer);
  task->outbuffer = subbuffer;

  if (task->params.num_refs > 0) {
#if 0
    task->metric_to_cost =
      (double)(task->bits->offset - residue_bits_start) /
      task->stats_metric;
#endif
    SCHRO_INFO("pred bits %d, residue bits %d, dc %d, global = %d, motion %d",
        residue_bits_start, task->bits->offset - residue_bits_start,
        task->stats_dc, task->stats_global, task->stats_motion);
  }

  if (task->is_ref) {
    schro_frame_inverse_iwt_transform (task->tmp_frame0, &task->params,
        task->tmpbuf);

    schro_frame_convert (task->dest_ref->frames[0], task->tmp_frame0);
    task->dest_ref->frame_number = task->frame_number;

    schro_encoder_reference_analyse (task->dest_ref);
  }

  schro_bits_free (task->bits);

  task->completed = TRUE;
}

static void
schro_encoder_encode_frame_prediction (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i,j;
  SchroArith *arith;
  int superblock_count = 0;

  schro_bits_sync(task->bits);

  /* block params flag */
  schro_bits_encode_bit (task->bits, FALSE);

  /* mv precision flag */
  schro_bits_encode_bit (task->bits, FALSE);

  /* global motion flag */
  schro_bits_encode_bit (task->bits, params->have_global_motion);
  if (params->have_global_motion) {
    int i;
    for(i=0;i<params->num_refs;i++){
      SchroGlobalMotion *gm = params->global_motion + i;

      if (gm->b0 == 0 && gm->b1 == 0) {
        schro_bits_encode_bit (task->bits, 0);
      } else {
        schro_bits_encode_bit (task->bits, 1);
        schro_bits_encode_sint (task->bits, gm->b0);
        schro_bits_encode_sint (task->bits, gm->b1);
      }

      if (gm->a_exp == 0 && gm->a00 == 1 && gm->a01 == 0 && gm->a10 == 0 &&
          gm->a11 == 1) {
        schro_bits_encode_bit (task->bits, 0);
      } else {
        schro_bits_encode_bit (task->bits, 1);
        schro_bits_encode_uint (task->bits, gm->a_exp);
        schro_bits_encode_sint (task->bits, gm->a00);
        schro_bits_encode_sint (task->bits, gm->a01);
        schro_bits_encode_sint (task->bits, gm->a10);
        schro_bits_encode_sint (task->bits, gm->a11);
      }

      if (gm->c_exp == 0 && gm->c0 == 0 && gm->c1 == 0) {
        schro_bits_encode_bit (task->bits, 0);
      } else {
        schro_bits_encode_bit (task->bits, 1);
        schro_bits_encode_uint (task->bits, gm->c_exp);
        schro_bits_encode_sint (task->bits, gm->c0);
        schro_bits_encode_sint (task->bits, gm->c1);
      }
    }
  }

  /* picture prediction mode flag */
  schro_bits_encode_bit (task->bits, FALSE);

  /* non-default weights flag */
  schro_bits_encode_bit (task->bits, FALSE);

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, task->subband_buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
        &task->motion_field->motion_vectors[j*params->x_num_blocks + i];

      SCHRO_ASSERT(mv->split < 3);

      split_prediction = schro_motion_split_prediction (
          task->motion_field->motion_vectors, params, i, j);
      split_residual = (mv->split - split_prediction + 3)%3;
      _schro_arith_context_encode_uint (arith, SCHRO_CTX_SB_F1,
          SCHRO_CTX_SB_DATA, split_residual);

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &task->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];
          int pred_mode;

          pred_mode = schro_motion_get_mode_prediction(task->motion_field,
              i+k,j+l) ^ mv->pred_mode;

          _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
              pred_mode & 1);
          if (params->num_refs > 1) {
            _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
                pred_mode >> 1);
          }
          if (mv->pred_mode == 0) {
            int pred[3];

            schro_motion_dc_prediction (task->motion_field->motion_vectors,
                params, i+k, j+l, pred);

            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                SCHRO_CTX_LUMA_DC_SIGN,
                mv->u.dc[0] - pred[0]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                SCHRO_CTX_CHROMA1_DC_SIGN,
                mv->u.dc[1] - pred[1]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
                SCHRO_CTX_CHROMA2_DC_SIGN,
                mv->u.dc[2] - pred[2]);
          } else {
            int pred_x, pred_y;

            if (params->have_global_motion) {
              int pred;
              schro_motion_field_get_global_prediction (task->motion_field,
                  i+k, j+l, &pred);
              _schro_arith_context_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
                  mv->using_global ^ pred);
            } else {
              SCHRO_ASSERT(mv->using_global == FALSE);
            }
            if (!mv->using_global) {
              schro_motion_vector_prediction (task->motion_field->motion_vectors,
                  params, i+k, j+l, &pred_x, &pred_y, mv->pred_mode);

              _schro_arith_context_encode_sint(arith,
                  SCHRO_CTX_MV_REF1_H_CONT_BIN1,
                  SCHRO_CTX_MV_REF1_H_VALUE,
                  SCHRO_CTX_MV_REF1_H_SIGN,
                  mv->u.xy.x - pred_x);
              _schro_arith_context_encode_sint(arith,
                  SCHRO_CTX_MV_REF1_V_CONT_BIN1,
                  SCHRO_CTX_MV_REF1_V_VALUE,
                  SCHRO_CTX_MV_REF1_V_SIGN,
                  mv->u.xy.y - pred_y);
            }
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

  schro_bits_sync (task->bits);
  schro_bits_encode_uint(task->bits, arith->offset);

  schro_bits_sync (task->bits);
  schro_bits_append (task->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}


void
schro_encoder_encode_access_unit_header (SchroEncoder *encoder,
    SchroBits *bits)
{
  SchroVideoFormat *format = &encoder->video_format;
  SchroVideoFormat _std_format;
  SchroVideoFormat *std_format = &_std_format;
  int i;

  schro_encoder_encode_parse_info (bits, SCHRO_PARSE_CODE_ACCESS_UNIT);
  
  /* parse parameters */
  schro_bits_encode_bits (bits, 32, encoder->au_frame);

  schro_bits_encode_uint (bits, encoder->version_major);
  schro_bits_encode_uint (bits, encoder->version_minor);
  schro_bits_encode_uint (bits, encoder->profile);
  schro_bits_encode_uint (bits, encoder->level);

  /* sequence parameters */
  schro_bits_encode_uint (bits, encoder->video_format_index);
  schro_params_set_video_format (std_format, encoder->video_format_index);

  if (std_format->width == format->width &&
      std_format->height == format->height) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_uint (bits, format->width);
    schro_bits_encode_uint (bits, format->height);
  }

  if (std_format->chroma_format == format->chroma_format) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_uint (bits, format->chroma_format);
  }

  if (std_format->video_depth == format->video_depth) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_uint (bits, format->video_depth);
  }

  /* source parameters */
  /* rather than figure out all the logic to encode this optimally, punt. */
  schro_bits_encode_bit (bits, TRUE);
  schro_bits_encode_bit (bits, format->interlaced_source);
  if(format->interlaced_source) {
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_bit (bits, format->top_field_first);
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_bit (bits, format->sequential_fields);
  }

  /* frame rate */
  if (std_format->frame_rate_numerator == format->frame_rate_numerator &&
      std_format->frame_rate_denominator == format->frame_rate_denominator) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    i = schro_params_get_frame_rate (format);
    schro_bits_encode_uint (bits, i);
    if (i==0) {
      schro_bits_encode_uint (bits, format->frame_rate_numerator);
      schro_bits_encode_uint (bits, format->frame_rate_denominator);
    }
  }

  /* pixel aspect ratio */
  if (std_format->aspect_ratio_numerator == format->aspect_ratio_numerator &&
      std_format->aspect_ratio_denominator == format->aspect_ratio_denominator) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    i = schro_params_get_aspect_ratio (format);
    schro_bits_encode_uint (bits, i);
    if (i==0) {
      schro_bits_encode_uint (bits, format->aspect_ratio_numerator);
      schro_bits_encode_uint (bits, format->aspect_ratio_denominator);
    }
  }

  /* clean area */
  if (std_format->clean_width == format->clean_width &&
      std_format->clean_height == format->clean_height &&
      std_format->left_offset == format->left_offset &&
      std_format->top_offset == format->top_offset) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    schro_bits_encode_uint (bits, format->clean_width);
    schro_bits_encode_uint (bits, format->clean_height);
    schro_bits_encode_uint (bits, format->left_offset);
    schro_bits_encode_uint (bits, format->top_offset);
  }

  /* signal range */
  if (std_format->luma_offset == format->luma_offset &&
      std_format->luma_excursion == format->luma_excursion &&
      std_format->chroma_offset == format->chroma_offset &&
      std_format->chroma_excursion == format->chroma_excursion) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    i = schro_params_get_signal_range (format);
    schro_bits_encode_uint (bits, i);
    if (i == 0) {
      schro_bits_encode_uint (bits, format->luma_offset);
      schro_bits_encode_uint (bits, format->luma_excursion);
      schro_bits_encode_uint (bits, format->chroma_offset);
      schro_bits_encode_uint (bits, format->chroma_excursion);
    }
  }

  /* colour spec */
  if (std_format->colour_primaries == format->colour_primaries &&
      std_format->colour_matrix == format->colour_matrix &&
      std_format->transfer_function == format->transfer_function) {
    schro_bits_encode_bit (bits, FALSE);
  } else {
    schro_bits_encode_bit (bits, TRUE);
    i = schro_params_get_colour_spec (format);
    schro_bits_encode_uint (bits, i);
    if (i == 0) {
      schro_bits_encode_bit (bits, TRUE);
      schro_bits_encode_uint (bits, format->colour_primaries);
      schro_bits_encode_bit (bits, TRUE);
      schro_bits_encode_uint (bits, format->colour_matrix);
      schro_bits_encode_bit (bits, TRUE);
      schro_bits_encode_uint (bits, format->transfer_function);
    }
  }

  schro_bits_sync (bits);
}

void
schro_encoder_encode_parse_info (SchroBits *bits,
    int parse_code)
{
  /* parse parameters */
  schro_bits_encode_bits (bits, 8, 'B');
  schro_bits_encode_bits (bits, 8, 'B');
  schro_bits_encode_bits (bits, 8, 'C');
  schro_bits_encode_bits (bits, 8, 'D');
  schro_bits_encode_bits (bits, 8, parse_code);

  /* offsets */
  schro_bits_encode_bits (bits, 24, 0);
  schro_bits_encode_bits (bits, 24, 0);
}

void
schro_encoder_encode_picture_header (SchroEncoderTask *task)
{
  int i;

  schro_bits_sync(task->bits);
  schro_bits_encode_bits (task->bits, 32, task->frame_number);

  for(i=0;i<task->params.num_refs;i++){
    schro_bits_encode_sint (task->bits,
        task->reference_frame_number[i] - task->frame_number);
  }

  /* retire list */
  schro_bits_encode_uint (task->bits, task->n_retire);
  for(i=0;i<task->n_retire;i++){
    schro_bits_encode_sint (task->bits, task->retire[i] - task->frame_number);
  }
}


static void
schro_encoder_encode_transform_parameters (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;

  if (params->num_refs > 0) {
    /* zero residual */
    schro_bits_encode_bit (task->bits, FALSE);
  } else {
#ifdef DIRAC_COMPAT
    schro_bits_sync (task->bits);
#endif
  }

  /* transform */
  if (params->wavelet_filter_index == SCHRO_WAVELET_DESL_9_3) {
    schro_bits_encode_bit (task->bits, 0);
  } else {
    schro_bits_encode_bit (task->bits, 1);
    schro_bits_encode_uint (task->bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    schro_bits_encode_bit (task->bits, 0);
  } else {
    schro_bits_encode_bit (task->bits, 1);
    schro_bits_encode_uint (task->bits, params->transform_depth);
  }

  /* spatial partitioning */
  schro_bits_encode_bit (task->bits, params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    schro_bits_encode_bit (task->bits, params->nondefault_partition_flag);
    if (params->nondefault_partition_flag) {
      int i;

      for(i=0;i<params->transform_depth+1;i++){
        schro_bits_encode_uint (task->bits, params->horiz_codeblocks[i]);
        schro_bits_encode_uint (task->bits, params->vert_codeblocks[i]);
      }
    }
    schro_bits_encode_uint (task->bits, params->codeblock_mode_index);
  }
}


void
schro_encoder_init_subbands (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i;
  int w;
  int h;
  int stride;
  int chroma_w;
  int chroma_h;
  int chroma_stride;
  SchroSubband *subbands = task->subbands;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = sizeof(int16_t)*(params->iwt_luma_width << params->transform_depth);
  chroma_w = params->iwt_chroma_width >> params->transform_depth;
  chroma_h = params->iwt_chroma_height >> params->transform_depth;
  chroma_stride = sizeof(int16_t)*(params->iwt_chroma_width << params->transform_depth);

  subbands[0].x = 0;
  subbands[0].y = 0;
  subbands[0].w = w;
  subbands[0].h = h;
  subbands[0].offset = 0;
  subbands[0].stride = stride;
  subbands[0].chroma_w = chroma_w;
  subbands[0].chroma_h = chroma_h;
  subbands[0].chroma_offset = 0;
  subbands[0].chroma_stride = chroma_stride;
  subbands[0].has_parent = 0;
  subbands[0].scale_factor_shift = 0;
  subbands[0].horizontally_oriented = 0;
  subbands[0].vertically_oriented = 0;

  for(i=0; i<params->transform_depth; i++) {
    /* hl */
    subbands[1+3*i].x = 1;
    subbands[1+3*i].y = 0;
    subbands[1+3*i].w = w;
    subbands[1+3*i].h = h;
    subbands[1+3*i].offset = w;
    subbands[1+3*i].stride = stride;
    subbands[1+3*i].chroma_w = chroma_w;
    subbands[1+3*i].chroma_h = chroma_h;
    subbands[1+3*i].chroma_offset = chroma_w;
    subbands[1+3*i].chroma_stride = chroma_stride;
    subbands[1+3*i].has_parent = (i>0);
    subbands[1+3*i].scale_factor_shift = i;
    subbands[1+3*i].horizontally_oriented = 0;
    subbands[1+3*i].vertically_oriented = 1;

    /* lh */
    subbands[2+3*i].x = 0;
    subbands[2+3*i].y = 1;
    subbands[2+3*i].w = w;
    subbands[2+3*i].h = h;
    subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    subbands[2+3*i].stride = stride;
    subbands[2+3*i].chroma_w = chroma_w;
    subbands[2+3*i].chroma_h = chroma_h;
    subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    subbands[2+3*i].chroma_stride = chroma_stride;
    subbands[2+3*i].has_parent = (i>0);
    subbands[2+3*i].scale_factor_shift = i;
    subbands[2+3*i].horizontally_oriented = 1;
    subbands[2+3*i].vertically_oriented = 0;

    /* hh */
    subbands[3+3*i].x = 1;
    subbands[3+3*i].y = 1;
    subbands[3+3*i].w = w;
    subbands[3+3*i].h = h;
    subbands[3+3*i].offset = w + (stride/2/sizeof(int16_t));
    subbands[3+3*i].stride = stride;
    subbands[3+3*i].chroma_w = chroma_w;
    subbands[3+3*i].chroma_h = chroma_h;
    subbands[3+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    subbands[3+3*i].chroma_stride = chroma_stride;
    subbands[3+3*i].has_parent = (i>0);
    subbands[3+3*i].scale_factor_shift = i;
    subbands[3+3*i].horizontally_oriented = 0;
    subbands[3+3*i].vertically_oriented = 0;

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
schro_encoder_clean_up_transform (SchroEncoderTask *task, int component,
    int index)
{
  static const int wavelet_extent[8] = { 2, 1, 2, 0, 0, 0, 4, 2 };
  SchroSubband *subband = task->subbands + index;
  SchroParams *params = &task->params;
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
    w = ROUND_UP_SHIFT(params->video_format->width, shift);
    height = subband->h;
    h = ROUND_UP_SHIFT(params->video_format->height, shift);
    offset = subband->offset;
  } else {
    stride = subband->chroma_stride >> 1;
    width = subband->chroma_w;
    w = ROUND_UP_SHIFT(params->video_format->width/2, shift);
    height = subband->chroma_h;
    h = ROUND_UP_SHIFT(params->video_format->height/2, shift);
    offset = subband->chroma_offset;
  }

  data = (int16_t *)task->tmp_frame0->components[component].data + offset;

  SCHRO_LOG("subband index=%d %d x %d at offset %d with stride %d; clean area %d %d", index,
      width, height, offset, stride, w, h);

  h = MIN (h + wavelet_extent[params->wavelet_filter_index], height);
  w = MIN (w + wavelet_extent[params->wavelet_filter_index], width);

  if (w < width) {
    for(j=0;j<h;j++){
      for(i=w;i<width;i++){
        data[j*stride + i] = 0;
      }
    }
  }
  for(j=h;j<height;j++){
    for(i=0;i<width;i++){
      data[j*stride + i] = 0;
    }
  }
}

static void
schro_encoder_encode_transform_data (SchroEncoderTask *task, int component)
{
  int i;
  SchroParams *params = &task->params;

  schro_encoder_init_subbands (task);

  schro_encoder_choose_quantisers (task);

  for (i=0;i < 1 + 3*params->transform_depth; i++) {
    schro_encoder_clean_up_transform (task, component, i);
    schro_encoder_encode_subband (task, component, i);
  }
}

static int
dequantize (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

static int
quantize (int value, int quant_factor, int quant_offset,
    unsigned int inv_quant_factor)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)<<2;
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = -x;
  } else {
    x = value<<2;
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = x;
  }
  return value;
}

static int
schro_encoder_quantize_subband (SchroEncoderTask *task, int component, int index,
    int16_t *quant_data)
{
  SchroSubband *subband = task->subbands + index;
  int pred_value;
  int quant_factor;
  unsigned int inv_quant_factor;
  int quant_offset;
  int stride;
  int width;
  int height;
  int offset;
  int i,j;
  int16_t *data;
  int subband_zero_flag;

  subband_zero_flag = 1;

  /* FIXME doesn't handle quantisation of codeblocks */

  quant_factor = schro_table_quant[subband->quant_index];
  inv_quant_factor = schro_table_inverse_quant[subband->quant_index];
  if (task->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[subband->quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[subband->quant_index];
  }

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

  data = (int16_t *)task->tmp_frame0->components[component].data + offset;

  if (index == 0) {
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int q;

        if (j>0) {
          if (i>0) {
            pred_value = schro_divide(data[j*stride + i - 1] +
                data[(j-1)*stride + i] + data[(j-1)*stride + i - 1] + 1,3);
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

        q = quantize(data[j*stride + i] - pred_value, quant_factor,
            quant_offset, inv_quant_factor);
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

        q = quantize(data[j*stride + i], quant_factor, quant_offset,
            inv_quant_factor);
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
schro_encoder_encode_subband (SchroEncoderTask *task, int component, int index)
{
  SchroParams *params = &task->params;
  SchroSubband *subband = task->subbands + index;
  SchroSubband *parent_subband = NULL;
  SchroArith *arith;
  int16_t *data;
  int16_t *parent_data = NULL;
  int i,j;
  int subband_zero_flag;
  int stride;
  int width;
  int height;
  int offset;
  int16_t *quant_data;
  int x,y;
  int horiz_codeblocks;
  int vert_codeblocks;
  int have_zero_flags;
  int coeff_reset;
  int coeff_count = 0;
  int have_quant_offset;

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

  SCHRO_LOG("subband index=%d %d x %d at offset %d with stride %d", index,
      width, height, offset, stride);

  data = (int16_t *)task->tmp_frame0->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)task->tmp_frame0->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)task->tmp_frame0->components[component].data +
        parent_subband->chroma_offset;
    }
  }

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, task->subband_buffer);
  schro_arith_init_contexts (arith);

  quant_data = task->quant_data;
  subband_zero_flag = schro_encoder_quantize_subband (task, component,
      index, quant_data);
  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_bits_sync (task->bits);
    schro_bits_encode_uint (task->bits, 0);
    schro_bits_sync (task->bits);
    schro_arith_free (arith);
    return;
  }

  coeff_reset = CLAMP(((width*height)>>5), 25, 800);
  if (params->spatial_partition_flag) {
    if (index == 0) {
      horiz_codeblocks = params->horiz_codeblocks[0];
      vert_codeblocks = params->vert_codeblocks[0];
    } else {
      horiz_codeblocks = params->horiz_codeblocks[subband->scale_factor_shift+1];
      vert_codeblocks = params->vert_codeblocks[subband->scale_factor_shift+1];
    }
  } else {
    horiz_codeblocks = 1;
    vert_codeblocks = 1;
  }
  if ((horiz_codeblocks > 1 || vert_codeblocks > 1) && index > 0) {
    have_zero_flags = TRUE;
  } else {
    have_zero_flags = FALSE;
  }
  if (horiz_codeblocks > 1 || vert_codeblocks > 1) {
    if (params->codeblock_mode_index == 1) {
      have_quant_offset = TRUE;
    } else {
      have_quant_offset = FALSE;
    }
  } else {
    have_quant_offset = FALSE;
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
          goto out;
        }
      }
    }
out:
    _schro_arith_context_encode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK,
        zero_codeblock);
    if (zero_codeblock) {
      continue;
    }
  }

  if (have_quant_offset) {
    _schro_arith_context_encode_sint (arith,
        SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
        SCHRO_CTX_QUANTISER_SIGN, 0);
  }

  for(j=ymin;j<ymax;j++){
    for(i=xmin;i<xmax;i++){
      int parent;
      int cont_context;
      int value_context;
      int nhood_or;
      int previous_value;
      int sign_context;

      /* FIXME This code is so ugly.  Most of these if statements
       * are constant over the entire codeblock. */

      if (subband->has_parent) {
        parent = parent_data[(j>>1)*(stride<<1) + (i>>1)];
      } else {
        parent = 0;
      }
//parent = 0;

      nhood_or = 0;
      if (j>0) {
        nhood_or |= quant_data[(j-1)*width + i];
      }
      if (i>0) {
        nhood_or |= quant_data[j*width + i - 1];
      }
      if (i>0 && j>0) {
        nhood_or |= quant_data[(j-1)*width + i - 1];
      }
//nhood_or = 0;
      
      previous_value = 0;
      if (subband->horizontally_oriented) {
        if (i > 0) {
          previous_value = quant_data[j*width + i - 1];
        }
      } else if (subband->vertically_oriented) {
        if (j > 0) {
          previous_value = quant_data[(j-1)*width + i];
        }
      }
//previous_value = 0;

      if (previous_value < 0) {
        sign_context = SCHRO_CTX_SIGN_NEG;
      } else {
        if (previous_value > 0) {
          sign_context = SCHRO_CTX_SIGN_POS;
        } else {
          sign_context = SCHRO_CTX_SIGN_ZERO;
        }
      }

      if (parent == 0) {
        if (nhood_or == 0) {
          cont_context = SCHRO_CTX_ZPZN_F1;
        } else {
          cont_context = SCHRO_CTX_ZPNN_F1;
        }
      } else {
        if (nhood_or == 0) {
          cont_context = SCHRO_CTX_NPZN_F1;
        } else {
          cont_context = SCHRO_CTX_NPNN_F1;
        }
      }

      value_context = SCHRO_CTX_COEFF_DATA;

      _schro_arith_context_encode_sint (arith, cont_context, value_context,
          sign_context, quant_data[j*width + i]);

      coeff_count++;
      if (coeff_count == coeff_reset) {
        coeff_count = 0;
        schro_arith_halve_all_counts(arith);
      }
    }
  }
    }
  }

  schro_arith_flush (arith);

  SCHRO_ASSERT(arith->offset < task->subband_size);

#ifdef DIRAC_COMPAT
  schro_bits_sync (task->bits);
#endif
  schro_bits_encode_uint (task->bits, arith->offset);
  if (arith->offset > 0) {
    schro_bits_encode_uint (task->bits, subband->quant_index);

    schro_bits_sync (task->bits);

    schro_bits_append (task->bits, arith->buffer->data, arith->offset);
  }
  schro_arith_free (arith);
}

/* frame queue */

SchroEncoderFrame *
schro_encoder_frame_new (void)
{
  SchroEncoderFrame *encoder_frame;

  encoder_frame = malloc(sizeof(SchroEncoderFrame));
  memset (encoder_frame, 0, sizeof(SchroEncoderFrame));
  encoder_frame->state = SCHRO_ENCODER_FRAME_STATE_NEW;

  return encoder_frame;
}

void
schro_encoder_frame_free (SchroEncoderFrame *encoder_frame)
{

#if 0
  if (encoder_frame->original_frame) {
    schro_frame_free (encoder_frame->original_frame);
  }
#endif
  free (encoder_frame);
}

static void
schro_encoder_frame_queue_push (SchroEncoder *encoder, SchroFrame *frame)
{
  SchroEncoderFrame *encoder_frame;

  if (encoder->frame_queue_length > SCHRO_FRAME_QUEUE_LENGTH-1) {
    SCHRO_ERROR("pushing too hard");
    return;
  }

  encoder_frame = schro_encoder_frame_new();
  encoder_frame->original_frame = frame;

  encoder_frame->frame_number = encoder->next_frame_number++;

  encoder->frame_queue[encoder->frame_queue_length] = encoder_frame;
  encoder->frame_queue_length++;
}

SchroFrame *
schro_encoder_frame_queue_get (SchroEncoder *encoder, int frame_index)
{
  int i;
  for(i=0;i<encoder->frame_queue_length;i++){
    if (encoder->frame_queue[i]->frame_number == frame_index) {
      return encoder->frame_queue[i]->original_frame;
    }
  }
  return NULL;
}

void
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
schro_encoder_reference_analyse (SchroEncoderFrame *ref)
{
  int i;

  for(i=1;i<5;i++){
    schro_frame_downsample (ref->frames[i], ref->frames[i-1], 1);
  }
}

SchroEncoderFrame *
schro_encoder_reference_add (SchroEncoder *encoder)
{
  int i;
  SchroEncoderFrame *ref;

  SCHRO_DEBUG("adding");

  for(i=0;i<encoder->n_reference_frames;i++){
    ref = encoder->reference_frames + i;
    if (!ref->valid) {
      ref->valid = 1;
      return ref;
    }
  }

  return NULL;
}

SchroEncoderFrame *
schro_encoder_reference_get (SchroEncoder *encoder, int frame_number)
{
  int i;
  SchroEncoderFrame *ref;

  for(i=0;i<encoder->n_reference_frames;i++){
    ref = encoder->reference_frames + i;
    if (ref->valid && ref->frame_number == frame_number) {
      return ref;
    }
  }
  return NULL;
}

static void
schro_encoder_reference_retire (SchroEncoder *encoder, int frame_number)
{
  int i;
  SchroEncoderFrame *ref;
  
  SCHRO_DEBUG("retiring %d", frame_number);

  for(i=0;i<encoder->n_reference_frames;i++){
    ref = encoder->reference_frames + i;
    if (ref->valid && ref->frame_number == frame_number) {
      ref->valid = 0;
      return;
    }
  }

  SCHRO_ASSERT(0);
}

/* output queue */

void
schro_encoder_output_push (SchroEncoder *encoder, SchroBuffer *buffer,
    int slot, int presentation_frame)
{
  int i;

  SCHRO_DEBUG("pushing slot %d", slot);
  for(i=0;i<ARRAY_SIZE(encoder->output_queue);i++){
    if (encoder->output_queue[i].buffer == NULL) {
      encoder->output_queue[i].slot = slot;
      encoder->output_queue[i].buffer = buffer;
      encoder->output_queue[i].presentation_frame = presentation_frame;
      encoder->n_output_queue++;
      return;
    }
  }
  SCHRO_ERROR("no slot available in output queue");
  SCHRO_ASSERT(0);
}

static int
schro_encoder_pull_is_ready (SchroEncoder *encoder)
{
  int i;

  if (encoder->inserted_buffer) {
    return TRUE;
  }

  for(i=0;i<encoder->frame_queue_length;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue[i];
    if (frame->slot == encoder->output_slot &&
        frame->state == SCHRO_ENCODER_FRAME_STATE_DONE) {
      return TRUE;
    }
  }

  if (encoder->frame_queue_length == 0 && encoder->end_of_stream) {
    return TRUE;
  }

  return FALSE;
}

static void
schro_encoder_shift_frame_queue (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;

  while (encoder->frame_queue_length > 0) {
    frame = encoder->frame_queue[0];
    if (frame->state != SCHRO_ENCODER_FRAME_STATE_FREE) {
      break;
    }

    memmove (encoder->frame_queue, encoder->frame_queue + 1,
        (encoder->frame_queue_length - 1) * sizeof(void *));
    encoder->frame_queue_length--;
    schro_encoder_frame_free (frame);
  }
}

SchroBuffer *
schro_encoder_pull (SchroEncoder *encoder, int *presentation_frame)
{
  SchroBuffer *buffer;
  int i;

  SCHRO_DEBUG("pulling slot %d", encoder->output_slot);

  if (encoder->inserted_buffer) {
    buffer = encoder->inserted_buffer;
    encoder->inserted_buffer = NULL;
    if (presentation_frame) {
      *presentation_frame = -1;
    }

    schro_encoder_fixup_offsets (encoder, buffer);

    return buffer;
  }
  
  for(i=0;i<encoder->frame_queue_length;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue[i];
    if (frame->slot == encoder->output_slot &&
        frame->state == SCHRO_ENCODER_FRAME_STATE_DONE) {
      if (frame->access_unit_buffer) {
        buffer = frame->access_unit_buffer;
        frame->access_unit_buffer = NULL;
      } else {
        buffer = frame->output_buffer;
        frame->output_buffer = NULL;

        frame->state = SCHRO_ENCODER_FRAME_STATE_FREE;
        encoder->output_slot++;

        schro_encoder_shift_frame_queue (encoder);
      }
      if (presentation_frame) {
        *presentation_frame = frame->presentation_frame;
      }

      schro_encoder_fixup_offsets (encoder, buffer);

      SCHRO_DEBUG("got buffer length=%d", buffer->length);
      return buffer;
    }
  }

  if (encoder->frame_queue_length == 0 && encoder->end_of_stream) {
    buffer = schro_encoder_encode_end_of_stream (encoder);
    schro_encoder_fixup_offsets (encoder, buffer);
    encoder->end_of_stream_pulled = TRUE;

    return buffer;
  }

  SCHRO_DEBUG("got nothing");
  return NULL;
}

static const int pref_range[][2] = {
  { 0, 3 },
  { 2, 20 },
  { 1, 8 },
  { 0, 7 },
  { 0, 7 },
  { 0, 60 },
  { 0, 8 },
  { -4, 4 },
  { 0, 60 },
  { 0, 8 },
  /* last */
  { 0, 0 }
};

int schro_encoder_preference_get_range (SchroEncoder *encoder,
    SchroPrefEnum pref, int *min, int *max)
{
  if (pref < 0 || pref >= SCHRO_PREF_LAST) {
    return 0;
  }

  if (min) *min = pref_range[pref][0];
  if (max) *max = pref_range[pref][1];

  return 1;
}

int schro_encoder_preference_get (SchroEncoder *encoder, SchroPrefEnum pref)
{
  if (pref >= 0 && pref < SCHRO_PREF_LAST) {
    return encoder->prefs[pref];
  }
  return 0;
}

int schro_encoder_preference_set (SchroEncoder *encoder, SchroPrefEnum pref,
    int value)
{
  if (pref < 0 || pref >= SCHRO_PREF_LAST) {
    return 0;
  }

  value = CLAMP(value, pref_range[pref][0], pref_range[pref][1]);

  switch (pref) {
    default:
      break;
  }

  encoder->prefs[pref] = value;

  return value;
}

