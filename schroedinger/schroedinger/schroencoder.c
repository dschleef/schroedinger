
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
//#include <stdlib.h>
#include <string.h>
//#include <stdio.h>

#if 0
static void schro_encoder_reference_retire (SchroEncoder *encoder,
    SchroPictureNumber frame_number);
static void schro_encoder_reference_retire_all (SchroEncoder *encoder,
    SchroPictureNumber frame_number);
#endif
static void schro_encoder_engine_init (SchroEncoder *encoder);
static void schro_encoder_encode_picture_prediction (SchroEncoderTask *task);
static void schro_encoder_encode_motion_data (SchroEncoderTask *task);
static void schro_encoder_encode_transform_parameters (SchroEncoderTask *task);
static void schro_encoder_encode_transform_data (SchroEncoderTask *task);
static int schro_encoder_pull_is_ready (SchroEncoder *encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder *encoder);
static void schro_encoder_clean_up_transform_subband (SchroEncoderTask *task,
    int component, int index);
static void schro_encoder_fixup_offsets (SchroEncoder *encoder,
    SchroBuffer *buffer);


SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;

  encoder = malloc(sizeof(SchroEncoder));
  memset (encoder, 0, sizeof(SchroEncoder));

  encoder->version_major = 0;
  encoder->version_minor = 11;
  encoder->profile = 0;
  encoder->level = 0;

  encoder->au_frame = -1;
  encoder->au_distance = 24;

  encoder->last_ref = -1;
  encoder->next_ref = -1;
  encoder->mid1_ref = -1;
  encoder->mid2_ref = -1;

  encoder->prefs[SCHRO_PREF_ENGINE] = 3;
  encoder->prefs[SCHRO_PREF_REF_DISTANCE] = 4;
  encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 4;
  encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_DESL_9_3;
  encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_5_3;
  encoder->prefs[SCHRO_PREF_QUANT_BASE] = 0;
  encoder->prefs[SCHRO_PREF_QUANT_OFFSET_NONREF] = 8;
  encoder->prefs[SCHRO_PREF_QUANT_OFFSET_SUBBAND] = 0;
  encoder->prefs[SCHRO_PREF_QUANT_DC] = 0;
  encoder->prefs[SCHRO_PREF_QUANT_DC_OFFSET_NONREF] = 0;

  schro_params_set_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_SD576);

  schro_encoder_encode_codec_comment (encoder);

  /* FIXME */
  encoder->queue_depth = 10;

  encoder->frame_queue = schro_queue_new (encoder->queue_depth,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);
  encoder->reference_queue = schro_queue_new (SCHRO_MAX_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  return encoder;
}

void
schro_encoder_free (SchroEncoder *encoder)
{
  if (encoder->async) {
    schro_async_free(encoder->async);
  }

  schro_queue_free (encoder->reference_queue);
  schro_queue_free (encoder->frame_queue);

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
    encoder->video_format.height / 4 * 2;
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
  if (task->prediction_frame == NULL) {
    task->prediction_frame = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_S16,
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
    schro_frame_unref (task->tmp_frame0);
  }
  if (task->prediction_frame) {
    schro_frame_unref (task->prediction_frame);
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
  if (task->bits) {
    schro_bits_free (task->bits);
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
  if (!encoder->end_of_stream && !schro_queue_is_full (encoder->frame_queue)) {
    return TRUE;
  }
  return FALSE;
}

void
schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame)
{
  SchroEncoderFrame *encoder_frame;

  //frame->frame_number = encoder->frame_queue_index;
  //encoder->frame_queue_index++;

  encoder_frame = schro_encoder_frame_new();
  encoder_frame->original_frame = frame;

  encoder_frame->frame_number = encoder->next_frame_number++;

  schro_queue_add (encoder->frame_queue, encoder_frame,
      encoder_frame->frame_number);

  encoder->queue_changed = TRUE;
}

static int
schro_encoder_pull_is_ready (SchroEncoder *encoder)
{
  int i;

  if (encoder->inserted_buffer) {
    return TRUE;
  }

  for(i=0;i<encoder->frame_queue->n;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        frame->state == SCHRO_ENCODER_FRAME_STATE_DONE) {
      return TRUE;
    }
  }

  if (schro_queue_is_empty(encoder->frame_queue) && encoder->end_of_stream) {
    return TRUE;
  }

  return FALSE;
}

static void
schro_encoder_shift_frame_queue (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;

  while (!schro_queue_is_empty(encoder->frame_queue)) {
    frame = encoder->frame_queue->elements[0].data;
    if (frame->state != SCHRO_ENCODER_FRAME_STATE_FREE) {
      break;
    }

    schro_queue_pop (encoder->frame_queue);
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
  
  for(i=0;i<encoder->frame_queue->n;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        frame->state == SCHRO_ENCODER_FRAME_STATE_DONE) {
      if (presentation_frame) {
        *presentation_frame = frame->presentation_frame;
      }
      if (frame->access_unit_buffer) {
        buffer = frame->access_unit_buffer;
        frame->access_unit_buffer = NULL;
      } else {
        buffer = frame->output_buffer;
        frame->output_buffer = NULL;

        frame->state = SCHRO_ENCODER_FRAME_STATE_FREE;
        encoder->output_slot++;

#if 0
        if (frame->n_retire > 0) {
          schro_encoder_reference_retire (encoder, frame->retire);
        }
#endif

        schro_encoder_shift_frame_queue (encoder);
      }

      schro_encoder_fixup_offsets (encoder, buffer);

      SCHRO_DEBUG("got buffer length=%d", buffer->length);
      return buffer;
    }
  }

  if (schro_queue_is_empty(encoder->frame_queue) && encoder->end_of_stream) {
    buffer = schro_encoder_encode_end_of_stream (encoder);
    schro_encoder_fixup_offsets (encoder, buffer);
    encoder->end_of_stream_pulled = TRUE;

    return buffer;
  }

  SCHRO_DEBUG("got nothing");
  return NULL;
}

void
schro_encoder_end_of_stream (SchroEncoder *encoder)
{
  encoder->end_of_stream = TRUE;
  if (encoder->frame_queue->n > 0) {
    SchroEncoderFrame *encoder_frame;
    
    encoder_frame = encoder->frame_queue->elements[encoder->frame_queue->n-1].data;
    encoder_frame->last_frame = TRUE;
  }
}

static int
schro_gain_to_index (int value)
{
  value = (value + 8)>>4;
  return CLAMP(value, 0, 63);
}

void
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
      schro_gain_to_index (base + (percep + gain)*band + gain_hv);
    subbands[2+3*i].quant_index =
      schro_gain_to_index (base + (percep + gain)*band + gain_hv);
    subbands[3+3*i].quant_index =
      schro_gain_to_index (base + (percep + gain)*band + gain_diag);
  }
  {
    int sec = task->frame_number/12;
    for(i=4; i<=11; i++) {
      subbands[i].quant_index = 12;
    }
    if ((sec & 1) == 0) {
      //subbands[4].quant_index -= 4;
      //subbands[5].quant_index -= 4;
      //subbands[6].quant_index -= 4;
      for(i=1; i<4; i++) {
        //subbands[i].quant_index = 20;
      }
    } else {
      for(i=4; i<7; i++) {
        subbands[i].quant_index = 20;
      }
    }
  }

  /* hard coded.  muhuhuhahaha */
  if (task->is_ref) {
    subbands[0].quant_index = 12;
    subbands[1].quant_index = 16;
    subbands[2].quant_index = 16;
    subbands[3].quant_index = 20;
    subbands[4].quant_index = 16;
    subbands[5].quant_index = 16;
    subbands[6].quant_index = 20;
    subbands[7].quant_index = 17;
    subbands[8].quant_index = 17;
    subbands[9].quant_index = 21;
    subbands[10].quant_index = 22;
    subbands[11].quant_index = 22;
    subbands[12].quant_index = 26;
  } else {
    subbands[0].quant_index = 16;
    subbands[1].quant_index = 20;
    subbands[2].quant_index = 20;
    subbands[3].quant_index = 24;
    subbands[4].quant_index = 20;
    subbands[5].quant_index = 20;
    subbands[6].quant_index = 24;
    subbands[7].quant_index = 21;
    subbands[8].quant_index = 21;
    subbands[9].quant_index = 25;
    subbands[10].quant_index = 26;
    subbands[11].quant_index = 26;
    subbands[12].quant_index = 30;
  }
}

static void
schro_encoder_fixup_offsets (SchroEncoder *encoder, SchroBuffer *buffer)
{
  uint8_t *data = buffer->data;

  if (buffer->length < 13) {
    SCHRO_ERROR("packet too short (%d < 13)", buffer->length);
  }

  data[5] = (buffer->length >> 24) & 0xff;
  data[6] = (buffer->length >> 16) & 0xff;
  data[7] = (buffer->length >> 8) & 0xff;
  data[8] = (buffer->length >> 0) & 0xff;
  data[9] = (encoder->prev_offset >> 24) & 0xff;
  data[10] = (encoder->prev_offset >> 16) & 0xff;
  data[11] = (encoder->prev_offset >> 8) & 0xff;
  data[12] = (encoder->prev_offset >> 0) & 0xff;

  encoder->prev_offset = buffer->length;
}

static void
schro_encoder_encode_codec_comment (SchroEncoder *encoder)
{
  char *s = "\001Schrodinger " VERSION;
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

  schro_bits_free (bits);

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

  buffer = schro_buffer_new_and_alloc (SCHRO_PARSE_HEADER_SIZE);

  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);

  schro_encoder_encode_parse_info (bits, SCHRO_PARSE_CODE_END_SEQUENCE);

  schro_bits_free (bits);

  return buffer;
}

static void
schro_encoder_task_complete (SchroEncoderTask *task)
{
  SchroEncoderFrame *frame;

  SCHRO_INFO("completing picture %d", task->frame_number);

  frame = task->encoder_frame;

  frame->state = SCHRO_ENCODER_FRAME_STATE_DONE;

  task->encoder->queue_changed = TRUE;

  frame->output_buffer = task->outbuffer;
  frame->presentation_frame = task->presentation_frame;
  if (task->ref_frame0) {
    schro_encoder_frame_unref (task->ref_frame0);
  }
  if (task->ref_frame1) {
    schro_encoder_frame_unref (task->ref_frame1);
  }
  if (task->is_ref) {
    schro_encoder_reference_add (task->encoder, task->encoder_frame);
  }

  SCHRO_INFO("PICTURE: %d %d %d %d",
      task->frame_number, task->is_ref, task->params.num_refs, task->bits->offset);

  if (frame->start_access_unit) {
    frame->access_unit_buffer = schro_encoder_encode_access_unit (task->encoder);
  }
  if (frame->last_frame) {
    /* FIXME push an EOS */
    task->encoder->completed_eos = TRUE;
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

  SCHRO_INFO("iterate %d %d %d",
      encoder->queue_changed, schro_encoder_push_ready(encoder),
      encoder->completed_eos);

#if 0
  {
    int i;
    for(i=0;i<encoder->frame_queue->n;i++){
      SchroEncoderFrame *frame = encoder->frame_queue->elements[i].data;
      SCHRO_ERROR("%p %d %d", frame, frame->frame_number, frame->state);
    }
  }
#endif

  while (!schro_encoder_pull_is_ready(encoder) &&
      !encoder->queue_changed && !schro_encoder_push_ready(encoder) &&
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
      case 3:
        ret = schro_encoder_engine_tworef (encoder);
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

static void
schro_encoder_engine_init (SchroEncoder *encoder)
{
  encoder->engine_init = 1;

  encoder->engine = encoder->prefs[SCHRO_PREF_ENGINE];
  encoder->ref_distance = encoder->prefs[SCHRO_PREF_REF_DISTANCE];

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

  schro_encoder_frame_analyse (task->encoder, task->encoder_frame);

  if (task->params.num_refs > 0) {
    schro_encoder_motion_predict (task);

    schro_encoder_encode_picture_prediction (task);
    schro_encoder_encode_motion_data (task);

    schro_frame_convert (task->tmp_frame0, task->encode_frame);

    {
      SchroMotion *motion;

      motion = malloc(sizeof(*motion));
      memset(motion, 0, sizeof(*motion));

      motion->src1[0] = task->ref_frame0->reconstructed_frame;

      SCHRO_ASSERT(motion->src1[0] != NULL);
      if (task->params.num_refs == 2) {
        motion->src2[0] = task->ref_frame1->reconstructed_frame;
        SCHRO_ASSERT(motion->src2[0] != NULL);
      } else {
        motion->src2[0] = NULL;
      }
      motion->motion_vectors = task->motion_field->motion_vectors;
      motion->params = &task->params;
      schro_motion_verify (motion);
      schro_frame_copy_with_motion (task->prediction_frame, motion);

      free(motion);
    }

    SCHRO_DEBUG("luma %d %d ref %d",
        schro_frame_calculate_average_luma (task->encoder_frame->original_frame),
        schro_frame_calculate_average_luma (task->prediction_frame),
        schro_frame_calculate_average_luma (task->ref_frame0->reconstructed_frame)
        );

    schro_frame_subtract (task->tmp_frame0, task->prediction_frame);

    schro_frame_zero_extend (task->tmp_frame0,
        task->params.video_format->width,
        task->params.video_format->height);
  } else {
    schro_frame_convert (task->tmp_frame0, task->encode_frame);
  }

  schro_encoder_encode_transform_parameters (task);

  schro_frame_iwt_transform (task->tmp_frame0, &task->params,
      task->tmpbuf);
  schro_encoder_clean_up_transform (task);

  residue_bits_start = task->bits->offset;

  schro_encoder_encode_transform_data (task);

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
    if (task->params.num_refs > 0) {
      schro_frame_add (task->tmp_frame0, task->prediction_frame);
    }

    task->encoder_frame->reconstructed_frame = 
      schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8,
          task->encoder->video_format.width,
          task->encoder->video_format.height,
          task->encoder->video_format.chroma_width,
          task->encoder->video_format.chroma_height);

    schro_frame_convert (task->encoder_frame->reconstructed_frame,
        task->tmp_frame0);

    SCHRO_DEBUG("luma ref %d",
        schro_frame_calculate_average_luma (task->encoder_frame->reconstructed_frame)
        );
  }

  task->completed = TRUE;
}

static void
schro_encoder_encode_picture_prediction (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;

  schro_bits_sync(task->bits);

  /* block params flag */
  /* FIXME */
  if (TRUE) {
    schro_bits_encode_bit (task->bits, FALSE);
  } else {
    int index = 0;
    schro_bits_encode_uint (task->bits, 0);
    if (index == 0) {
      schro_bits_encode_uint (task->bits, params->xblen_luma);
      schro_bits_encode_uint (task->bits, params->yblen_luma);
      schro_bits_encode_uint (task->bits, params->xbsep_luma);
      schro_bits_encode_uint (task->bits, params->xbsep_luma);
    }
  }

  /* mv precision flag */
  /* FIXME */
  if (params->mv_precision == 0) {
    schro_bits_encode_bit (task->bits, FALSE);
  } else {
    schro_bits_encode_bit (task->bits, TRUE);
    schro_bits_encode_uint (task->bits, params->mv_precision);
  }

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
  if (params->picture_pred_mode == 0) {
    schro_bits_encode_bit (task->bits, FALSE);
  } else {
    schro_bits_encode_bit (task->bits, TRUE);
    schro_bits_encode_uint (task->bits, params->picture_pred_mode);
  }

  /* non-default weights flag */
  /* FIXME */
  if (TRUE) {
    schro_bits_encode_bit (task->bits, FALSE);
  } else {
    schro_bits_encode_bit (task->bits, TRUE);
    schro_bits_encode_uint (task->bits, params->picture_weight_bits);
    schro_bits_encode_sint (task->bits, params->picture_weight_1);
    if (params->num_refs > 1) {
      schro_bits_encode_sint (task->bits, params->picture_weight_2);
    }
  }

}


static void
schro_encoder_encode_motion_data (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i,j;
  SchroArith *arith;

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

              /* FIXME assumption that mv precision is 0 */
              _schro_arith_context_encode_sint(arith,
                  SCHRO_CTX_MV_REF1_H_CONT_BIN1,
                  SCHRO_CTX_MV_REF1_H_VALUE,
                  SCHRO_CTX_MV_REF1_H_SIGN,
                  (mv->u.xy.x - pred_x)>>3);
              _schro_arith_context_encode_sint(arith,
                  SCHRO_CTX_MV_REF1_V_CONT_BIN1,
                  SCHRO_CTX_MV_REF1_V_VALUE,
                  SCHRO_CTX_MV_REF1_V_SIGN,
                  (mv->u.xy.y - pred_y)>>3);
            }
          }
        }
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
  schro_bits_encode_bit (bits, format->interlaced);
  if(format->interlaced) {
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
schro_encoder_encode_parse_info (SchroBits *bits, int parse_code)
{
  /* parse parameters */
  schro_bits_encode_bits (bits, 8, 'B');
  schro_bits_encode_bits (bits, 8, 'B');
  schro_bits_encode_bits (bits, 8, 'C');
  schro_bits_encode_bits (bits, 8, 'D');
  schro_bits_encode_bits (bits, 8, parse_code);

  /* offsets */
  schro_bits_encode_bits (bits, 32, 0);
  schro_bits_encode_bits (bits, 32, 0);
}

void
schro_encoder_encode_picture_header (SchroEncoderTask *task)
{
  int i;

  schro_bits_sync(task->bits);
  schro_bits_encode_bits (task->bits, 32, task->frame_number);

  for(i=0;i<task->params.num_refs;i++){
    schro_bits_encode_sint (task->bits,
        (int32_t)(task->reference_frame_number[i] - task->frame_number));
  }

  /* retire list */
  schro_bits_encode_uint (task->bits, task->n_retire);
  for(i=0;i<task->n_retire;i++){
    schro_bits_encode_sint (task->bits,
        (int32_t)(task->retire[i] - task->frame_number));
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
schro_encoder_clean_up_transform (SchroEncoderTask *task)
{
  int i;
  int component;
  SchroParams *params = &task->params;

  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      schro_encoder_clean_up_transform_subband (task, component, i);
    }
  }
}

static void
schro_encoder_clean_up_transform_subband (SchroEncoderTask *task, int component,
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
    w = ROUND_UP_SHIFT(params->video_format->width,
        shift + params->video_format->chroma_h_shift);
    height = subband->chroma_h;
    h = ROUND_UP_SHIFT(params->video_format->height,
        shift + params->video_format->chroma_v_shift);
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

static int
ilog2 (unsigned int x)
{
  int i;
  for(i=0;i<60;i++){
    if (x*4 < schro_table_quant[i]) return i;
  }
#if 0
  for(i=0;x>1;i++){
    x >>= 1;
  }
#endif
  return i;
}

static void
schro_encoder_estimate_subband (SchroEncoderTask *task, int component,
    int index)
{
  SchroSubband *subband = task->subbands + index;
  int i;
  int j;
  int16_t *data;
  int stride;
  int width;
  int height;
  int offset;
  int x;
  int hist[64];
  int entropy;

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

  for(i=0;i<64;i++) hist[i] = 0;

  data = (int16_t *)task->tmp_frame0->components[component].data + offset;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      x = ilog2(abs(data[i]));
      hist[x]++;
    }
    data += stride;
  }

  entropy = 0;
  for(i=subband->quant_index;i<64;i++){
    x = 4 + i - subband->quant_index;
    entropy += x * hist[i];
  }

  task->estimated_entropy = entropy/16;
}

static void
schro_encoder_encode_transform_data (SchroEncoderTask *task)
{
  int i;
  int component;
  SchroParams *params = &task->params;

  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      if (0) schro_encoder_estimate_subband (task, component, i);
      schro_encoder_encode_subband (task, component, i);
    }
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

        if (task->params.num_refs == 0) {
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
        } else {
          pred_value = 0;
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
    }
  }
    }
  }

  schro_arith_flush (arith);

  SCHRO_ASSERT(arith->offset < task->subband_size);

  if (component == 0 && index > 0) {
    SCHRO_INFO("SUBBAND_EST: %d %d %d %d", component, index,
        task->estimated_entropy, arith->offset);
  }

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
  encoder_frame->refcount = 1;

  return encoder_frame;
}

void
schro_encoder_frame_ref (SchroEncoderFrame *frame)
{
  frame->refcount++;
}

void
schro_encoder_frame_unref (SchroEncoderFrame *frame)
{
  int i;

  frame->refcount--;
  if (frame->refcount == 0) {
    if (frame->original_frame) {
      schro_frame_unref (frame->original_frame);
    }
    if (frame->reconstructed_frame) {
      schro_frame_unref (frame->reconstructed_frame);
    }
    for(i=0;i<5;i++){
      if (frame->downsampled_frames[i]) {
        schro_frame_unref (frame->downsampled_frames[i]);
      }
    }
    free (frame);
  }
}

/* reference pool */

void
schro_encoder_reference_add (SchroEncoder *encoder, SchroEncoderFrame *frame)
{
  if (schro_queue_is_full(encoder->reference_queue)) {
    schro_queue_pop (encoder->reference_queue);
  }

  SCHRO_DEBUG("adding reference %p %d", frame, frame->frame_number);

  schro_encoder_frame_ref (frame);
  schro_queue_add (encoder->reference_queue, frame, frame->frame_number);
}

SchroEncoderFrame *
schro_encoder_reference_get (SchroEncoder *encoder,
    SchroPictureNumber frame_number)
{
  return schro_queue_find (encoder->reference_queue, frame_number);
}

#if 0
void
schro_encoder_reference_retire_all (SchroEncoder *encoder,
    SchroPictureNumber frame_number)
{
  schro_queue_clear (encoder->reference_queue);
}
#endif

#if 0
void
schro_encoder_reference_retire (SchroEncoder *encoder,
    SchroPictureNumber frame_number)
{
  schro_queue_delete (encoder->reference_queue, frame_number);
}
#endif

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

