
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
static void schro_encoder_encode_picture_prediction (SchroEncoderFrame *frame);
//static void schro_encoder_encode_motion_data (SchroEncoderFrame *frame);
static void schro_encoder_encode_superblock_split (SchroEncoderFrame *frame);
static void schro_encoder_encode_prediction_modes (SchroEncoderFrame *frame);
static void schro_encoder_encode_vector_data (SchroEncoderFrame *frame, int ref, int xy);
static void schro_encoder_encode_dc_data (SchroEncoderFrame *frame, int comp);
static void schro_encoder_encode_transform_parameters (SchroEncoderFrame *frame);
static void schro_encoder_encode_transform_data (SchroEncoderFrame *frame);
static int schro_encoder_pull_is_ready (SchroEncoder *encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder *encoder);
static void schro_encoder_clean_up_transform_subband (SchroEncoderFrame *frame,
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
  encoder->prefs[SCHRO_PREF_QUANT_BASE] = 20;

  schro_params_set_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_SD576);

  schro_encoder_encode_codec_comment (encoder);

  /* FIXME this should be a parameter */
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
  SchroFrameFormat frame_format;
  int frame_width;
  int frame_height;

  task = malloc(sizeof(SchroEncoderTask));
  memset (task, 0, sizeof(*task));

  frame_format = schro_params_get_frame_format (16,
      encoder->video_format.chroma_format);
  
  frame_width = ROUND_UP_POW2(encoder->video_format.width,
      SCHRO_MAX_TRANSFORM_DEPTH + encoder->video_format.chroma_h_shift);
  frame_height = ROUND_UP_POW2(encoder->video_format.height,
      SCHRO_MAX_TRANSFORM_DEPTH + encoder->video_format.chroma_v_shift);

  task->iwt_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);
  
  frame_width = MAX(
      4 * 12 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*12),
      4 * 16 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*16));
  frame_height = MAX(
      4 * 12 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*12),
      4 * 16 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*16));

  task->prediction_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);

  return task;
}

void
schro_encoder_task_free (SchroEncoderTask *task)
{
  if (task->iwt_frame) {
    schro_frame_unref (task->iwt_frame);
  }
  if (task->prediction_frame) {
    schro_frame_unref (task->prediction_frame);
  }
  if (task->motion_field) {
    schro_motion_field_free (task->motion_field);
  }

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
  SchroFrameFormat format;

  //frame->frame_number = encoder->frame_queue_index;
  //encoder->frame_queue_index++;

  encoder_frame = schro_encoder_frame_new();

  format = schro_params_get_frame_format (8, encoder->video_format.chroma_format);
  if (0 /* !filtering */ && format == frame->format) {
    encoder_frame->original_frame = frame;
  } else {
    encoder_frame->original_frame = schro_frame_new_and_alloc (format,
        encoder->video_format.width, encoder->video_format.height);
    schro_frame_convert (encoder_frame->original_frame, frame);
    schro_frame_unref (frame);
  }

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

  schro_bits_flush (bits);

  subbuffer = schro_buffer_new_subbuffer (buffer, 0,
      schro_bits_get_offset (bits));
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

  frame = task->encoder_frame;

  SCHRO_INFO("completing picture %d", frame->frame_number);

  frame->state = SCHRO_ENCODER_FRAME_STATE_DONE;

  frame->encoder->queue_changed = TRUE;

  if (task->ref_frame0) {
    schro_encoder_frame_unref (task->ref_frame0);
  }
  if (task->ref_frame1) {
    schro_encoder_frame_unref (task->ref_frame1);
  }
  if (frame->is_ref) {
    schro_encoder_reference_add (frame->encoder, frame);
  }

  SCHRO_INFO("PICTURE: %d %d %d %d",
      frame->frame_number, frame->is_ref, frame->params.num_refs,
      frame->output_buffer->length);

  if (frame->start_access_unit) {
    frame->access_unit_buffer = schro_encoder_encode_access_unit (frame->encoder);
  }
  if (frame->last_frame) {
    frame->encoder->completed_eos = TRUE;
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
    SchroEncoderFrame *frame;

    frame = schro_async_pull (encoder->async);
    SCHRO_ASSERT(frame != NULL);

    schro_encoder_task_complete (frame->task);
    schro_encoder_task_free (frame->task);
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
    SchroEncoderFrame *frame;

    schro_async_wait_one (encoder->async);

    frame = schro_async_pull (encoder->async);
    SCHRO_ASSERT(frame != NULL);

    schro_encoder_task_complete (frame->task);
    schro_encoder_task_free (frame->task);
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
      case 4:
        ret = schro_encoder_engine_test_intra (encoder);
        break;
      case 5:
        ret = schro_encoder_engine_lossless (encoder);
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
schro_encoder_analyse_picture (SchroEncoderFrame *frame)
{

  //schro_frame_filter_lowpass2 (encoder_frame->original_frame, 5.0);
  //schro_frame_filter_cwm7 (encoder_frame->original_frame);
  //schro_frame_filter_cwmN (encoder_frame->original_frame, 5);

  schro_encoder_frame_analyse (frame->encoder, frame);

  schro_frame_calculate_average_luma (frame->original_frame);
}

void
schro_encoder_predict_picture (SchroEncoderFrame *frame)
{
  frame->tmpbuf = malloc(SCHRO_LIMIT_WIDTH * 2);
  frame->tmpbuf2 = malloc(SCHRO_LIMIT_WIDTH * 2);

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict (frame);

    schro_frame_convert (frame->task->iwt_frame, frame->original_frame);

    {
      SchroMotion *motion;

      motion = malloc(sizeof(*motion));
      memset(motion, 0, sizeof(*motion));

      motion->src1 = frame->task->ref_frame0->reconstructed_frame;
      
      if (frame->params.num_refs == 2) {
        motion->src2 = frame->task->ref_frame1->reconstructed_frame;
      }
      motion->motion_vectors = frame->task->motion_field->motion_vectors;
      motion->params = &frame->params;
      schro_motion_verify (motion);
      schro_frame_copy_with_motion (frame->task->prediction_frame, motion);

      free(motion);
    }

    schro_frame_subtract (frame->task->iwt_frame, frame->task->prediction_frame);

    schro_frame_zero_extend (frame->task->iwt_frame,
        frame->params.video_format->width,
        frame->params.video_format->height);
  } else {
    schro_frame_convert (frame->task->iwt_frame, frame->original_frame);
  }

  schro_frame_iwt_transform (frame->task->iwt_frame, &frame->params,
      frame->tmpbuf);
  schro_encoder_clean_up_transform (frame);
}

void
schro_encoder_encode_picture_all (SchroEncoderFrame *frame)
{
  schro_encoder_analyse_picture (frame);
  schro_encoder_predict_picture (frame);
  schro_encoder_encode_picture (frame);
  schro_encoder_reconstruct_picture (frame);
  schro_encoder_postanalyse_picture (frame);
}

void
schro_encoder_encode_picture (SchroEncoderFrame *frame)
{
  int residue_bits_start;
  SchroBuffer *subbuffer;
  int frame_width, frame_height;

  frame->output_buffer = schro_buffer_new_and_alloc (frame->output_buffer_size);

  frame->subband_size = frame->encoder->video_format.width *
    frame->encoder->video_format.height / 4 * 2;
  frame->subband_buffer = schro_buffer_new_and_alloc (frame->subband_size);

  frame_width = ROUND_UP_POW2(frame->encoder->video_format.width,
      SCHRO_MAX_TRANSFORM_DEPTH + frame->encoder->video_format.chroma_h_shift);
  frame_height = ROUND_UP_POW2(frame->encoder->video_format.height,
      SCHRO_MAX_TRANSFORM_DEPTH + frame->encoder->video_format.chroma_v_shift);

  frame->quant_data = malloc (sizeof(int16_t) * frame_width * frame_height / 4);

  frame->bits = schro_bits_new ();
  schro_bits_encode_init (frame->bits, frame->output_buffer);

  /* encode header */
  schro_encoder_encode_parse_info (frame->bits,
      SCHRO_PARSE_CODE_PICTURE(frame->is_ref, frame->params.num_refs));
  schro_encoder_encode_picture_header (frame);

  if (frame->params.num_refs > 0) {
    schro_bits_sync(frame->bits);
    schro_encoder_encode_picture_prediction (frame);
    schro_bits_sync(frame->bits);
    schro_encoder_encode_superblock_split (frame);
    schro_encoder_encode_prediction_modes (frame);
    schro_encoder_encode_vector_data (frame, 0, 0);
    schro_encoder_encode_vector_data (frame, 0, 1);
    if (frame->params.num_refs > 1) {
      schro_encoder_encode_vector_data (frame, 1, 0);
      schro_encoder_encode_vector_data (frame, 1, 1);
    }
    schro_encoder_encode_dc_data (frame, 0);
    schro_encoder_encode_dc_data (frame, 1);
    schro_encoder_encode_dc_data (frame, 2);
  }

  schro_bits_sync(frame->bits);
  schro_encoder_encode_transform_parameters (frame);

  residue_bits_start = schro_bits_get_offset(frame->bits) * 8;

  schro_bits_sync(frame->bits);
  schro_encoder_encode_transform_data (frame);

  schro_bits_flush (frame->bits);

  subbuffer = schro_buffer_new_subbuffer (frame->output_buffer, 0,
      schro_bits_get_offset (frame->bits));
  schro_buffer_unref (frame->output_buffer);
  frame->output_buffer = subbuffer;

  if (frame->params.num_refs > 0) {
#if 0
    frame->task->metric_to_cost =
      (double)(frame->bits->offset - residue_bits_start) /
      frame->task->stats_metric;
#endif
    SCHRO_INFO("pred bits %d, residue bits %d, dc %d, global = %d, motion %d",
        residue_bits_start, schro_bits_get_offset(frame->bits)*8 - residue_bits_start,
        frame->stats_dc, frame->stats_global, frame->stats_motion);
  }

  if (frame->subband_buffer) {
    schro_buffer_unref (frame->subband_buffer);
  }
  if (frame->quant_data) {
    free (frame->quant_data);
  }
  if (frame->bits) {
    schro_bits_free (frame->bits);
    frame->bits = NULL;
  }
}

void
schro_encoder_reconstruct_picture (SchroEncoderFrame *encoder_frame)
{
  SchroFrameFormat frame_format;
  SchroFrame *frame;

  if (!encoder_frame->is_ref) return;

  schro_frame_inverse_iwt_transform (encoder_frame->task->iwt_frame, &encoder_frame->params,
      encoder_frame->tmpbuf);
  if (encoder_frame->params.num_refs > 0) {
    schro_frame_add (encoder_frame->task->iwt_frame, encoder_frame->task->prediction_frame);
  }

  frame_format = schro_params_get_frame_format (8,
      encoder_frame->encoder->video_format.chroma_format);
  frame = schro_frame_new_and_alloc (frame_format,
      encoder_frame->encoder->video_format.width,
      encoder_frame->encoder->video_format.height);
  schro_frame_convert (frame, encoder_frame->task->iwt_frame);
  encoder_frame->reconstructed_frame =
    schro_upsampled_frame_new (frame);
  schro_upsampled_frame_upsample (encoder_frame->reconstructed_frame);

}

void
schro_encoder_postanalyse_picture (SchroEncoderFrame *frame)
{
#if 0
  double mssim;

  mssim = schro_ssim (frame->original_frame,
      frame->reconstructed_frame->frames[0]);
#if 0
  if (mssim < 0.9) {
    frame->encoder->prefs[SCHRO_PREF_QUANT_BASE]--;
  } else {
    frame->encoder->prefs[SCHRO_PREF_QUANT_BASE]++;
  }
#endif
  SCHRO_ERROR("mssim %g quant_base %d", mssim,
      frame->encoder->prefs[SCHRO_PREF_QUANT_BASE]);
#endif

}

static void
schro_encoder_encode_picture_prediction (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroParams def;

  memset(&def, 0, sizeof(def));
  def.num_refs = params->num_refs;
  schro_params_init (&def, frame->encoder->video_format.index);

  /* block params flag */
  if (def.xblen_luma == params->xblen_luma &&
      def.xbsep_luma == params->xbsep_luma &&
      def.yblen_luma == params->yblen_luma &&
      def.ybsep_luma == params->ybsep_luma) {
    schro_bits_encode_bit (frame->bits, FALSE);
  } else {
    int index;
    schro_bits_encode_bit (frame->bits, TRUE);
    index = schro_params_get_block_params (params);
    schro_bits_encode_uint (frame->bits, index);
    if (index == 0) {
      schro_bits_encode_uint (frame->bits, params->xblen_luma);
      schro_bits_encode_uint (frame->bits, params->yblen_luma);
      schro_bits_encode_uint (frame->bits, params->xbsep_luma);
      schro_bits_encode_uint (frame->bits, params->xbsep_luma);
    }
  }

  /* mv precision flag */
  if (params->mv_precision == def.mv_precision) {
    schro_bits_encode_bit (frame->bits, FALSE);
  } else {
    schro_bits_encode_bit (frame->bits, TRUE);
    schro_bits_encode_uint (frame->bits, params->mv_precision);
  }

  /* global motion flag */
  schro_bits_encode_bit (frame->bits, params->have_global_motion);
  if (params->have_global_motion) {
    int i;
    for(i=0;i<params->num_refs;i++){
      SchroGlobalMotion *gm = params->global_motion + i;

      if (gm->b0 == 0 && gm->b1 == 0) {
        schro_bits_encode_bit (frame->bits, 0);
      } else {
        schro_bits_encode_bit (frame->bits, 1);
        schro_bits_encode_sint (frame->bits, gm->b0);
        schro_bits_encode_sint (frame->bits, gm->b1);
      }

      if (gm->a_exp == 0 && gm->a00 == 1 && gm->a01 == 0 && gm->a10 == 0 &&
          gm->a11 == 1) {
        schro_bits_encode_bit (frame->bits, 0);
      } else {
        schro_bits_encode_bit (frame->bits, 1);
        schro_bits_encode_uint (frame->bits, gm->a_exp);
        schro_bits_encode_sint (frame->bits, gm->a00);
        schro_bits_encode_sint (frame->bits, gm->a01);
        schro_bits_encode_sint (frame->bits, gm->a10);
        schro_bits_encode_sint (frame->bits, gm->a11);
      }

      if (gm->c_exp == 0 && gm->c0 == 0 && gm->c1 == 0) {
        schro_bits_encode_bit (frame->bits, 0);
      } else {
        schro_bits_encode_bit (frame->bits, 1);
        schro_bits_encode_uint (frame->bits, gm->c_exp);
        schro_bits_encode_sint (frame->bits, gm->c0);
        schro_bits_encode_sint (frame->bits, gm->c1);
      }
    }
  }

  /* picture prediction mode flag */
  if (params->picture_pred_mode == 0) {
    schro_bits_encode_bit (frame->bits, FALSE);
  } else {
    schro_bits_encode_bit (frame->bits, TRUE);
    schro_bits_encode_uint (frame->bits, params->picture_pred_mode);
  }

  /* non-default weights flag */
  if (params->picture_weight_bits == def.picture_weight_bits &&
      params->picture_weight_1 == def.picture_weight_1 &&
      (params->picture_weight_2 == def.picture_weight_2 ||
       params->num_refs < 2)) {
    schro_bits_encode_bit (frame->bits, FALSE);
  } else {
    schro_bits_encode_bit (frame->bits, TRUE);
    schro_bits_encode_uint (frame->bits, params->picture_weight_bits);
    schro_bits_encode_sint (frame->bits, params->picture_weight_1);
    if (params->num_refs > 1) {
      schro_bits_encode_sint (frame->bits, params->picture_weight_2);
    }
  }

}

static void
schro_encoder_encode_superblock_split (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
        &frame->task->motion_field->motion_vectors[j*params->x_num_blocks + i];

      SCHRO_ASSERT(mv->split < 3);

      split_prediction = schro_motion_split_prediction (
          frame->task->motion_field->motion_vectors, params, i, j);
      split_residual = (mv->split - split_prediction + 3)%3;
      _schro_arith_context_encode_uint (arith, SCHRO_CTX_SB_F1,
          SCHRO_CTX_SB_DATA, split_residual);
    }
  }

  schro_arith_flush (arith);

  schro_bits_sync (frame->bits);
  schro_bits_encode_uint(frame->bits, arith->offset);

  schro_bits_sync (frame->bits);
  schro_bits_append (frame->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}

static void
schro_encoder_encode_prediction_modes (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->task->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->task->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];
          int pred_mode;

          pred_mode = schro_motion_get_mode_prediction(frame->task->motion_field,
              i+k,j+l) ^ mv->pred_mode;

          _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
              pred_mode & 1);
          if (params->num_refs > 1) {
            _schro_arith_context_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
                pred_mode >> 1);
          }
          if (mv->pred_mode != 0) {
            if (params->have_global_motion) {
              int pred;
              schro_motion_field_get_global_prediction (frame->task->motion_field,
                  i+k, j+l, &pred);
              _schro_arith_context_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
                  mv->using_global ^ pred);
            } else {
              SCHRO_ASSERT(mv->using_global == FALSE);
            }
          }
        }
      }
    }
  }

  schro_arith_flush (arith);

  schro_bits_sync (frame->bits);
  schro_bits_encode_uint(frame->bits, arith->offset);

  schro_bits_sync (frame->bits);
  schro_bits_append (frame->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}

static void
schro_encoder_encode_vector_data (SchroEncoderFrame *frame, int ref, int xy)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith;
  int cont, value, sign;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);
  schro_arith_init_contexts (arith);

  if (xy == 0) {
    if (ref == 0) {
      cont = SCHRO_CTX_MV_REF1_H_CONT_BIN1;
      value = SCHRO_CTX_MV_REF1_H_VALUE;
      sign = SCHRO_CTX_MV_REF1_H_SIGN;
    } else {
      cont = SCHRO_CTX_MV_REF2_H_CONT_BIN1;
      value = SCHRO_CTX_MV_REF2_H_VALUE;
      sign = SCHRO_CTX_MV_REF2_H_SIGN;
    }
  } else {
    if (ref == 0) {
      cont = SCHRO_CTX_MV_REF1_V_CONT_BIN1;
      value = SCHRO_CTX_MV_REF1_V_VALUE;
      sign = SCHRO_CTX_MV_REF1_V_SIGN;
    } else {
      cont = SCHRO_CTX_MV_REF2_V_CONT_BIN1;
      value = SCHRO_CTX_MV_REF2_V_VALUE;
      sign = SCHRO_CTX_MV_REF2_V_SIGN;
    }
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->task->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          int pred_x, pred_y;
          SchroMotionVector *mv =
            &frame->task->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if ((mv->pred_mode>>ref) & 1 && !mv->using_global) {
            schro_motion_vector_prediction (frame->task->motion_field->motion_vectors,
                params, i+k, j+l, &pred_x, &pred_y, 1<<ref);

            if (xy == 0) {
              _schro_arith_context_encode_sint(arith,
                  cont, value, sign,
                  (mv->x1 - pred_x)>>(3-params->mv_precision));
            } else {
              _schro_arith_context_encode_sint(arith,
                  cont, value, sign,
                  (mv->y1 - pred_y)>>(3-params->mv_precision));
            }
          }
        }
      }
    }
  }

  schro_arith_flush (arith);

  schro_bits_sync (frame->bits);
  schro_bits_encode_uint(frame->bits, arith->offset);

  schro_bits_sync (frame->bits);
  schro_bits_append (frame->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}

static void
schro_encoder_encode_dc_data (SchroEncoderFrame *frame, int comp)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->task->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->task->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if (mv->pred_mode == 0) {
            int pred[3];
            SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

            schro_motion_dc_prediction (frame->task->motion_field->motion_vectors,
                params, i+k, j+l, pred);

            switch (comp) {
              case 0:
                _schro_arith_context_encode_sint (arith,
                    SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                    SCHRO_CTX_LUMA_DC_SIGN,
                    mvdc->dc[0] - pred[0]);
                break;
              case 1:
                _schro_arith_context_encode_sint (arith,
                    SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                    SCHRO_CTX_CHROMA1_DC_SIGN,
                    mvdc->dc[1] - pred[1]);
                break;
              case 2:
                _schro_arith_context_encode_sint (arith,
                    SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
                    SCHRO_CTX_CHROMA2_DC_SIGN,
                    mvdc->dc[2] - pred[2]);
                break;
              default:
                SCHRO_ASSERT(0);
            }
          }
        }
      }
    }
  }

  schro_arith_flush (arith);

  schro_bits_sync (frame->bits);
  schro_bits_encode_uint(frame->bits, arith->offset);

  schro_bits_sync (frame->bits);
  schro_bits_append (frame->bits, arith->buffer->data, arith->offset);

  schro_arith_free (arith);
}

#if 0
static void
schro_encoder_encode_motion_data (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith;

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->task->subband_buffer);
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
            SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

            schro_motion_dc_prediction (task->motion_field->motion_vectors,
                params, i+k, j+l, pred);

            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                SCHRO_CTX_LUMA_DC_SIGN,
                mvdc->dc[0] - pred[0]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                SCHRO_CTX_CHROMA1_DC_SIGN,
                mvdc->dc[1] - pred[1]);
            _schro_arith_context_encode_sint (arith,
                SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
                SCHRO_CTX_CHROMA2_DC_SIGN,
                mvdc->dc[2] - pred[2]);
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
              if (mv->pred_mode & 1) {
                schro_motion_vector_prediction (task->motion_field->motion_vectors,
                    params, i+k, j+l, &pred_x, &pred_y, 1);

                _schro_arith_context_encode_sint(arith,
                    SCHRO_CTX_MV_REF1_H_CONT_BIN1,
                    SCHRO_CTX_MV_REF1_H_VALUE,
                    SCHRO_CTX_MV_REF1_H_SIGN,
                    (mv->x1 - pred_x)>>3);
                _schro_arith_context_encode_sint(arith,
                    SCHRO_CTX_MV_REF1_V_CONT_BIN1,
                    SCHRO_CTX_MV_REF1_V_VALUE,
                    SCHRO_CTX_MV_REF1_V_SIGN,
                    (mv->y1 - pred_y)>>3);
              }
              if (mv->pred_mode & 2) {
                schro_motion_vector_prediction (task->motion_field->motion_vectors,
                    params, i+k, j+l, &pred_x, &pred_y, 2);

                _schro_arith_context_encode_sint(arith,
                    SCHRO_CTX_MV_REF2_H_CONT_BIN1,
                    SCHRO_CTX_MV_REF2_H_VALUE,
                    SCHRO_CTX_MV_REF2_H_SIGN,
                    (mv->x2 - pred_x)>>3);
                _schro_arith_context_encode_sint(arith,
                    SCHRO_CTX_MV_REF2_V_CONT_BIN1,
                    SCHRO_CTX_MV_REF2_V_VALUE,
                    SCHRO_CTX_MV_REF2_V_SIGN,
                    (mv->y2 - pred_y)>>3);
              }
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
#endif


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
  schro_bits_encode_uint (bits, encoder->video_format.index);
  schro_params_set_video_format (std_format, encoder->video_format.index);

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
schro_encoder_encode_picture_header (SchroEncoderFrame *frame)
{
  int i;

  schro_bits_sync(frame->bits);
  schro_bits_encode_bits (frame->bits, 32, frame->frame_number);

  for(i=0;i<frame->params.num_refs;i++){
    schro_bits_encode_sint (frame->bits,
        (int32_t)(frame->task->reference_frame_number[i] - frame->frame_number));
  }

  /* retire list */
  schro_bits_encode_uint (frame->bits, frame->n_retire);
  for(i=0;i<frame->n_retire;i++){
    schro_bits_encode_sint (frame->bits,
        (int32_t)(frame->retire[i] - frame->frame_number));
  }
}


static void
schro_encoder_encode_transform_parameters (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroBits *bits = frame->bits;

  if (params->num_refs > 0) {
    /* zero residual */
    schro_bits_encode_bit (bits, FALSE);
  }

  /* transform */
  if (params->wavelet_filter_index == SCHRO_WAVELET_DESL_9_3) {
    schro_bits_encode_bit (bits, 0);
  } else {
    schro_bits_encode_bit (bits, 1);
    schro_bits_encode_uint (bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    schro_bits_encode_bit (bits, 0);
  } else {
    schro_bits_encode_bit (bits, 1);
    schro_bits_encode_uint (bits, params->transform_depth);
  }

  /* spatial partitioning */
  schro_bits_encode_bit (bits, params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    schro_bits_encode_bit (bits, params->nondefault_partition_flag);
    if (params->nondefault_partition_flag) {
      int i;

      for(i=0;i<params->transform_depth+1;i++){
        schro_bits_encode_uint (bits, params->horiz_codeblocks[i]);
        schro_bits_encode_uint (bits, params->vert_codeblocks[i]);
      }
    }
    schro_bits_encode_uint (bits, params->codeblock_mode_index);
  }
}



void
schro_encoder_clean_up_transform (SchroEncoderFrame *frame)
{
  int i;
  int component;
  SchroParams *params = &frame->params;

  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      schro_encoder_clean_up_transform_subband (frame, component, i);
    }
  }
}

static void
schro_encoder_clean_up_transform_subband (SchroEncoderFrame *frame, int component,
    int index)
{
  static const int wavelet_extent[8] = { 2, 1, 2, 0, 0, 0, 4, 2 };
  SchroSubband *subband = frame->subbands + index;
  SchroParams *params = &frame->params;
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
    w = ROUND_UP_SHIFT(params->video_format->width, shift);
    height = subband->chroma_h;
    h = ROUND_UP_SHIFT(params->video_format->height, shift);
    offset = subband->chroma_offset;
  }

  data = (int16_t *)frame->task->iwt_frame->components[component].data + offset;

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
schro_encoder_estimate_subband (SchroEncoderFrame *frame, int component,
    int index)
{
  SchroSubband *subband = frame->subbands + index;
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

  data = (int16_t *)frame->task->iwt_frame->components[component].data + offset;
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

  frame->estimated_entropy = entropy/16;
}

static void
schro_encoder_encode_transform_data (SchroEncoderFrame *frame)
{
  int i;
  int component;
  SchroParams *params = &frame->params;

  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      if (i != 0) schro_bits_sync (frame->bits);
      if (0) schro_encoder_estimate_subband (frame, component, i);
      schro_encoder_encode_subband (frame, component, i);
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
schro_encoder_quantize_subband (SchroEncoderFrame *frame, int component, int index,
    int16_t *quant_data)
{
  SchroSubband *subband = frame->subbands + index;
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
  if (frame->params.num_refs > 0) {
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

  data = (int16_t *)frame->task->iwt_frame->components[component].data + offset;

  if (index == 0) {
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int q;

        if (frame->params.num_refs == 0) {
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
schro_encoder_encode_subband (SchroEncoderFrame *frame, int component, int index)
{
  SchroParams *params = &frame->params;
  SchroSubband *subband = frame->subbands + index;
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

  data = (int16_t *)frame->task->iwt_frame->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)frame->task->iwt_frame->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)frame->task->iwt_frame->components[component].data +
        parent_subband->chroma_offset;
    }
  }

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);
  schro_arith_init_contexts (arith);

  quant_data = frame->quant_data;
  subband_zero_flag = schro_encoder_quantize_subband (frame, component,
      index, quant_data);

  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_bits_encode_uint (frame->bits, 0);
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

  SCHRO_ASSERT(arith->offset < frame->subband_size);

  if (component == 0 && index > 0) {
    SCHRO_INFO("SUBBAND_EST: %d %d %d %d", component, index,
        frame->estimated_entropy, arith->offset);
  }

  schro_bits_encode_uint (frame->bits, arith->offset);
  if (arith->offset > 0) {
    schro_bits_encode_uint (frame->bits, subband->quant_index);

    schro_bits_sync (frame->bits);

    schro_bits_append (frame->bits, arith->buffer->data, arith->offset);
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
      schro_upsampled_frame_free (frame->reconstructed_frame);
    }
    for(i=0;i<5;i++){
      if (frame->downsampled_frames[i]) {
        schro_frame_unref (frame->downsampled_frames[i]);
      }
    }

    if (frame->tmpbuf) free (frame->tmpbuf);
    if (frame->tmpbuf2) free (frame->tmpbuf2);
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
  { 0, 5 },
  { 2, 20 },
  { 1, 8 },
  { 0, 7 },
  { 0, 7 },
  { 0, 60 },
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

