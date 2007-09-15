
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

static void schro_encoder_engine_init (SchroEncoder *encoder);
static void schro_encoder_encode_picture_prediction (SchroEncoderFrame *frame);
static void schro_encoder_encode_superblock_split (SchroEncoderFrame *frame);
static void schro_encoder_encode_prediction_modes (SchroEncoderFrame *frame);
static void schro_encoder_encode_vector_data (SchroEncoderFrame *frame, int ref, int xy);
static void schro_encoder_encode_dc_data (SchroEncoderFrame *frame, int comp);
static void schro_encoder_encode_transform_parameters (SchroEncoderFrame *frame);
static void schro_encoder_encode_transform_data (SchroEncoderFrame *frame);
static void schro_encoder_encode_lowdelay_transform_data (SchroEncoderFrame *frame);
static int schro_encoder_pull_is_ready_locked (SchroEncoder *encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder *encoder);
static void schro_encoder_clean_up_transform_subband (SchroEncoderFrame *frame,
    int component, int index);
static void schro_encoder_fixup_offsets (SchroEncoder *encoder,
    SchroBuffer *buffer);
static int schro_encoder_iterate (SchroEncoder *encoder);
void schro_encoder_encode_slice (SchroEncoderFrame *frame, int x, int y,
    int slice_bytes, int qindex);
int schro_encoder_estimate_slice (SchroEncoderFrame *frame, int x, int y,
    int slice_bytes, int qindex);


SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;

  encoder = malloc(sizeof(SchroEncoder));
  memset (encoder, 0, sizeof(SchroEncoder));

  encoder->version_major = 0;
  encoder->version_minor = 108;
  encoder->profile = 0;
  encoder->level = 0;

  encoder->au_frame = -1;
  encoder->au_distance = 24;

  encoder->last_ref = -1;
  encoder->next_ref = -1;
  encoder->mid1_ref = -1;
  encoder->mid2_ref = -1;

  encoder->prefs[SCHRO_PREF_ENGINE] = 0;
  encoder->prefs[SCHRO_PREF_REF_DISTANCE] = 4;
  encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 4;
  encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_DESL_9_3;
  encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_5_3;
  encoder->prefs[SCHRO_PREF_LAMBDA] = 50;
  encoder->prefs[SCHRO_PREF_PSNR] = 30;

  schro_params_set_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_SD576);

  schro_encoder_encode_codec_comment (encoder);

  /* FIXME this should be a parameter */
  encoder->queue_depth = 20;

  encoder->frame_queue = schro_queue_new (encoder->queue_depth,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  encoder->reference_queue = schro_queue_new (SCHRO_MAX_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  schro_encoder_set_default_subband_weights (encoder);

  return encoder;
}

void
schro_encoder_start (SchroEncoder *encoder)
{
  encoder->async = schro_async_new (0, (int (*)(void *))schro_encoder_iterate,
      encoder);
  schro_encoder_engine_init (encoder);
}

void
schro_encoder_stop (SchroEncoder *encoder)
{

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

void
schro_encoder_use_perceptual_weighting (SchroEncoder *encoder,
    SchroEncoderPerceptualEnum type, double viewing_distance)
{

  encoder->pixels_per_degree_horiz =
    encoder->video_format.width/(2.0*atan(0.5/3.0)*180/M_PI);
  encoder->pixels_per_degree_vert =
    encoder->video_format.aspect_ratio_numerator * 
    (encoder->pixels_per_degree_horiz / encoder->video_format.aspect_ratio_denominator);

  SCHRO_ERROR("pixels per degree horiz=%g vert=%g",
      encoder->pixels_per_degree_horiz, encoder->pixels_per_degree_vert);

  switch(type) {
    default:
    case SCHRO_ENCODER_PERCEPTUAL_CONSTANT:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_constant);
      break;
    case SCHRO_ENCODER_PERCEPTUAL_CCIR959:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_ccir959);
      break;
    case SCHRO_ENCODER_PERCEPTUAL_MOO:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_moo);
      break;
  }
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
  int ret;

  if (encoder->end_of_stream){
    return FALSE;
  }
  
  schro_async_lock (encoder->async);
  ret = schro_queue_is_full (encoder->frame_queue);
  schro_async_unlock (encoder->async);

  return (ret == FALSE);
}

void
schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame)
{
  SchroEncoderFrame *encoder_frame;
  SchroFrameFormat format;

  encoder_frame = schro_encoder_frame_new(encoder);
  encoder_frame->encoder = encoder;

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

  schro_async_lock (encoder->async);
  if (schro_queue_is_full (encoder->frame_queue)) {
    SCHRO_ERROR("push when queue full");
    SCHRO_ASSERT(0);
  }
  schro_queue_add (encoder->frame_queue, encoder_frame,
      encoder_frame->frame_number);
  schro_async_signal_scheduler (encoder->async);
  schro_async_unlock (encoder->async);
}

static int
schro_encoder_pull_is_ready_locked (SchroEncoder *encoder)
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
  
  schro_async_lock (encoder->async);
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

        schro_encoder_shift_frame_queue (encoder);
      }

      schro_encoder_fixup_offsets (encoder, buffer);

      SCHRO_DEBUG("got buffer length=%d", buffer->length);
      schro_async_unlock (encoder->async);
      return buffer;
    }
  }

  if (schro_queue_is_empty(encoder->frame_queue) && encoder->end_of_stream) {
    buffer = schro_encoder_encode_end_of_stream (encoder);
    schro_encoder_fixup_offsets (encoder, buffer);
    encoder->end_of_stream_pulled = TRUE;

    schro_async_unlock (encoder->async);
    return buffer;
  }
  schro_async_unlock (encoder->async);

  SCHRO_DEBUG("got nothing");
  return NULL;
}

void
schro_encoder_end_of_stream (SchroEncoder *encoder)
{
  encoder->end_of_stream = TRUE;
  schro_async_lock (encoder->async);
  if (encoder->frame_queue->n > 0) {
    SchroEncoderFrame *encoder_frame;
    
    encoder_frame = encoder->frame_queue->elements[encoder->frame_queue->n-1].data;
    encoder_frame->last_frame = TRUE;
  }
  schro_async_unlock (encoder->async);
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

SchroStateEnum
schro_encoder_wait (SchroEncoder *encoder)
{
  SchroStateEnum ret = SCHRO_STATE_AGAIN;

  schro_async_lock (encoder->async);
  while (1) {
    if (!encoder->end_of_stream && !schro_queue_is_full (encoder->frame_queue)) {
      SCHRO_DEBUG("need frame");
      ret = SCHRO_STATE_NEED_FRAME;
      break;
    }
    if (schro_encoder_pull_is_ready_locked (encoder)) {
      SCHRO_DEBUG("have buffer");
      ret = SCHRO_STATE_HAVE_BUFFER;
      break;
    }
    if (schro_queue_is_empty(encoder->frame_queue) && encoder->end_of_stream) {
      ret = SCHRO_STATE_END_OF_STREAM;
      break;
    }
 
    SCHRO_DEBUG("encoder waiting");
    schro_async_wait_locked (encoder->async);
  }
  schro_async_unlock (encoder->async);

  return ret;
}

static void
schro_encoder_frame_complete (SchroEncoderFrame *frame)
{
  SCHRO_INFO("completing task, picture %d in state %d",
      frame->frame_number, frame->state);

  SCHRO_ASSERT(frame->busy == TRUE);

  frame->busy = FALSE;

  if (frame->state == SCHRO_ENCODER_FRAME_STATE_POSTANALYSE) {
    frame->state = SCHRO_ENCODER_FRAME_STATE_DONE;

    if (frame->ref_frame0) {
      schro_encoder_frame_unref (frame->ref_frame0);
    }
    if (frame->ref_frame1) {
      schro_encoder_frame_unref (frame->ref_frame1);
    }
    if (frame->is_ref) {
      schro_encoder_reference_add (frame->encoder, frame);
    }

    SCHRO_INFO("PICTURE: %d %d %d",
        frame->frame_number, frame->is_ref, frame->params.num_refs);

    if (frame->start_access_unit) {
      frame->access_unit_buffer = schro_encoder_encode_access_unit (frame->encoder);
    }
    if (frame->last_frame) {
      frame->encoder->completed_eos = TRUE;
    }
  }
}

static int
schro_encoder_iterate (SchroEncoder *encoder)
{
  int ret = FALSE;

  SCHRO_DEBUG("iterate");

  while (schro_async_get_num_completed (encoder->async) > 0) {
    SchroEncoderFrame *frame;

    frame = schro_async_pull_locked (encoder->async);
    SCHRO_ASSERT(frame != NULL);

    schro_encoder_frame_complete (frame);
  }

  SCHRO_INFO("iterate %d", encoder->completed_eos);

#if 0
  /* For debugging purposes */
  {
    int i;
    for(i=0;i<encoder->frame_queue->n;i++){
      SchroEncoderFrame *frame = encoder->frame_queue->elements[i].data;
      SCHRO_ERROR("%p %d %d", frame, frame->frame_number, frame->state);
    }
  }
#endif

  if (1) {
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
      case 6:
        ret = schro_encoder_engine_backtest (encoder);
        break;
      case 7:
        ret = schro_encoder_engine_lowdelay (encoder);
        break;
      default:
        ret = FALSE;
        break;
    }
  }

  return ret;
}

static void
schro_encoder_engine_init (SchroEncoder *encoder)
{
  encoder->engine_init = 1;

  encoder->engine = encoder->prefs[SCHRO_PREF_ENGINE];
  encoder->ref_distance = encoder->prefs[SCHRO_PREF_REF_DISTANCE];

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
  SCHRO_INFO("predict picture %d", frame->frame_number);

  frame->tmpbuf = malloc(SCHRO_LIMIT_WIDTH * 2);
  frame->tmpbuf2 = malloc(SCHRO_LIMIT_WIDTH * 2);

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict (frame);

    schro_frame_convert (frame->iwt_frame, frame->original_frame);

    {
      SchroMotion *motion;

      motion = malloc(sizeof(*motion));
      memset(motion, 0, sizeof(*motion));

      motion->src1 = frame->ref_frame0->reconstructed_frame;
      
      if (frame->params.num_refs == 2) {
        motion->src2 = frame->ref_frame1->reconstructed_frame;
      }
      motion->motion_vectors = frame->motion_field->motion_vectors;
      motion->params = &frame->params;
      schro_motion_verify (motion);
      schro_frame_copy_with_motion (frame->prediction_frame, motion);

      free(motion);
    }

    schro_frame_subtract (frame->iwt_frame, frame->prediction_frame);

    schro_frame_zero_extend (frame->iwt_frame,
        frame->params.video_format->width,
        frame->params.video_format->height);
  } else {
    schro_frame_convert (frame->iwt_frame, frame->original_frame);
  }

  schro_frame_iwt_transform (frame->iwt_frame, &frame->params,
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

  SCHRO_INFO("encode picture %d", frame->frame_number);

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
      SCHRO_PARSE_CODE_PICTURE(frame->is_ref, frame->params.num_refs,
        frame->params.is_lowdelay));
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
  if (frame->params.is_lowdelay) {
    schro_encoder_encode_lowdelay_transform_data (frame);
  } else {
    schro_encoder_encode_transform_data (frame);
  }

  schro_bits_flush (frame->bits);

  subbuffer = schro_buffer_new_subbuffer (frame->output_buffer, 0,
      schro_bits_get_offset (frame->bits));
  schro_buffer_unref (frame->output_buffer);
  frame->output_buffer = subbuffer;

  if (frame->params.num_refs > 0) {
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

  schro_frame_inverse_iwt_transform (encoder_frame->iwt_frame, &encoder_frame->params,
      encoder_frame->tmpbuf);
  if (encoder_frame->params.num_refs > 0) {
    schro_frame_add (encoder_frame->iwt_frame, encoder_frame->prediction_frame);
  }

  frame_format = schro_params_get_frame_format (8,
      encoder_frame->encoder->video_format.chroma_format);
  frame = schro_frame_new_and_alloc (frame_format,
      encoder_frame->encoder->video_format.width,
      encoder_frame->encoder->video_format.height);
  schro_frame_convert (frame, encoder_frame->iwt_frame);
  encoder_frame->reconstructed_frame =
    schro_upsampled_frame_new (frame);

  if (encoder_frame->is_ref) {
    schro_upsampled_frame_upsample (encoder_frame->reconstructed_frame);
  }
}

void
schro_encoder_postanalyse_picture (SchroEncoderFrame *frame)
{
#if 0
  double mse;

  mse = schro_frame_mean_squared_error (frame->original_frame,
      frame->reconstructed_frame->frames[0]);

  SCHRO_ERROR("mse %g psnr %g", mse, 10*log(255*255/mse)/log(10));
#endif

#if 0
  double mssim;

  mssim = schro_ssim (frame->original_frame,
      frame->reconstructed_frame->frames[0]);
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

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
        &frame->motion_field->motion_vectors[j*params->x_num_blocks + i];

      SCHRO_ASSERT(mv->split < 3);

      split_prediction = schro_motion_split_prediction (
          frame->motion_field->motion_vectors, params, i, j);
      split_residual = (mv->split - split_prediction + 3)%3;
      _schro_arith_encode_uint (arith, SCHRO_CTX_SB_F1,
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

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];
          int pred_mode;

          pred_mode = schro_motion_get_mode_prediction(frame->motion_field,
              i+k,j+l) ^ mv->pred_mode;

          _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
              pred_mode & 1);
          if (params->num_refs > 1) {
            _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
                pred_mode >> 1);
          }
          if (mv->pred_mode != 0) {
            if (params->have_global_motion) {
              int pred;
              schro_motion_field_get_global_prediction (frame->motion_field,
                  i+k, j+l, &pred);
              _schro_arith_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
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
        &frame->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          int pred_x, pred_y;
          SchroMotionVector *mv =
            &frame->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if ((mv->pred_mode>>ref) & 1 && !mv->using_global) {
            schro_motion_vector_prediction (frame->motion_field->motion_vectors,
                params, i+k, j+l, &pred_x, &pred_y, 1<<ref);

            if (xy == 0) {
              _schro_arith_encode_sint(arith,
                  cont, value, sign,
                  (mv->x1 - pred_x)>>(3-params->mv_precision));
            } else {
              _schro_arith_encode_sint(arith,
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

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->motion_field->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->motion_field->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if (mv->pred_mode == 0) {
            int pred[3];
            SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

            schro_motion_dc_prediction (frame->motion_field->motion_vectors,
                params, i+k, j+l, pred);

            switch (comp) {
              case 0:
                _schro_arith_encode_sint (arith,
                    SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
                    SCHRO_CTX_LUMA_DC_SIGN,
                    mvdc->dc[0] - pred[0]);
                break;
              case 1:
                _schro_arith_encode_sint (arith,
                    SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
                    SCHRO_CTX_CHROMA1_DC_SIGN,
                    mvdc->dc[1] - pred[1]);
                break;
              case 2:
                _schro_arith_encode_sint (arith,
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
        (int32_t)(frame->reference_frame_number[i] - frame->frame_number));
  }

  /* retire list */
  if (!frame->params.is_lowdelay) {
    schro_bits_encode_uint (frame->bits, frame->n_retire);
    for(i=0;i<frame->n_retire;i++){
      schro_bits_encode_sint (frame->bits,
          (int32_t)(frame->retire[i] - frame->frame_number));
    }
  } else {
    SCHRO_ASSERT(frame->n_retire == 0);
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
  if (!params->is_lowdelay) {
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
  } else {
    int encode_quant_matrix;
    int encode_quant_offsets;

    schro_bits_encode_uint (bits, params->slice_width_exp);
    schro_bits_encode_uint (bits, params->slice_height_exp);
    schro_bits_encode_uint (bits, params->slice_bytes_num);
    schro_bits_encode_uint (bits, params->slice_bytes_denom);

    /* FIXME */
    encode_quant_matrix = TRUE;
    encode_quant_offsets = TRUE;

    schro_bits_encode_bit (bits, encode_quant_matrix);
    if (encode_quant_matrix) {
      int i;
      schro_bits_encode_uint (bits, params->quant_matrix[0]);
      for(i=0;i<params->transform_depth;i++){
        schro_bits_encode_uint (bits, params->quant_matrix[1+3*i]);
        schro_bits_encode_uint (bits, params->quant_matrix[2+3*i]);
        schro_bits_encode_uint (bits, params->quant_matrix[3+3*i]);
      }
    }
    schro_bits_encode_bit (bits, encode_quant_offsets);
    if (encode_quant_offsets) {
      schro_bits_encode_uint (bits, params->luma_quant_offset);
      schro_bits_encode_uint (bits, params->chroma1_quant_offset);
      schro_bits_encode_uint (bits, params->chroma2_quant_offset);
    }
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
  SchroParams *params = &frame->params;
  int stride;
  int width;
  int height;
  int w;
  int h;
  int shift;
  int16_t *data;
  int16_t *line;
  int i,j;
  int position;

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      params, &data, &stride, &width, &height);

  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);
  if (component == 0) {
    w = ROUND_UP_SHIFT(params->video_format->width, shift);
    h = ROUND_UP_SHIFT(params->video_format->height, shift);
  } else {
    w = ROUND_UP_SHIFT(params->video_format->chroma_width, shift);
    h = ROUND_UP_SHIFT(params->video_format->chroma_height, shift);
  }

  h = MIN (h + wavelet_extent[params->wavelet_filter_index], height);
  w = MIN (w + wavelet_extent[params->wavelet_filter_index], width);

  if (w < width) {
    for(j=0;j<h;j++){
      line = OFFSET(data, j*stride);
      for(i=w;i<width;i++){
        line[i] = 0;
      }
    }
  }
  for(j=h;j<height;j++){
    line = OFFSET(data, j*stride);
    for(i=0;i<width;i++){
      line[i] = 0;
    }
  }
}

static void
schro_encoder_encode_transform_data (SchroEncoderFrame *frame)
{
  int i;
  int component;
  SchroParams *params = &frame->params;

  schro_encoder_choose_quantisers (frame);
  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      if (i != 0) schro_bits_sync (frame->bits);
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
quantize (int value, int quant_factor, int quant_offset)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)<<2;
    x /= quant_factor;
    value = -x;
  } else {
    x = value<<2;
    x /= quant_factor;
    value = x;
  }
  return value;
}

static int
schro_encoder_quantize_subband (SchroEncoderFrame *frame, int component, int index,
    int16_t *quant_data)
{
  int pred_value;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int stride;
  int width;
  int height;
  int i,j;
  int16_t *data;
  int16_t *line;
  int16_t *prev_line;
  int subband_zero_flag;
  int position;

  subband_zero_flag = 1;

  /* FIXME doesn't handle quantisation of codeblocks */

  quant_index = frame->quant_index[component][index];
  quant_factor = schro_table_quant[quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);

  if (index == 0) {
    for(j=0;j<height;j++){
      line = OFFSET(data, j*stride);
      prev_line = OFFSET(data, (j-1)*stride);

      for(i=0;i<width;i++){
        int q;

        if (frame->params.num_refs == 0) {
          if (j>0) {
            if (i>0) {
              pred_value = schro_divide(line[i - 1] +
                  prev_line[i] + prev_line[i - 1] + 1,3);
            } else {
              pred_value = prev_line[i];
            }
          } else {
            if (i>0) {
              pred_value = line[i - 1];
            } else {
              pred_value = 0;
            }
          }
        } else {
          pred_value = 0;
        }

        q = quantize(line[i] - pred_value, quant_factor, quant_offset);
        line[i] = dequantize(q, quant_factor, quant_offset) +
          pred_value;
        quant_data[j*width + i] = q;
        if (line[i] != 0) {
          subband_zero_flag = 0;
        }

      }
    }
  } else {
    for(j=0;j<height;j++){
      line = OFFSET(data, j*stride);

      for(i=0;i<width;i++){
        int q;

        q = quantize(line[i], quant_factor, quant_offset);
        line[i] = dequantize(q, quant_factor, quant_offset);
        quant_data[j*width + i] = q;
        if (line[i] != 0) {
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
  SchroArith *arith;
  int16_t *data;
  int16_t *parent_data;
  int parent_stride;
  int i,j;
  int subband_zero_flag;
  int stride;
  int width;
  int height;
  int16_t *quant_data;
  int x,y;
  int horiz_codeblocks;
  int vert_codeblocks;
  int have_zero_flags;
  int have_quant_offset;
  int position;

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      params, &data, &stride, &width, &height);

  if (position >= 4) {
    int parent_width;
    int parent_height;
    schro_subband_get (frame->iwt_frame, component, position - 4,
        params, &parent_data, &parent_stride, &parent_width, &parent_height);
  } else {
    parent_data = NULL;
    parent_stride = 0;
  }

  arith = schro_arith_new ();
  schro_arith_encode_init (arith, frame->subband_buffer);

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
      horiz_codeblocks = params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
      vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
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
    _schro_arith_encode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK,
        zero_codeblock);
    if (zero_codeblock) {
      continue;
    }
  }

  if (have_quant_offset) {
    _schro_arith_encode_sint (arith,
        SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
        SCHRO_CTX_QUANTISER_SIGN, 0);
  }

  for(j=ymin;j<ymax;j++){
    int16_t *parent_line = OFFSET(parent_data, (j>>1)*parent_stride);

    for(i=xmin;i<xmax;i++){
      int parent;
      int cont_context;
      int value_context;
      int nhood_or;
      int previous_value;
      int sign_context;

      /* FIXME This code is so ugly.  Most of these if statements
       * are constant over the entire codeblock. */

      if (position >= 4) {
        parent = parent_line[(i>>1)];
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
      if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(position)) {
        if (i > 0) {
          previous_value = quant_data[j*width + i - 1];
        }
      } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(position)) {
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

      _schro_arith_encode_sint (arith, cont_context, value_context,
          sign_context, quant_data[j*width + i]);
    }
  }
    }
  }

  schro_arith_flush (arith);

  SCHRO_ASSERT(arith->offset < frame->subband_size);

  SCHRO_INFO("SUBBAND_EST: %d %d %d %d", component, index,
      frame->estimated_entropy, arith->offset*8);

  schro_bits_encode_uint (frame->bits, arith->offset);
  if (arith->offset > 0) {
    schro_bits_encode_uint (frame->bits,
        frame->quant_index[component][index]);

    schro_bits_sync (frame->bits);

    schro_bits_append (frame->bits, arith->buffer->data, arith->offset);
  }
  schro_arith_free (arith);
}

static void
schro_encoder_encode_lowdelay_transform_data (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int x,y;
  int slice_width;
  int slice_height;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;

  slice_width = 1<<params->slice_width_exp;
  slice_height = 1<<params->slice_height_exp;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  accumulator = 0;
  for(y=0;y<params->iwt_luma_height;y+=slice_height) {
    for(x=0;x<params->iwt_luma_width;x+=slice_width) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      schro_encoder_estimate_slice (frame, x, y, n_bytes + extra, 20);
      schro_encoder_encode_slice (frame, x, y, n_bytes + extra, 20);
    }
  }

}

/* FIXME dup'd in schrodecoder.c */
static void
schro_slice_get (SchroFrame *frame, int component, int position,
    SchroParams *params,
    int16_t **data, int *stride, int *width, int *height)
{
  int shift;
  int w;
  SchroFrameComponent *comp = &frame->components[component];

  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);

  *stride = comp->stride << shift;
  *width = (1<<params->slice_width_exp) >> shift;
  *height = (1<<params->slice_height_exp) >> shift;
  w = params->iwt_luma_width >> shift;
  if (component > 0) {
    *width >>= params->video_format->chroma_h_shift;
    *height >>= params->video_format->chroma_v_shift;
    w = params->iwt_chroma_width >> shift;
  }

  *data = comp->data;
  if (position & 2) {
    *data = OFFSET(*data, (*stride)>>1);
  }
  if (position & 1) {
    *data = OFFSET(*data, w*sizeof(int16_t));
  }
}

/* FIXME dup'd in schrodecoder.c */
static int
ilog2up (unsigned int x)
{
  int i;

  for(i=0;i<32;i++){
    if (x == 0) return i;
    x >>= 1;
  }
  return 0;
}

static int
schro_dc_predict (int16_t *data, int stride, int x, int y)
{
  int16_t *line = OFFSET(data, stride * y);
  int16_t *prev_line = OFFSET(data, stride * (y-1));

  if (y > 0) {
    if (x > 0) {
      return schro_divide(line[x-1] + prev_line[x] + prev_line[x-1] + 1,3);
    } else {
      return prev_line[x];
    }
  } else {
    if (x > 0) {
      return line[x-1];
    } else {
      return 0;
    }
  }
}

void
schro_encoder_encode_slice (SchroEncoderFrame *frame, int x, int y,
    int slice_bytes, int qindex)
{
  SchroParams *params = &frame->params;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;
  int length_bits;
  int i;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int ix, iy;
  int q;
  int y_bits;
  int start_bits;
  int end_bits;

  SCHRO_DEBUG("bytes %d index %d", slice_bytes, qindex);

  start_bits = schro_bits_get_bit_offset (frame->bits);

  if (frame->bits->shift != 7) {
    SCHRO_ERROR("unsynchronized bits");
  }

  schro_bits_encode_bits (frame->bits, 7, qindex);

  length_bits = ilog2up (8*slice_bytes);
  y_bits = (8*slice_bytes - 7 - length_bits) / 2;
  schro_bits_encode_bits (frame->bits, length_bits, y_bits);
  
  for (i=0;i<1+3*params->transform_depth;i++){
    int pos = schro_subband_get_position(i);
    int sx = x >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos));
    int sy = y >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos));

    schro_slice_get (frame->iwt_frame, 0, pos,
        params, &data, &stride, &width, &height);

    quant_index = qindex + params->quant_matrix[i] + params->luma_quant_offset;
    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    if (i==0) {
      int pred_value;

      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          pred_value = schro_dc_predict (data, stride, sx+ix, sy+iy);
          q = quantize (line[sx+ix] - pred_value, quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, q);
          line[sx+ix] = pred_value + dequantize (q, quant_factor, quant_offset);
        }
      }
    } else {
      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          q = quantize (line[sx+ix], quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, q);
//          line[sx+ix] = dequantize (q, quant_factor, quant_offset);
        }
      }
    }
  }

  for (i=0;i<1+3*params->transform_depth;i++){
    int16_t *data2;
    int quant_factor2;
    int quant_offset2;
    int pos = schro_subband_get_position(i);
    int sx = x >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos) + params->video_format->chroma_h_shift);
    int sy = y >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos) + params->video_format->chroma_v_shift);

    schro_slice_get (frame->iwt_frame, 1, pos,
        params, &data, &stride, &width, &height);
    schro_slice_get (frame->iwt_frame, 2, pos,
        params, &data2, &stride, &width, &height);

    quant_index = qindex + params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    quant_index = qindex + params->quant_matrix[i] + params->chroma2_quant_offset;
    quant_factor2 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset2  = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    if (i==0) {
      int pred_value;
      int16_t *line2;

      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        line2 = OFFSET(data2, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          pred_value = schro_dc_predict (data, stride, sx+ix, sy+iy);
          q = quantize (line[sx+ix] - pred_value, quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, q);
          line[sx+ix] = pred_value + dequantize (q, quant_factor, quant_offset);

          pred_value = schro_dc_predict (data2, stride, sx+ix, sy+iy);
          q = quantize (line2[sx+ix] - pred_value, quant_factor2, quant_offset2);
          schro_bits_encode_sint (frame->bits, q);
          line2[sx+ix] = pred_value + dequantize (q, quant_factor2, quant_offset2);
        }
      }
    } else {
      int16_t *line2;

      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        line2 = OFFSET(data2, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          q = quantize (line[sx+ix], quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, q);
//          line[sx+ix] = dequantize (q, quant_factor, quant_offset);
          q = quantize (line2[sx+ix], quant_factor2, quant_offset2);
          schro_bits_encode_sint (frame->bits, q);
//          line2[sx+ix] = dequantize (q, quant_factor2, quant_offset2);
        }
      }
    }
  }

  end_bits = schro_bits_get_bit_offset (frame->bits);
  SCHRO_DEBUG("total bits %d used bits %d", slice_bytes*8,
      end_bits - start_bits);

  if (end_bits - start_bits > slice_bytes*8) {
    SCHRO_ERROR("slice overran buffer (oops)");
  } else {
    int left = slice_bytes*8 - (end_bits - start_bits);
    for(i=0;i<left; i++) {
      schro_bits_encode_bit (frame->bits, 0);
    }
  }
}

int
schro_encoder_estimate_slice (SchroEncoderFrame *frame, int x, int y,
    int slice_bytes, int qindex)
{
  SchroParams *params = &frame->params;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;
  int length_bits;
  int i;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int ix, iy;
  int q;
  int n_bits = 0;

  SCHRO_DEBUG("estimating slice at %d %d", x, y);

  n_bits += 7;

  length_bits = ilog2up (8*slice_bytes);
  n_bits += length_bits;
  
  for (i=0;i<1+3*params->transform_depth;i++){
    int pos = schro_subband_get_position(i);
    int sx = x >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos));
    int sy = y >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos));

    schro_slice_get (frame->iwt_frame, 0, pos,
        params, &data, &stride, &width, &height);

    SCHRO_DEBUG("slice subband %d pos %dx%d size %dx%d", i, sx, sy, width, height);

    quant_index = qindex + params->quant_matrix[i] + params->luma_quant_offset;
    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    if (i==0) {
      int pred_value;

      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          pred_value = schro_dc_predict (data, stride, sx+ix, sy+iy);
          q = quantize (data[sx+ix] - pred_value, quant_factor, quant_offset);
          n_bits += schro_bits_estimate_sint (q);
        }
      }
    } else {
      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          q = quantize (data[sx+ix], quant_factor, quant_offset);
          n_bits += schro_bits_estimate_sint (q);
        }
      }
    }
  }

  for (i=0;i<1+3*params->transform_depth;i++){
    int16_t *data2;
    int quant_factor2;
    int quant_offset2;
    int pos = schro_subband_get_position(i);
    int sx = x >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos) + params->video_format->chroma_h_shift);
    int sy = y >> (params->transform_depth - SCHRO_SUBBAND_SHIFT(pos) + params->video_format->chroma_v_shift);

    schro_slice_get (frame->iwt_frame, 1, pos,
        params, &data, &stride, &width, &height);
    schro_slice_get (frame->iwt_frame, 2, pos,
        params, &data2, &stride, &width, &height);

    SCHRO_DEBUG("slice subband %d pos %dx%d size %dx%d", i, sx, sy, width, height);

    quant_index = qindex + params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    quant_index = qindex + params->quant_matrix[i] + params->chroma2_quant_offset;
    quant_factor2 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset2  = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    if (i==0) {
      int pred_value;
      int16_t *line2;

      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        line2 = OFFSET(data2, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          pred_value = schro_dc_predict (data, stride, sx+ix, sy+iy);
          q = quantize (data[sx+ix] - pred_value, quant_factor, quant_offset);
          n_bits += schro_bits_estimate_sint (q);

          pred_value = schro_dc_predict (data2, stride, sx+ix, sy+iy);
          q = quantize (data2[sx+ix] - pred_value, quant_factor2,
              quant_offset2);
          n_bits += schro_bits_estimate_sint (q);
        }
      }
    } else {
      for(iy=0;iy<height;iy++) {
        line = OFFSET(data, stride * (sy+iy));
        for(ix=0;ix<width;ix++){
          q = quantize (data[sx+ix], quant_factor, quant_offset);
          n_bits += schro_bits_estimate_sint (q);

          q = quantize (data2[sx+ix], quant_factor2, quant_offset2);
          n_bits += schro_bits_estimate_sint (q);
        }
      }
    }
  }

  return n_bits;
}

/* frame queue */

SchroEncoderFrame *
schro_encoder_frame_new (SchroEncoder *encoder)
{
  SchroEncoderFrame *encoder_frame;
  SchroFrameFormat frame_format;
  int frame_width;
  int frame_height;

  encoder_frame = malloc(sizeof(SchroEncoderFrame));
  memset (encoder_frame, 0, sizeof(SchroEncoderFrame));
  encoder_frame->state = SCHRO_ENCODER_FRAME_STATE_NEW;
  encoder_frame->refcount = 1;

  frame_format = schro_params_get_frame_format (16,
      encoder->video_format.chroma_format);
  
  frame_width = ROUND_UP_POW2(encoder->video_format.width,
      SCHRO_MAX_TRANSFORM_DEPTH + encoder->video_format.chroma_h_shift);
  frame_height = ROUND_UP_POW2(encoder->video_format.height,
      SCHRO_MAX_TRANSFORM_DEPTH + encoder->video_format.chroma_v_shift);

  encoder_frame->iwt_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);
  
  frame_width = MAX(
      4 * 12 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*12),
      4 * 16 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*16));
  frame_height = MAX(
      4 * 12 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*12),
      4 * 16 * DIVIDE_ROUND_UP(encoder->video_format.width, 4*16));

  encoder_frame->prediction_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);

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
    if (frame->iwt_frame) {
      schro_frame_unref (frame->iwt_frame);
    }
    if (frame->prediction_frame) {
      schro_frame_unref (frame->prediction_frame);
    }
    if (frame->motion_field) {
      schro_motion_field_free (frame->motion_field);
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

static const int pref_range[][2] = {
  { 0, 7 }, /* engine */
  { 2, 20 }, /* ref distance */
  { 1, SCHRO_MAX_ENCODER_TRANSFORM_DEPTH }, /* transform depth */
  { 0, 7 }, /* intra wavelet */
  { 0, 7 }, /* inter wavelet */
  { 0, 100 }, /* lambda */
  { 0, 100 }, /* psnr */
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

