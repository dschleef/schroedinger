
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#if 0
/* Used for testing bitstream */
#define MARKER(pack) schro_pack_encode_uint (pack, 1234567)
#else
#define MARKER(pack)
#endif

static void schro_encoder_encode_picture_prediction (SchroEncoderFrame *frame);
static void schro_encoder_encode_superblock_split (SchroEncoderFrame *frame);
static void schro_encoder_encode_prediction_modes (SchroEncoderFrame *frame);
static void schro_encoder_encode_vector_data (SchroEncoderFrame *frame, int ref, int xy);
static void schro_encoder_encode_dc_data (SchroEncoderFrame *frame, int comp);
static void schro_encoder_encode_transform_parameters (SchroEncoderFrame *frame);
static void schro_encoder_encode_transform_data (SchroEncoderFrame *frame);
static int schro_encoder_pull_is_ready_locked (SchroEncoder *encoder);
static void schro_encoder_encode_codec_comment (SchroEncoder *encoder);
static void schro_encoder_encode_bitrate_comment (SchroEncoder *encoder,
    unsigned int bitrate);
static void schro_encoder_clean_up_transform_subband (SchroEncoderFrame *frame,
    int component, int index);
static void schro_encoder_fixup_offsets (SchroEncoder *encoder,
    SchroBuffer *buffer);
static int schro_encoder_iterate (SchroEncoder *encoder);
static void schro_encoder_init_perceptual_weighting (SchroEncoder *encoder);

SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;

  encoder = malloc(sizeof(SchroEncoder));
  memset (encoder, 0, sizeof(SchroEncoder));

  encoder->version_major = 0;
  encoder->version_minor = 110;

  encoder->au_frame = -1;

  encoder->last_ref = -1;
  encoder->last_ref2 = -1;
  encoder->next_ref = -1;
  encoder->mid1_ref = -1;
  encoder->mid2_ref = -1;

#if 0
  encoder->prefs[SCHRO_PREF_ENGINE] = 1;
  encoder->prefs[SCHRO_PREF_QUANT_ENGINE] = 0;
  encoder->prefs[SCHRO_PREF_REF_DISTANCE] = 4;
  encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 4;
  encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_DESLAURIES_DUBUC_9_7;
  encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_LE_GALL_5_3;
  encoder->prefs[SCHRO_PREF_LAMBDA] = 1;
  encoder->prefs[SCHRO_PREF_PSNR] = 25;
  encoder->prefs[SCHRO_PREF_BITRATE] = 13824000;
  encoder->prefs[SCHRO_PREF_NOARITH] = 0;
  encoder->prefs[SCHRO_PREF_MD5] = 0;
#endif
  encoder->rate_control = 0;
  encoder->bitrate = 13824000;
  encoder->max_bitrate = 13824000;
  encoder->min_bitrate = 13824000;
  encoder->noise_threshold = 25.0;
  encoder->gop_structure = 0;
  encoder->perceptual_weighting = 0;
  encoder->filtering = 0;
  encoder->filter_value = 5.0;
  encoder->profile = 0;
  encoder->level = 0;
  encoder->au_distance = 30;
  encoder->enable_psnr = FALSE;
  encoder->enable_ssim = FALSE;
  encoder->enable_md5 = FALSE;

  encoder->ref_distance = 4;
  encoder->transform_depth = 4;
  encoder->intra_wavelet = SCHRO_WAVELET_DESLAURIES_DUBUC_9_7;
  encoder->inter_wavelet = SCHRO_WAVELET_LE_GALL_5_3;
  encoder->interlaced_coding = FALSE;
  encoder->enable_internal_testing = FALSE;
  encoder->enable_noarith = FALSE;
  encoder->enable_fullscan_prediction = FALSE;
  encoder->enable_hierarchical_prediction = TRUE;
  encoder->enable_zero_prediction = FALSE;
  encoder->enable_phasecorr_prediction = FALSE;

  encoder->magic_dc_metric_offset = 1.0;

  schro_video_format_set_std_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_CUSTOM);

  /* FIXME this should be a parameter */
  encoder->queue_depth = 20;

  encoder->frame_queue = schro_queue_new (encoder->queue_depth,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  encoder->reference_queue = schro_queue_new (SCHRO_LIMIT_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  schro_encoder_set_default_subband_weights (encoder);

  encoder->inserted_buffers =
    schro_list_new_full ((SchroListFreeFunc)schro_buffer_unref, NULL);

  return encoder;
}

static void
handle_gop_enum (SchroEncoder *encoder)
{
  switch (encoder->gop_structure) {
    case SCHRO_ENCODER_GOP_ADAPTIVE:
    case SCHRO_ENCODER_GOP_BACKREF:
    case SCHRO_ENCODER_GOP_CHAINED_BACKREF:
      encoder->engine_iterate = schro_encoder_engine_backref;
      break;
    case SCHRO_ENCODER_GOP_INTRA_ONLY:
      encoder->engine_iterate = schro_encoder_engine_intra_only;
      break;
    case SCHRO_ENCODER_GOP_BIREF:
    case SCHRO_ENCODER_GOP_CHAINED_BIREF:
      encoder->engine_iterate = schro_encoder_engine_tworef;
      break;
  }
}

void
schro_encoder_start (SchroEncoder *encoder)
{
  encoder->engine_init = 1;

  schro_encoder_encode_codec_comment (encoder);

  schro_encoder_init_perceptual_weighting (encoder);

  encoder->async = schro_async_new (0, (int (*)(void *))schro_encoder_iterate,
      encoder);

  switch (encoder->rate_control) {
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_NOISE_THRESHOLD:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_SIMPLE;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_RATE_DISTORTION;

      encoder->buffer_size = encoder->bitrate;
      encoder->buffer_level = 0;
      encoder->bits_per_picture = muldiv64 (encoder->bitrate,
            encoder->video_format.frame_rate_denominator,
            encoder->video_format.frame_rate_numerator);

      schro_encoder_recalculate_allocations (encoder);

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY:
      encoder->engine_iterate = schro_encoder_engine_lowdelay;
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOSSLESS:
      encoder->engine_iterate = schro_encoder_engine_lossless;
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOSSLESS;
      break;
  }


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

  schro_list_free (encoder->inserted_buffers);

  free (encoder);
}

static void
schro_encoder_init_perceptual_weighting (SchroEncoder *encoder)
{
  encoder->pixels_per_degree_horiz =
    encoder->video_format.width/
    (2.0*atan(0.5/encoder->perceptual_distance)*180/M_PI);
  encoder->pixels_per_degree_vert =
    encoder->video_format.aspect_ratio_numerator * 
    (encoder->pixels_per_degree_horiz / encoder->video_format.aspect_ratio_denominator);

  SCHRO_DEBUG("pixels per degree horiz=%g vert=%g",
      encoder->pixels_per_degree_horiz, encoder->pixels_per_degree_vert);

  switch(encoder->perceptual_weighting) {
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

  schro_video_format_validate (&encoder->video_format);
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
  if (format == frame->format) {
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

  if (schro_list_get_size(encoder->inserted_buffers)>0) {
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

  if (schro_list_get_size(encoder->inserted_buffers)>0) {
    buffer = schro_list_remove (encoder->inserted_buffers, 0);
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
      } else if (schro_list_get_size(frame->inserted_buffers)>0) {
        buffer = schro_list_remove (frame->inserted_buffers, 0);
        *presentation_frame = -1;
      } else {
        buffer = frame->output_buffer;
        frame->output_buffer = NULL;

        frame->state = SCHRO_ENCODER_FRAME_STATE_FREE;
        encoder->output_slot++;

        encoder->buffer_level -= encoder->bits_per_picture;
        {
          /* FIXME move this */
          double x;

          x = (double)frame->actual_bits / frame->estimated_entropy;
          if (encoder->average_arith_context_ratio == 0) {
            encoder->average_arith_context_ratio = x;
          } else {
            double alpha = 0.9;
            encoder->average_arith_context_ratio *= alpha;
            encoder->average_arith_context_ratio += (1.0-alpha) * x;
          }

        }

        schro_encoder_shift_frame_queue (encoder);
      }

      encoder->buffer_level += buffer->length * 8;
      if (encoder->buffer_level < 0) {
        SCHRO_DEBUG("buffer underrun");
        encoder->buffer_level = 0;
      }
      if (encoder->buffer_level < 0) {
      }
      SCHRO_DEBUG("buffer level %d (%d)", encoder->buffer_level,
          encoder->buffer_size);

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
  char *s = "Schroedinger " VERSION;
  SchroBuffer *buffer;

  buffer = schro_encoder_encode_auxiliary_data (encoder,
      SCHRO_AUX_DATA_ENCODER_STRING, s, strlen(s));
  
  schro_encoder_insert_buffer (encoder, buffer);
}

static void
schro_encoder_encode_bitrate_comment (SchroEncoder *encoder,
    unsigned int bitrate)
{
  uint8_t s[4];
  SchroBuffer *buffer;

  s[0] = (bitrate>>24)&0xff;
  s[1] = (bitrate>>16)&0xff;
  s[2] = (bitrate>>8)&0xff;
  s[3] = (bitrate>>0)&0xff;
  buffer = schro_encoder_encode_auxiliary_data (encoder,
      SCHRO_AUX_DATA_BITRATE, s, 4);
  
  schro_encoder_insert_buffer (encoder, buffer);
}

static void
schro_encoder_encode_md5_checksum (SchroEncoderFrame *frame)
{
  SchroBuffer *buffer;
  uint32_t checksum[4];

  schro_frame_md5 (frame->reconstructed_frame->frames[0], checksum);
  buffer = schro_encoder_encode_auxiliary_data (frame->encoder,
      SCHRO_AUX_DATA_MD5_CHECKSUM, checksum, 16);
  
  schro_encoder_frame_insert_buffer (frame, buffer);
}

void
schro_encoder_insert_buffer (SchroEncoder *encoder, SchroBuffer *buffer)
{
  schro_list_append (encoder->inserted_buffers, buffer);
}

void
schro_encoder_frame_insert_buffer (SchroEncoderFrame *frame,
    SchroBuffer *buffer)
{
  schro_list_append (frame->inserted_buffers, buffer);
}

SchroBuffer *
schro_encoder_encode_auxiliary_data (SchroEncoder *encoder,
    SchroAuxiliaryDataID id, void *data, int size)
{
  SchroPack *pack;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (size + SCHRO_PARSE_HEADER_SIZE + 1);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_AUXILIARY_DATA);
  schro_pack_encode_bits (pack, 8, id);
  schro_pack_append (pack, data, size);

  schro_pack_free (pack);

  return buffer;
}

SchroBuffer *
schro_encoder_encode_access_unit (SchroEncoder *encoder)
{
  SchroPack *pack;
  SchroBuffer *buffer;
  SchroBuffer *subbuffer;

  buffer = schro_buffer_new_and_alloc (0x100);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_access_unit_header (encoder, pack);

  schro_pack_flush (pack);

  subbuffer = schro_buffer_new_subbuffer (buffer, 0,
      schro_pack_get_offset (pack));
  schro_pack_free (pack);
  schro_buffer_unref (buffer);

  return subbuffer;
}

SchroBuffer *
schro_encoder_encode_end_of_stream (SchroEncoder *encoder)
{
  SchroPack *pack;
  SchroBuffer *buffer;

  buffer = schro_buffer_new_and_alloc (SCHRO_PARSE_HEADER_SIZE);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_END_SEQUENCE);

  schro_pack_free (pack);

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

  return encoder->engine_iterate (encoder);
#if 0
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
#endif
}

void
schro_encoder_analyse_picture (SchroEncoderFrame *frame)
{
  if (frame->need_filtering) {
    frame->filtered_frame = schro_frame_dup (frame->original_frame);
    //schro_frame_filter_addnoise (frame->filtered_frame, 0);
    //schro_frame_filter_lowpass2 (frame->filtered_frame, 2.0);
    //schro_frame_filter_lowpass (frame->filtered_frame);
    //schro_frame_filter_adaptive_lowpass (frame->filtered_frame);
    //schro_frame_filter_cwm7 (frame->filtered_frame);
    schro_frame_filter_cwmN (frame->filtered_frame, 5);
  } else {
    frame->filtered_frame = schro_frame_ref (frame->original_frame);
  }

  if (frame->need_downsampling) {
    schro_encoder_frame_downsample (frame);
    frame->have_downsampling = TRUE;
  }

  if (frame->need_average_luma) {
    if (frame->have_downsampling) {
      frame->average_luma =
        schro_frame_calculate_average_luma (frame->downsampled_frames[3]);
    } else {
      frame->average_luma =
        schro_frame_calculate_average_luma (frame->filtered_frame);
    }
    frame->have_average_luma = TRUE;
  }
}

void
schro_encoder_predict_picture (SchroEncoderFrame *frame)
{
  SCHRO_INFO("predict picture %d", frame->frame_number);

  frame->tmpbuf = malloc(SCHRO_LIMIT_WIDTH * 2);
  frame->tmpbuf2 = malloc(SCHRO_LIMIT_WIDTH * 2);

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict (frame);

    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);

    schro_motion_verify (frame->motion);
    schro_motion_render (frame->motion, frame->prediction_frame);

    schro_frame_subtract (frame->iwt_frame, frame->prediction_frame);

    schro_frame_zero_extend (frame->iwt_frame,
        frame->params.video_format->width,
        frame->params.video_format->height);
  } else {
    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);
  }

  schro_frame_iwt_transform (frame->iwt_frame, &frame->params,
      frame->tmpbuf);
  schro_encoder_clean_up_transform (frame);

  /* FIXME this needs a better place */
  schro_encoder_choose_quantisers (frame);
  schro_encoder_estimate_entropy (frame);
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
      SCHRO_LIMIT_TRANSFORM_DEPTH + frame->encoder->video_format.chroma_h_shift);
  frame_height = ROUND_UP_POW2(frame->encoder->video_format.height,
      SCHRO_LIMIT_TRANSFORM_DEPTH + frame->encoder->video_format.chroma_v_shift);

  frame->quant_data = malloc (sizeof(int16_t) * frame_width * frame_height / 4);

  frame->pack = schro_pack_new ();
  schro_pack_encode_init (frame->pack, frame->output_buffer);

  /* encode header */
  schro_encoder_encode_parse_info (frame->pack,
      SCHRO_PARSE_CODE_PICTURE(frame->is_ref, frame->params.num_refs,
        frame->params.is_lowdelay, frame->params.is_noarith));
  schro_encoder_encode_picture_header (frame);

  if (frame->params.num_refs > 0) {
    schro_pack_sync(frame->pack);
    schro_encoder_encode_picture_prediction (frame);
    schro_pack_sync(frame->pack);
    frame->actual_mc_bits = -schro_pack_get_offset(frame->pack) * 8;
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
    frame->actual_mc_bits += schro_pack_get_offset(frame->pack) * 8;
  }

  schro_pack_sync(frame->pack);
  schro_encoder_encode_transform_parameters (frame);

  residue_bits_start = schro_pack_get_offset(frame->pack) * 8;

  schro_pack_sync(frame->pack);
  if (frame->params.is_lowdelay) {
    schro_encoder_encode_lowdelay_transform_data (frame);
  } else {
    schro_encoder_encode_transform_data (frame);
  }

  schro_pack_flush (frame->pack);
  frame->actual_bits = schro_pack_get_offset (frame->pack)*8;

  schro_dump (SCHRO_DUMP_PICTURE, "%d %d %g %d %d %g %d %d %d %d\n",
      frame->frame_number, frame->num_refs,
      frame->allocated_bits*frame->allocation_modifier,
      frame->estimated_entropy, frame->actual_bits,
      frame->scene_change_score, frame->actual_mc_bits,
      frame->stats_dc, frame->stats_global, frame->stats_motion);

  subbuffer = schro_buffer_new_subbuffer (frame->output_buffer, 0,
      schro_pack_get_offset (frame->pack));
  schro_buffer_unref (frame->output_buffer);
  frame->output_buffer = subbuffer;

  if (frame->params.num_refs > 0) {
    SCHRO_INFO("pred bits %d, residue bits %d, dc %d, global = %d, motion %d",
        residue_bits_start, schro_pack_get_offset(frame->pack)*8 - residue_bits_start,
        frame->stats_dc, frame->stats_global, frame->stats_motion);
  }

  if (frame->subband_buffer) {
    schro_buffer_unref (frame->subband_buffer);
  }
  if (frame->quant_data) {
    free (frame->quant_data);
  }
  if (frame->pack) {
    schro_pack_free (frame->pack);
    frame->pack = NULL;
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

  if (encoder_frame->encoder->enable_md5) {
    schro_encoder_encode_md5_checksum (encoder_frame);
  }

  if (encoder_frame->is_ref) {
    schro_upsampled_frame_upsample (encoder_frame->reconstructed_frame);
  }
}

void
schro_encoder_postanalyse_picture (SchroEncoderFrame *frame)
{
  if (frame->encoder->enable_psnr) {
    double mse;
    double psnr;

    mse = schro_frame_mean_squared_error (frame->filtered_frame,
        frame->reconstructed_frame->frames[0]);
    psnr = 10*log(255*255/mse)/log(10);
    if (frame->encoder->average_psnr == 0) {
      frame->encoder->average_psnr = psnr;
    } else {
      double alpha = 0.9;
      frame->encoder->average_psnr *= alpha;
      frame->encoder->average_psnr += (1.0-alpha) * psnr;
    }

    schro_dump(SCHRO_DUMP_PSNR, "%d %g %g %g\n",
        frame->frame_number, mse, psnr, frame->encoder->average_psnr);
  }

  if (frame->encoder->enable_ssim) {
    double mssim;

    mssim = schro_frame_ssim (frame->filtered_frame,
        frame->reconstructed_frame->frames[0]);
    schro_dump(SCHRO_DUMP_SSIM, "%d %g\n", frame->frame_number, mssim);
  }
}

static void
schro_encoder_encode_picture_prediction (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int index;

  /* block params */
  index = schro_params_get_block_params (params);
  schro_pack_encode_uint (frame->pack, index);
  if (index == 0) {
    schro_pack_encode_uint (frame->pack, params->xblen_luma);
    schro_pack_encode_uint (frame->pack, params->yblen_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
  }

  MARKER(frame->pack);

  /* mv precision flag */
  schro_pack_encode_uint (frame->pack, params->mv_precision);

  MARKER(frame->pack);

  /* global motion flag */
  schro_pack_encode_bit (frame->pack, params->have_global_motion);
  if (params->have_global_motion) {
    int i;
    for(i=0;i<params->num_refs;i++){
      SchroGlobalMotion *gm = params->global_motion + i;

      if (gm->b0 == 0 && gm->b1 == 0) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_sint (frame->pack, gm->b0);
        schro_pack_encode_sint (frame->pack, gm->b1);
      }

      if (gm->a_exp == 0 && gm->a00 == 1 && gm->a01 == 0 && gm->a10 == 0 &&
          gm->a11 == 1) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_uint (frame->pack, gm->a_exp);
        schro_pack_encode_sint (frame->pack, gm->a00);
        schro_pack_encode_sint (frame->pack, gm->a01);
        schro_pack_encode_sint (frame->pack, gm->a10);
        schro_pack_encode_sint (frame->pack, gm->a11);
      }

      if (gm->c_exp == 0 && gm->c0 == 0 && gm->c1 == 0) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_uint (frame->pack, gm->c_exp);
        schro_pack_encode_sint (frame->pack, gm->c0);
        schro_pack_encode_sint (frame->pack, gm->c1);
      }
    }
  }

  MARKER(frame->pack);

  /* picture prediction mode flag */
  schro_pack_encode_uint (frame->pack, params->picture_pred_mode);

  /* non-default weights flag */
  if (params->picture_weight_bits == 1 &&
      params->picture_weight_1 == 1 &&
      (params->picture_weight_2 == 1 || params->num_refs < 2)) {
    schro_pack_encode_bit (frame->pack, FALSE);
  } else {
    schro_pack_encode_bit (frame->pack, TRUE);
    schro_pack_encode_uint (frame->pack, params->picture_weight_bits);
    schro_pack_encode_sint (frame->pack, params->picture_weight_1);
    if (params->num_refs > 1) {
      schro_pack_encode_sint (frame->pack, params->picture_weight_2);
    }
  }

  MARKER(frame->pack);

}

static void
schro_encoder_encode_superblock_split (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int split_prediction;
      int split_residual;
      SchroMotionVector *mv =
        &frame->motion->motion_vectors[j*params->x_num_blocks + i];

      SCHRO_ASSERT(mv->split < 3);

      split_prediction = schro_motion_split_prediction (frame->motion, i, j);
      split_residual = (mv->split - split_prediction + 3)%3;
      if (params->is_noarith) {
        schro_pack_encode_uint (pack, split_residual);
      } else {
        _schro_arith_encode_uint (arith, SCHRO_CTX_SB_F1,
            SCHRO_CTX_SB_DATA, split_residual);
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint(frame->pack, schro_pack_get_offset(pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset(pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint(frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_prediction_modes (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->motion->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->motion->motion_vectors[(j+l)*params->x_num_blocks + i + k];
          int pred_mode;

          pred_mode = mv->pred_mode ^ 
            schro_motion_get_mode_prediction(frame->motion, i+k,j+l);

          if (params->is_noarith) {
            schro_pack_encode_bit (pack, pred_mode & 1);
          } else {
            _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1,
                pred_mode & 1);
          }
          if (params->num_refs > 1) {
            if (params->is_noarith) {
              schro_pack_encode_bit (pack, pred_mode >> 1);
            } else {
              _schro_arith_encode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2,
                  pred_mode >> 1);
            }
          }
          if (mv->pred_mode != 0) {
            if (params->have_global_motion) {
              int pred;
              pred = schro_motion_get_global_prediction (frame->motion,
                  i+k, j+l);
              if (params->is_noarith) {
                schro_pack_encode_bit (pack, mv->using_global ^ pred);
              } else {
                _schro_arith_encode_bit (arith, SCHRO_CTX_GLOBAL_BLOCK,
                    mv->using_global ^ pred);
              }
            } else {
              SCHRO_ASSERT(mv->using_global == FALSE);
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint(frame->pack, schro_pack_get_offset(pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset(pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint(frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_vector_data (SchroEncoderFrame *frame, int ref, int xy)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith = NULL;
  int cont, value, sign;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

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
        &frame->motion->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          int pred_x, pred_y;
          int x, y;
          SchroMotionVector *mv =
            &frame->motion->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if ((mv->pred_mode&(1<<ref)) && !mv->using_global) {
            schro_motion_vector_prediction (frame->motion,
                i+k, j+l, &pred_x, &pred_y, 1<<ref);
            if (ref == 0) {
              x = mv->x1;
              y = mv->y1;
            } else {
              x = mv->x2;
              y = mv->y2;
            }

            if (!params->is_noarith) {
              if (xy == 0) {
                _schro_arith_encode_sint(arith,
                    cont, value, sign, x - pred_x);
              } else {
                _schro_arith_encode_sint(arith,
                    cont, value, sign, y - pred_y);
              }
            } else {
              if (xy == 0) {
                schro_pack_encode_sint(pack, x - pred_x);
              } else {
                schro_pack_encode_sint(pack, y - pred_y);
              }
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint(frame->pack, schro_pack_get_offset(pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset(pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint(frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}

static void
schro_encoder_encode_dc_data (SchroEncoderFrame *frame, int comp)
{
  SchroParams *params = &frame->params;
  int i,j;
  SchroArith *arith = NULL;
  SchroPack b, *pack = &b;

  if (params->is_noarith) {
    schro_pack_encode_init (pack, frame->subband_buffer);
  } else {
    arith = schro_arith_new ();
    schro_arith_encode_init (arith, frame->subband_buffer);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      int k,l;
      SchroMotionVector *mv =
        &frame->motion->motion_vectors[j*params->x_num_blocks + i];

      for(l=0;l<4;l+=(4>>mv->split)) {
        for(k=0;k<4;k+=(4>>mv->split)) {
          SchroMotionVector *mv =
            &frame->motion->motion_vectors[(j+l)*params->x_num_blocks + i + k];

          if (mv->pred_mode == 0) {
            int pred[3];
            SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

            schro_motion_dc_prediction (frame->motion, i+k, j+l, pred);

            if (!params->is_noarith) {
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
            } else {
              schro_pack_encode_sint (pack, mvdc->dc[comp] - pred[comp]);
            }
          }
        }
      }
    }
  }

  schro_pack_sync (frame->pack);
  if (params->is_noarith) {
    schro_pack_flush (pack);
    schro_pack_encode_uint(frame->pack, schro_pack_get_offset(pack));
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset(pack));
  } else {
    schro_arith_flush (arith);
    schro_pack_encode_uint(frame->pack, arith->offset);
    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
    schro_arith_free (arith);
  }
}


void
schro_encoder_encode_access_unit_header (SchroEncoder *encoder,
    SchroPack *pack)
{
  SchroVideoFormat *format = &encoder->video_format;
  SchroVideoFormat _std_format;
  SchroVideoFormat *std_format = &_std_format;
  int i;

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_ACCESS_UNIT);
  
  /* parse parameters */
  schro_pack_encode_uint (pack, encoder->version_major);
  schro_pack_encode_uint (pack, encoder->version_minor);
  schro_pack_encode_uint (pack, encoder->profile);
  schro_pack_encode_uint (pack, encoder->level);

  /* sequence parameters */
  schro_pack_encode_uint (pack, encoder->video_format.index);
  schro_video_format_set_std_video_format (std_format, encoder->video_format.index);

  if (std_format->width == format->width &&
      std_format->height == format->height) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->width);
    schro_pack_encode_uint (pack, format->height);
  }

  if (std_format->chroma_format == format->chroma_format) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->chroma_format);
  }

  /* scan format */
  if (std_format->interlaced == format->interlaced &&
      (!format->interlaced ||
       (std_format->top_field_first == format->top_field_first))) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_bit (pack, format->interlaced);
    if (format->interlaced) {
      schro_pack_encode_bit (pack, format->top_field_first);
    }
  }

  MARKER(pack);

  /* frame rate */
  if (std_format->frame_rate_numerator == format->frame_rate_numerator &&
      std_format->frame_rate_denominator == format->frame_rate_denominator) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_frame_rate (format);
    schro_pack_encode_uint (pack, i);
    if (i==0) {
      schro_pack_encode_uint (pack, format->frame_rate_numerator);
      schro_pack_encode_uint (pack, format->frame_rate_denominator);
    }
  }

  MARKER(pack);

  /* pixel aspect ratio */
  if (std_format->aspect_ratio_numerator == format->aspect_ratio_numerator &&
      std_format->aspect_ratio_denominator == format->aspect_ratio_denominator) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_aspect_ratio (format);
    schro_pack_encode_uint (pack, i);
    if (i==0) {
      schro_pack_encode_uint (pack, format->aspect_ratio_numerator);
      schro_pack_encode_uint (pack, format->aspect_ratio_denominator);
    }
  }

  MARKER(pack);

  /* clean area */
  if (std_format->clean_width == format->clean_width &&
      std_format->clean_height == format->clean_height &&
      std_format->left_offset == format->left_offset &&
      std_format->top_offset == format->top_offset) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    schro_pack_encode_uint (pack, format->clean_width);
    schro_pack_encode_uint (pack, format->clean_height);
    schro_pack_encode_uint (pack, format->left_offset);
    schro_pack_encode_uint (pack, format->top_offset);
  }

  MARKER(pack);

  /* signal range */
  if (std_format->luma_offset == format->luma_offset &&
      std_format->luma_excursion == format->luma_excursion &&
      std_format->chroma_offset == format->chroma_offset &&
      std_format->chroma_excursion == format->chroma_excursion) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_signal_range (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_uint (pack, format->luma_offset);
      schro_pack_encode_uint (pack, format->luma_excursion);
      schro_pack_encode_uint (pack, format->chroma_offset);
      schro_pack_encode_uint (pack, format->chroma_excursion);
    }
  }

  MARKER(pack);

  /* colour spec */
  if (std_format->colour_primaries == format->colour_primaries &&
      std_format->colour_matrix == format->colour_matrix &&
      std_format->transfer_function == format->transfer_function) {
    schro_pack_encode_bit (pack, FALSE);
  } else {
    schro_pack_encode_bit (pack, TRUE);
    i = schro_video_format_get_std_colour_spec (format);
    schro_pack_encode_uint (pack, i);
    if (i == 0) {
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->colour_primaries);
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->colour_matrix);
      schro_pack_encode_bit (pack, TRUE);
      schro_pack_encode_uint (pack, format->transfer_function);
    }
  }

  /* interlaced coding */
  schro_pack_encode_bit (pack, encoder->interlaced_coding);

  MARKER(pack);

  schro_pack_sync (pack);
}

void
schro_encoder_encode_parse_info (SchroPack *pack, int parse_code)
{
  /* parse parameters */
  schro_pack_encode_bits (pack, 8, 'B');
  schro_pack_encode_bits (pack, 8, 'B');
  schro_pack_encode_bits (pack, 8, 'C');
  schro_pack_encode_bits (pack, 8, 'D');
  schro_pack_encode_bits (pack, 8, parse_code);

  /* offsets */
  schro_pack_encode_bits (pack, 32, 0);
  schro_pack_encode_bits (pack, 32, 0);
}

void
schro_encoder_encode_picture_header (SchroEncoderFrame *frame)
{
  int i;

  schro_pack_sync(frame->pack);
  schro_pack_encode_bits (frame->pack, 32, frame->frame_number);

  if (frame->params.num_refs > 0) {
    schro_pack_encode_sint (frame->pack,
        (int32_t)(frame->picture_number_ref0 - frame->frame_number));
    if (frame->params.num_refs > 1) {
      schro_pack_encode_sint (frame->pack,
          (int32_t)(frame->picture_number_ref1 - frame->frame_number));
    }
  }

  /* retire list */
  if (frame->is_ref) {
    schro_pack_encode_uint (frame->pack, frame->n_retire);
    for(i=0;i<frame->n_retire;i++){
      schro_pack_encode_sint (frame->pack,
          (int32_t)(frame->retire[i] - frame->frame_number));
    }
  } else {
    //SCHRO_ASSERT(frame->n_retire == 0);
    if(frame->n_retire != 0) SCHRO_ERROR("frame->n_retire != 0");
  }
}


static void
schro_encoder_encode_transform_parameters (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroPack *pack = frame->pack;

  if (params->num_refs > 0) {
    /* zero residual */
    schro_pack_encode_bit (pack, FALSE);
  }

  /* transform */
  schro_pack_encode_uint (pack, params->wavelet_filter_index);

  /* transform depth */
  schro_pack_encode_uint (pack, params->transform_depth);

  /* spatial partitioning */
  if (!params->is_lowdelay) {
    schro_pack_encode_bit (pack, params->spatial_partition_flag);
    if (params->spatial_partition_flag) {
      schro_pack_encode_bit (pack, params->nondefault_partition_flag);
      if (params->nondefault_partition_flag) {
        int i;

        for(i=0;i<params->transform_depth+1;i++){
          schro_pack_encode_uint (pack, params->horiz_codeblocks[i]);
          schro_pack_encode_uint (pack, params->vert_codeblocks[i]);
        }
      }
      schro_pack_encode_uint (pack, params->codeblock_mode_index);
    }
  } else {
    int encode_quant_matrix;

    schro_pack_encode_uint (pack, params->n_horiz_slices);
    schro_pack_encode_uint (pack, params->n_vert_slices);
    schro_pack_encode_uint (pack, params->slice_bytes_num);
    schro_pack_encode_uint (pack, params->slice_bytes_denom);

    /* FIXME */
    encode_quant_matrix = TRUE;

    schro_pack_encode_bit (pack, encode_quant_matrix);
    if (encode_quant_matrix) {
      int i;
      schro_pack_encode_uint (pack, params->quant_matrix[0]);
      for(i=0;i<params->transform_depth;i++){
        schro_pack_encode_uint (pack, params->quant_matrix[1+3*i]);
        schro_pack_encode_uint (pack, params->quant_matrix[2+3*i]);
        schro_pack_encode_uint (pack, params->quant_matrix[3+3*i]);
      }
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

  for(component=0;component<3;component++) {
    for (i=0;i < 1 + 3*params->transform_depth; i++) {
      schro_pack_sync (frame->pack);
      if (params->is_noarith) {
        schro_encoder_encode_subband_noarith (frame, component, i);
      } else {
        schro_encoder_encode_subband (frame, component, i);
      }
    }
  }
}

static int
schro_encoder_quantise_subband (SchroEncoderFrame *frame, int component, int index,
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

        q = schro_quantise(line[i] - pred_value, quant_factor, quant_offset);
        line[i] = schro_dequantise(q, quant_factor, quant_offset) +
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

      schro_quantise_s16 (quant_data + j*width, line, quant_factor,
          quant_offset, width);

      /* FIXME do this in a better way */
      for(i=0;i<width;i++){
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
  subband_zero_flag = schro_encoder_quantise_subband (frame, component,
      index, quant_data);

  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_pack_encode_uint (frame->pack, 0);
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

  schro_dump(SCHRO_DUMP_SUBBAND_EST, "%d %d %d %g %d %g\n",
      frame->frame_number, component, index,
      frame->est_entropy[component][index][frame->quant_index[component][index]],
      arith->offset*8, frame->subband_info[component][index]);

  schro_pack_encode_uint (frame->pack, arith->offset);
  if (arith->offset > 0) {
    schro_pack_encode_uint (frame->pack,
        frame->quant_index[component][index]);

    schro_pack_sync (frame->pack);

    schro_pack_append (frame->pack, arith->buffer->data, arith->offset);
  }
  schro_arith_free (arith);
}

static int
check_block_zero (int16_t *quant_data, int width, int xmin, int xmax,
    int ymin, int ymax)
{
  int i,j;
  for(j=ymin;j<ymax;j++){
    for(i=xmin;i<xmax;i++){
      if (quant_data[j*width + i] != 0) {
        return 0;
      }
    }
  }
  return 1;
}

void
schro_encoder_encode_subband_noarith (SchroEncoderFrame *frame,
    int component, int index)
{
  SchroParams *params = &frame->params;
  SchroPack b;
  SchroPack *pack = &b;
  int16_t *data;
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

  quant_data = frame->quant_data;
  subband_zero_flag = schro_encoder_quantise_subband (frame, component,
      index, quant_data);

  if (subband_zero_flag) {
    SCHRO_DEBUG ("subband is zero");
    schro_pack_encode_uint (frame->pack, 0);
    return;
  }

  schro_pack_encode_init (pack, frame->subband_buffer);

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
    int zero_codeblock = check_block_zero (quant_data, width, xmin, xmax,
        ymin, ymax);
    schro_pack_encode_bit (pack, zero_codeblock);
    if (zero_codeblock) {
      continue;
    }
  }

  if (have_quant_offset) {
    schro_pack_encode_sint (pack, 0);
  }

  for(j=ymin;j<ymax;j++){
    for(i=xmin;i<xmax;i++){
      schro_pack_encode_sint (pack, quant_data[j*width + i]);
    }
  }
    }
  }
  schro_pack_flush (pack);

  SCHRO_ASSERT(schro_pack_get_offset(pack) < frame->subband_size);

  schro_dump(SCHRO_DUMP_SUBBAND_EST, "%d %d %d %d %d\n",
      frame->frame_number, component, index,
      frame->estimated_entropy, schro_pack_get_offset(pack)*8);

  schro_pack_encode_uint (frame->pack, schro_pack_get_offset(pack));
  if (schro_pack_get_offset(pack) > 0) {
    schro_pack_encode_uint (frame->pack,
        frame->quant_index[component][index]);

    schro_pack_sync (frame->pack);
    schro_pack_append (frame->pack, pack->buffer->data,
        schro_pack_get_offset(pack));
  }

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
      SCHRO_LIMIT_TRANSFORM_DEPTH + encoder->video_format.chroma_h_shift);
  frame_height = ROUND_UP_POW2(encoder->video_format.height,
      SCHRO_LIMIT_TRANSFORM_DEPTH + encoder->video_format.chroma_v_shift);

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

  encoder_frame->inserted_buffers =
    schro_list_new_full ((SchroListFreeFunc)schro_buffer_unref, NULL);

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
    if (frame->filtered_frame) {
      schro_frame_unref (frame->filtered_frame);
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
    if (frame->motion) {
      schro_motion_free (frame->motion);
    }

    if (frame->tmpbuf) free (frame->tmpbuf);
    if (frame->tmpbuf2) free (frame->tmpbuf2);
    schro_list_free (frame->inserted_buffers);

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
static const int pref_range[][2] = {
  { 0, 7 }, /* engine */
  { 0, 4 }, /* quant engine */
  { 2, 20 }, /* ref distance */
  { 1, SCHRO_LIMIT_ENCODER_TRANSFORM_DEPTH }, /* transform depth */
  { 0, 7 }, /* intra wavelet */
  { 0, 7 }, /* inter wavelet */
  { 0, 100 }, /* lambda */
  { 0, 100 }, /* psnr */
  { 0, 1000000000 }, /* bitrate */
  { 0, 1 }, /* noarith */
  { 0, 1 }, /* md5 */
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

  SCHRO_DEBUG("setting encoder preference %d to %d", pref, value);

  encoder->prefs[pref] = value;

  return value;
}
#endif

/* settings */

#define ENUM(name,list,def) \
  { name , SCHRO_ENCODER_SETTING_TYPE_ENUM, 0, ARRAY_SIZE(list), def, list }
#define INT(name,min,max,def) \
  { name , SCHRO_ENCODER_SETTING_TYPE_INT, min, max, def }
#define BOOL(name,def) \
  { name , SCHRO_ENCODER_SETTING_TYPE_BOOLEAN, 0, 1, def }
#define DOUB(name,min,max,def) \
  { name , SCHRO_ENCODER_SETTING_TYPE_DOUBLE, min, max, def }

static char *rate_control_list[] = {
  "constant_noise_threshold",
  "constant_bitrate",
  "low_delay",
  "lossless"
};
static char *gop_structure_list[] = {
  "adaptive",
  "intra_only",
  "backref",
  "chained_backref",
  "biref",
  "chained_biref"
};
static char *perceptual_weighting_list[] = {
  "none",
  "ccir959",
  "moo"
};
static char *filtering_list[] = {
  "none",
  "center_weighted_median",
  "gaussian",
  "add_noise",
  "adaptive_gaussian"
};
static char *wavelet_list[] = {
  "desl_debuc_9_7",
  "le_gall_5_3",
  "desl_debuc_13_7",
  "haar_0",
  "haar_1",
  "fidelity",
  "daub_9_7"
};

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

static SchroEncoderSetting encoder_settings[] = {
  ENUM("rate_control", rate_control_list, 0),
  INT ("bitrate", 0, INT_MAX, 13824000),
  INT ("max_bitrate", 0, INT_MAX, 13824000),
  INT ("min_bitrate", 0, INT_MAX, 13824000),
  DOUB("noise_threshold", 0, 100.0, 25.0),
  ENUM("gop_structure", gop_structure_list, 0),
  ENUM("perceptual_weighting", perceptual_weighting_list, 0),
  DOUB("perceptual_distance", 0, 100.0, 3.0),
  ENUM("filtering", filtering_list, 0),
  DOUB("filter_value", 0, 100.0, 5.0),
  INT ("profile", 0, 0, 0),
  INT ("level", 0, 0, 0),
  INT ("au_distance", 1, INT_MAX, 30),
  BOOL("enable_psnr", FALSE),
  BOOL("enable_ssim", FALSE),

  INT ("ref_distance", 2, 20, 4),
  INT ("transform_depth", 1, SCHRO_LIMIT_ENCODER_TRANSFORM_DEPTH, 4),
  ENUM("intra_wavelet", wavelet_list, SCHRO_WAVELET_DESLAURIES_DUBUC_9_7),
  ENUM("inter_wavelet", wavelet_list, SCHRO_WAVELET_LE_GALL_5_3),
  INT ("mv_precision", 0, 3, 0),
  BOOL("interlaced_coding", FALSE),
  BOOL("enable_internal_testing", FALSE),
  BOOL("enable_noarith", FALSE),
  BOOL("enable_md5", FALSE),
  BOOL("enable_fullscan_prediction", FALSE),
  BOOL("enable_hierarchical_prediction", TRUE),
  BOOL("enable_zero_prediction", FALSE),
  BOOL("enable_phasecorr_prediction", FALSE),
  INT ("magic_dc_metric_offset", 0, 255, 1.0),
};

int
schro_encoder_get_n_settings (void)
{
  return ARRAY_SIZE(encoder_settings);
}

const SchroEncoderSetting *
schro_encoder_get_setting_info (int i)
{
  if (i>=0 && i < ARRAY_SIZE(encoder_settings)) {
    return encoder_settings+i;
  }
  return NULL;
}

#define VAR_SET(x) if (strcmp (name, #x) == 0) { \
  encoder->x = value; \
  return; \
}
#define VAR_GET(x) if (strcmp (name, #x) == 0) { \
  return encoder->x; \
}

void
schro_encoder_setting_set_double (SchroEncoder *encoder, const char *name,
    double value)
{
  VAR_SET(rate_control);
  VAR_SET(bitrate);
  VAR_SET(max_bitrate);
  VAR_SET(min_bitrate);
  VAR_SET(noise_threshold);
  VAR_SET(gop_structure);
  VAR_SET(perceptual_weighting);
  VAR_SET(perceptual_distance);
  VAR_SET(filtering);
  VAR_SET(filter_value);
  VAR_SET(interlaced_coding);
  VAR_SET(profile);
  VAR_SET(level);
  VAR_SET(au_distance);
  VAR_SET(ref_distance);
  VAR_SET(transform_depth);
  VAR_SET(intra_wavelet);
  VAR_SET(inter_wavelet);
  VAR_SET(mv_precision);
  VAR_SET(enable_psnr);
  VAR_SET(enable_ssim);
  VAR_SET(enable_internal_testing);
  VAR_SET(enable_noarith);
  VAR_SET(enable_md5);
  VAR_SET(enable_fullscan_prediction);
  VAR_SET(enable_hierarchical_prediction);
  VAR_SET(enable_zero_prediction);
  VAR_SET(enable_phasecorr_prediction);
  VAR_SET(magic_dc_metric_offset);
  //VAR_SET();
}

double
schro_encoder_setting_get_double (SchroEncoder *encoder, const char *name)
{
  VAR_GET(rate_control);
  VAR_GET(bitrate);
  VAR_GET(max_bitrate);
  VAR_GET(min_bitrate);
  VAR_GET(noise_threshold);
  VAR_GET(gop_structure);
  VAR_GET(perceptual_weighting);
  VAR_GET(perceptual_distance);
  VAR_GET(filtering);
  VAR_GET(filter_value);
  VAR_GET(interlaced_coding);
  VAR_GET(profile);
  VAR_GET(level);
  VAR_GET(au_distance);
  VAR_GET(ref_distance);
  VAR_GET(transform_depth);
  VAR_GET(intra_wavelet);
  VAR_GET(inter_wavelet);
  VAR_GET(mv_precision);
  VAR_GET(enable_psnr);
  VAR_GET(enable_ssim);
  VAR_GET(enable_internal_testing);
  VAR_GET(enable_noarith);
  VAR_GET(enable_md5);
  VAR_GET(enable_fullscan_prediction);
  VAR_GET(enable_hierarchical_prediction);
  VAR_GET(enable_zero_prediction);
  VAR_GET(enable_phasecorr_prediction);
  VAR_GET(magic_dc_metric_offset);
  //VAR_GET();

  return 0;
}

