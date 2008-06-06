
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

void schro_encoder_render_picture (SchroEncoderFrame *frame);
static void schro_encoder_encode_picture_prediction_parameters (SchroEncoderFrame *frame);
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
static int schro_encoder_encode_padding (SchroEncoder *encoder, int n);
static void schro_encoder_clean_up_transform_subband (SchroEncoderFrame *frame,
    int component, int index);
static void schro_encoder_fixup_offsets (SchroEncoder *encoder,
    SchroBuffer *buffer);
static void schro_encoder_frame_complete (SchroEncoderFrame *frame);
static int schro_encoder_async_schedule (SchroEncoder *encoder, SchroExecDomain exec_domain);
static void schro_encoder_init_perceptual_weighting (SchroEncoder *encoder);

SchroEncoder *
schro_encoder_new (void)
{
  SchroEncoder *encoder;

  encoder = schro_malloc0 (sizeof(SchroEncoder));

  encoder->version_major = 2;
  encoder->version_minor = 1;

  encoder->au_frame = -1;

  encoder->intra_ref = -1;
  encoder->last_ref = -1;
  encoder->last_ref2 = -1;

  encoder->rate_control = 0;
  encoder->bitrate = 13824000;
  encoder->max_bitrate = 13824000;
  encoder->min_bitrate = 13824000;
  encoder->buffer_size = 0;
  encoder->buffer_level = 0;
  encoder->noise_threshold = 25.0;
  encoder->gop_structure = 0;
  encoder->queue_depth = 20;
  encoder->perceptual_weighting = 0;
  encoder->perceptual_distance = 4.0;
  encoder->filtering = 0;
  encoder->filter_value = 5.0;
  encoder->profile = 0;
  encoder->level = 0;
  encoder->au_distance = 30;
  encoder->enable_psnr = TRUE;
  encoder->enable_ssim = FALSE;
  encoder->enable_md5 = FALSE;

  encoder->ref_distance = 4;
  encoder->transform_depth = 4;
  encoder->intra_wavelet = SCHRO_WAVELET_DESLAURIES_DUBUC_9_7;
  encoder->inter_wavelet = SCHRO_WAVELET_LE_GALL_5_3;
  encoder->mv_precision = 0;
  encoder->motion_block_size = 0;
  encoder->motion_block_overlap = 0;
  encoder->interlaced_coding = FALSE;
  encoder->enable_internal_testing = FALSE;
  encoder->enable_noarith = FALSE;
  encoder->enable_fullscan_estimation = FALSE;
  encoder->enable_hierarchical_estimation = FALSE;
  encoder->enable_zero_estimation = FALSE;
  encoder->enable_phasecorr_estimation = FALSE;
  encoder->enable_bigblock_estimation = TRUE;
  encoder->horiz_slices = 8;
  encoder->vert_slices = 6;

  encoder->magic_dc_metric_offset = 1.0;
  encoder->magic_subband0_lambda_scale = 2.0;
  encoder->magic_chroma_lambda_scale = 1.0;
  encoder->magic_nonref_lambda_scale = 0.5;
  encoder->magic_allocation_scale = 1.1;
  encoder->magic_keyframe_weight = 7.5;
  encoder->magic_scene_change_threshold = 0.2;
  encoder->magic_inter_p_weight = 1.5;
  encoder->magic_inter_b_weight = 0.2;
  encoder->magic_mc_bailout_limit = 0.5;
  encoder->magic_bailout_weight = 4.0;
  encoder->magic_error_power = 2.0;
  encoder->magic_mc_lambda = 0.5;
  encoder->magic_subgroup_length = 4;
  encoder->magic_lambda = 1.0;
  encoder->magic_badblock_multiplier_nonref = 4.0;
  encoder->magic_badblock_multiplier_ref = 8.0;

  schro_video_format_set_std_video_format (&encoder->video_format,
      SCHRO_VIDEO_FORMAT_CUSTOM);

  encoder->frame_queue = schro_queue_new (encoder->queue_depth,
      (SchroQueueFreeFunc)schro_encoder_frame_unref);

  encoder->inserted_buffers =
    schro_list_new_full ((SchroListFreeFunc)schro_buffer_unref, NULL);

  encoder->average_arith_context_ratio_intra = 1.0;
  encoder->average_arith_context_ratio_inter = 1.0;

  return encoder;
}

static void
handle_gop_enum (SchroEncoder *encoder)
{
  switch (encoder->gop_structure) {
    case SCHRO_ENCODER_GOP_BACKREF:
    case SCHRO_ENCODER_GOP_CHAINED_BACKREF:
      SCHRO_DEBUG("Setting backref\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_backref;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_backref;
      break;
    case SCHRO_ENCODER_GOP_INTRA_ONLY:
      SCHRO_DEBUG("Setting intra only\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_intra_only;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_intra_only;
      break;
    case SCHRO_ENCODER_GOP_ADAPTIVE:
    case SCHRO_ENCODER_GOP_BIREF:
    case SCHRO_ENCODER_GOP_CHAINED_BIREF:
      SCHRO_DEBUG("Setting tworef engine\n");
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_tworef;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_tworef;
      break;
  }
}

void
schro_encoder_start (SchroEncoder *encoder)
{
  encoder->engine_init = 1;

  if (encoder->video_format.luma_excursion >= 256 ||
      encoder->video_format.chroma_excursion >= 256) {
    SCHRO_ERROR("luma or chroma excursion is too large for 8 bit");
  }

  schro_encoder_encode_codec_comment (encoder);

  schro_encoder_init_perceptual_weighting (encoder);

  schro_encoder_init_error_tables (encoder);

  encoder->async = schro_async_new (0,
      (SchroAsyncScheduleFunc)schro_encoder_async_schedule,
      (SchroAsyncCompleteFunc)schro_encoder_frame_complete,
      encoder);

  switch (encoder->rate_control) {
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_NOISE_THRESHOLD:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_SIMPLE;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_RATE_DISTORTION;

      if (encoder->buffer_size == 0) {
        encoder->buffer_size = 3 * encoder->bitrate;
      }
      if (encoder->buffer_level == 0) {
        encoder->buffer_level = encoder->buffer_size;
      }
      encoder->bits_per_picture = muldiv64 (encoder->bitrate,
            encoder->video_format.frame_rate_denominator,
            encoder->video_format.frame_rate_numerator);
      if (encoder->video_format.interlaced_coding) {
        encoder->bits_per_picture /= 2;
      }

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOW_DELAY:
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_lowdelay;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_lowdelay;

      schro_encoder_encode_bitrate_comment (encoder, encoder->bitrate);
      break;
    case SCHRO_ENCODER_RATE_CONTROL_LOSSLESS:
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOSSLESS;
      encoder->init_frame = schro_encoder_init_frame;
      encoder->handle_gop = schro_encoder_handle_gop_lossless;
      encoder->handle_quants = schro_encoder_handle_quants;
      encoder->setup_frame = schro_encoder_setup_frame_lossless;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_LAMBDA:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_CONSTANT_LAMBDA;
      break;
    case SCHRO_ENCODER_RATE_CONTROL_CONSTANT_ERROR:
      handle_gop_enum (encoder);
      encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_CONSTANT_ERROR;
      break;
  }

  encoder->start_time = schro_utils_get_time ();
}

void
schro_encoder_free (SchroEncoder *encoder)
{
  int i;

  if (encoder->async) {
    schro_async_free(encoder->async);
  }

  for(i=0;i<SCHRO_LIMIT_REFERENCE_FRAMES;i++){
    if (encoder->reference_pictures[i]) {
      schro_encoder_frame_unref (encoder->reference_pictures[i]);
    }
  }
  schro_queue_free (encoder->frame_queue);

  schro_list_free (encoder->inserted_buffers);

  schro_free (encoder);
}

static void
schro_encoder_init_perceptual_weighting (SchroEncoder *encoder)
{
  encoder->cycles_per_degree_vert = 0.5 * encoder->video_format.height/
    (2.0*atan(0.5/encoder->perceptual_distance)*180/M_PI);
  encoder->cycles_per_degree_horiz = encoder->cycles_per_degree_vert *
    encoder->video_format.aspect_ratio_denominator /
    encoder->video_format.aspect_ratio_numerator;

  if (encoder->video_format.interlaced_coding) {
    encoder->cycles_per_degree_vert *= 0.5;
  }

  SCHRO_DEBUG("cycles per degree horiz=%g vert=%g",
      encoder->cycles_per_degree_horiz, encoder->cycles_per_degree_vert);

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
    case SCHRO_ENCODER_PERCEPTUAL_MANOS_SAKRISON:
      schro_encoder_calculate_subband_weights (encoder,
          schro_encoder_perceptual_weight_manos_sakrison);
      break;
  }
}

SchroVideoFormat *
schro_encoder_get_video_format (SchroEncoder *encoder)
{
  SchroVideoFormat *format;

  format = malloc (sizeof(SchroVideoFormat));
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

static int
schro_encoder_push_is_ready_locked (SchroEncoder *encoder)
{
  int n;

  if (encoder->end_of_stream){
    return FALSE;
  }
  
  n = schro_queue_slots_available (encoder->frame_queue);

  if (encoder->video_format.interlaced_coding) {
    return (n >= 2);
  } else {
    return (n >= 1);
  }
}

int
schro_encoder_push_ready (SchroEncoder *encoder)
{
  int ret;

  schro_async_lock (encoder->async);
  ret = schro_encoder_push_is_ready_locked (encoder);
  schro_async_unlock (encoder->async);

  return ret;
}

void
schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame)
{
  if (encoder->video_format.interlaced_coding == 0) {
    SchroEncoderFrame *encoder_frame;
    SchroFrameFormat format;

    encoder_frame = schro_encoder_frame_new(encoder);
    encoder_frame->encoder = encoder;

    format = schro_params_get_frame_format (8, encoder->video_format.chroma_format);
    if (format == frame->format) {
      encoder_frame->original_frame = frame;
    } else {
      encoder_frame->original_frame = schro_frame_new_and_alloc (NULL, format,
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
  } else {
    SchroEncoderFrame *encoder_frame1;
    SchroEncoderFrame *encoder_frame2;
    SchroFrameFormat format;
    int width, height;

    encoder_frame1 = schro_encoder_frame_new(encoder);
    encoder_frame1->encoder = encoder;
    encoder_frame2 = schro_encoder_frame_new(encoder);
    encoder_frame2->encoder = encoder;

    schro_video_format_get_picture_luma_size (&encoder->video_format,
        &width, &height);
    format = schro_params_get_frame_format (8,
        encoder->video_format.chroma_format);

    encoder_frame1->original_frame = schro_frame_new_and_alloc (NULL, format,
        width, height);
    encoder_frame2->original_frame = schro_frame_new_and_alloc (NULL, format,
        width, height);
    schro_frame_split_fields (encoder_frame1->original_frame,
        encoder_frame2->original_frame, frame);
    schro_frame_unref (frame);

    encoder_frame1->frame_number = encoder->next_frame_number++;
    encoder_frame2->frame_number = encoder->next_frame_number++;

    schro_async_lock (encoder->async);
    if (schro_queue_slots_available (encoder->frame_queue) < 2) {
      SCHRO_ERROR("push when queue full");
      SCHRO_ASSERT(0);
    }
    schro_queue_add (encoder->frame_queue, encoder_frame1,
        encoder_frame1->frame_number);
    schro_queue_add (encoder->frame_queue, encoder_frame2,
        encoder_frame2->frame_number);
    schro_async_signal_scheduler (encoder->async);
    schro_async_unlock (encoder->async);
  }
}

static int
schro_encoder_pull_is_ready_locked (SchroEncoder *encoder)
{
  int i;

  for(i=0;i<encoder->frame_queue->n;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        (frame->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
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
    if (!(frame->state & SCHRO_ENCODER_FRAME_STATE_FREE)) {
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

  schro_async_lock (encoder->async);
  for(i=0;i<encoder->frame_queue->n;i++){
    SchroEncoderFrame *frame;
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->output_slot &&
        (frame->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
      int is_picture = FALSE;

      if (presentation_frame) {
        *presentation_frame = frame->presentation_frame;
      }
      if (frame->access_unit_buffer) {
        buffer = frame->access_unit_buffer;
        frame->access_unit_buffer = NULL;
      } else if (schro_list_get_size(frame->inserted_buffers)>0) {
        buffer = schro_list_remove (frame->inserted_buffers, 0);
      } else if (schro_list_get_size(encoder->inserted_buffers)>0) {
        buffer = schro_list_remove (encoder->inserted_buffers, 0);
      } else {
        double elapsed_time;

        buffer = frame->output_buffer;
        frame->output_buffer = NULL;

        is_picture = TRUE;
        frame->state |= SCHRO_ENCODER_FRAME_STATE_FREE;
        encoder->output_slot++;

        elapsed_time = schro_utils_get_time() - encoder->start_time;

        if (frame->num_refs == 0) {
          frame->badblock_ratio = 0;
          frame->mc_error = 0;
        }

        schro_dump (SCHRO_DUMP_PICTURE, "%d %d %d %d %d %g %d %d %d %d %g %d %g %g %g %g %g %g %g\n",
            frame->frame_number, /* 0 */
            frame->num_refs,
            frame->is_ref,
            frame->allocated_mc_bits,
            frame->allocated_residual_bits,
            frame->picture_weight, /* 5 */
            frame->estimated_mc_bits,
            frame->estimated_residual_bits,
            frame->actual_mc_bits,
            frame->actual_residual_bits,
            frame->scene_change_score, /* 10 */
            encoder->buffer_level,
            frame->base_lambda,
            frame->mc_error,
            frame->mean_squared_error_luma,
            frame->mean_squared_error_chroma, /* 15 */
            elapsed_time,
            frame->badblock_ratio,
            frame->hist_slope);

        /* FIXME move this */
        if (frame->num_refs == 0) {
          double x;
          double alpha = 0.9;

          x = frame->estimated_arith_context_ratio * (double)frame->actual_residual_bits / frame->estimated_residual_bits;
          encoder->average_arith_context_ratio_intra *= alpha;
          encoder->average_arith_context_ratio_intra += (1.0-alpha) * x;
          SCHRO_DEBUG("arith ratio %g", encoder->average_arith_context_ratio_intra);
        } else {
          double x;
          double alpha = 0.9;

          x = frame->estimated_arith_context_ratio * (double)frame->actual_residual_bits / frame->estimated_residual_bits;
          encoder->average_arith_context_ratio_inter *= alpha;
          encoder->average_arith_context_ratio_inter += (1.0-alpha) * x;
          SCHRO_DEBUG("arith ratio %g", encoder->average_arith_context_ratio_inter);
        }

        schro_encoder_shift_frame_queue (encoder);
      }

      if (encoder->rate_control == SCHRO_ENCODER_RATE_CONTROL_CONSTANT_BITRATE) {
        encoder->buffer_level -= buffer->length * 8;
        if (is_picture) {
          if (encoder->buffer_level < 0) {
            SCHRO_ERROR("buffer underrun by %d bytes", -encoder->buffer_level);
            encoder->buffer_level = 0;
          }
          encoder->buffer_level += encoder->bits_per_picture;
          if (encoder->buffer_level > encoder->buffer_size) {
            int n;

            n = (encoder->buffer_level - encoder->buffer_size + 7)/8;
            SCHRO_DEBUG("buffer overrun, adding padding of %d bytes", n);
            n = schro_encoder_encode_padding (encoder, n);
            encoder->buffer_level -= n*8;
          }
          SCHRO_DEBUG("buffer level %d of %d bits", encoder->buffer_level,
              encoder->buffer_size);
        }
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

static int
schro_encoder_encode_padding (SchroEncoder *encoder, int n)
{
  SchroBuffer *buffer;
  SchroPack *pack;

  if (n < SCHRO_PARSE_HEADER_SIZE) n = SCHRO_PARSE_HEADER_SIZE;

  buffer = schro_buffer_new_and_alloc (n);

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_PADDING);
  
  schro_pack_append_zero (pack, n - SCHRO_PARSE_HEADER_SIZE);

  schro_pack_free (pack);

  schro_encoder_insert_buffer (encoder, buffer);

  return n;
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

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_END_OF_SEQUENCE);

  schro_pack_free (pack);

  return buffer;
}

SchroStateEnum
schro_encoder_wait (SchroEncoder *encoder)
{
  SchroStateEnum ret = SCHRO_STATE_AGAIN;

  schro_async_lock (encoder->async);
  while (1) {
    if (schro_encoder_pull_is_ready_locked (encoder)) {
      SCHRO_DEBUG("have buffer");
      ret = SCHRO_STATE_HAVE_BUFFER;
      break;
    }
    if (schro_encoder_push_is_ready_locked (encoder)) {
      SCHRO_DEBUG("need frame");
      ret = SCHRO_STATE_NEED_FRAME;
      break;
    }
    if (schro_queue_is_empty(encoder->frame_queue) && encoder->end_of_stream) {
      ret = SCHRO_STATE_END_OF_STREAM;
      break;
    }
 
    SCHRO_DEBUG("encoder waiting");
    ret = schro_async_wait_locked (encoder->async);
    if (!ret) {
      int i;

      SCHRO_WARNING ("deadlock?  kicking scheduler");
      for(i=0;i<encoder->frame_queue->n;i++){
        SchroEncoderFrame *frame = encoder->frame_queue->elements[i].data;
        SCHRO_WARNING("%d: %d %d %d %d %04x", i, frame->frame_number,
            frame->picture_number_ref[0], frame->picture_number_ref[1],
            frame->busy, frame->state);
      }
      for(i=0;i<SCHRO_LIMIT_REFERENCE_FRAMES;i++){
        SchroEncoderFrame *frame = encoder->reference_pictures[i];
        if (frame) {
          SCHRO_WARNING("ref %d: %d %d %04x", i, frame->frame_number,
              frame->busy, frame->state);
        } else {
          SCHRO_WARNING("ref %d: NULL", i);
        }
      }
      //SCHRO_ASSERT(0);
      schro_async_signal_scheduler (encoder->async);
      ret = SCHRO_STATE_AGAIN;
      break;
    }
  }
  schro_async_unlock (encoder->async);

  return ret;
}

static void
schro_encoder_frame_complete (SchroEncoderFrame *frame)
{
  SCHRO_INFO("completing task, picture %d working %02x in state %02x",
      frame->frame_number, frame->working, frame->state);

  SCHRO_ASSERT(frame->busy == TRUE);

  frame->busy = FALSE;
  frame->state |= frame->working;
  frame->working = 0;

  if ((frame->state & SCHRO_ENCODER_FRAME_STATE_POSTANALYSE)) {
    frame->state |= SCHRO_ENCODER_FRAME_STATE_DONE;

    SCHRO_ASSERT(frame->output_buffer_size > 0);

    if (frame->ref_frame[0]) {
      schro_encoder_frame_unref (frame->ref_frame[0]);
    }
    if (frame->ref_frame[1]) {
      schro_encoder_frame_unref (frame->ref_frame[1]);
    }

    if (frame->start_access_unit) {
      frame->access_unit_buffer = schro_encoder_encode_access_unit (frame->encoder);
    }
    if (frame->last_frame) {
      frame->encoder->completed_eos = TRUE;
    }
  }
}

/**
 * run_stage:
 * @frame:
 * @state:
 *
 * Runs a stage in the encoding process.
 */
static void
run_stage (SchroEncoderFrame *frame, SchroEncoderFrameStateEnum state)
{
  void *func;

  SCHRO_ASSERT(!(frame->state & state));

  frame->busy = TRUE;
  frame->working = state;
  switch (state) {
    case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
      func = schro_encoder_analyse_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_PREDICT:
      func = schro_encoder_predict_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_ENCODING:
      func = schro_encoder_encode_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
      func = schro_encoder_reconstruct_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_POSTANALYSE:
      func = schro_encoder_postanalyse_picture;
      break;
    default:
      SCHRO_ASSERT(0);
  }
  schro_async_run_locked (frame->encoder->async, func, frame);
}

/**
 * check_refs:
 * @frame: encoder frame
 *
 * Checks whether reference pictures are available to be used for motion
 * rendering.
 */
static int
check_refs (SchroEncoderFrame *frame)
{
  if (frame->num_refs == 0) return TRUE;

  if (frame->num_refs > 0 &&
      !(frame->ref_frame[0]->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
    return FALSE;
  }
  if (frame->num_refs > 1 &&
      !(frame->ref_frame[1]->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
    return FALSE;
  }

  return TRUE;
}

static int
schro_encoder_async_schedule (SchroEncoder *encoder, SchroExecDomain exec_domain)
{
  SchroEncoderFrame *frame;
  int i;
  int ref;
  unsigned int todo;

  SCHRO_INFO("iterate %d", encoder->completed_eos);

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("analyse i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    todo = frame->needed_state & (~frame->state);

    if (todo & SCHRO_ENCODER_FRAME_STATE_ANALYSE) {
      encoder->init_frame (frame);
      run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
      return TRUE;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      encoder->handle_gop (encoder, i);
      break;
    }
  }

  /* Reference pictures are higher priority, so we pass over the list
   * first for reference pictures, then for non-ref. */
  for(ref = 1; ref >= 0; ref--){
    for(i=0;i<encoder->frame_queue->n;i++) {
      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

      if (frame->busy) continue;

      if (frame->is_ref != ref) continue;

      todo = frame->needed_state & (~frame->state);

      if (todo & SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS &&
          frame->state & SCHRO_ENCODER_FRAME_STATE_HAVE_GOP) {
        if (encoder->setup_frame (frame)) {
          frame->state |= SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        }
      }
      if (todo & SCHRO_ENCODER_FRAME_STATE_PREDICT &&
          frame->state & SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS) {
        if (!check_refs(frame)) continue;
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      }
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->slot == encoder->quant_slot) {
      int ret;
      ret = encoder->handle_quants (encoder, i);
      if (!ret) break;
    }
  }

  for(ref = 1; ref >= 0; ref--){
    for(i=0;i<encoder->frame_queue->n;i++) {
      frame = encoder->frame_queue->elements[i].data;
      SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

      if (frame->busy) continue;

      todo = frame->needed_state & (~frame->state);

      if (todo & SCHRO_ENCODER_FRAME_STATE_ENCODING &&
          frame->state & SCHRO_ENCODER_FRAME_STATE_HAVE_QUANTS) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      }
      if (todo & SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT &&
          frame->state & SCHRO_ENCODER_FRAME_STATE_ENCODING) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      }
      if (todo & SCHRO_ENCODER_FRAME_STATE_POSTANALYSE &&
          frame->state & SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT) {
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
        return TRUE;
      }
    }
  }

  return FALSE;
}


void
schro_encoder_analyse_picture (SchroEncoderFrame *frame)
{
  if (frame->encoder->filtering != 0) {
    frame->filtered_frame = schro_frame_dup (frame->original_frame);
    switch (frame->encoder->filtering) {
      case 1:
        schro_frame_filter_cwmN (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 2:
        schro_frame_filter_lowpass2 (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 3:
        schro_frame_filter_addnoise (frame->filtered_frame,
            frame->encoder->filter_value);
        break;
      case 4:
        schro_frame_filter_adaptive_lowpass (frame->filtered_frame);
        break;
    }
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

  frame->tmpbuf = schro_malloc(sizeof(int16_t) *
      (frame->encoder->video_format.width + 16));

  if (frame->params.num_refs > 0) {
    schro_encoder_motion_predict (frame);
  }

  schro_encoder_render_picture (frame);
}

void
schro_encoder_render_picture (SchroEncoderFrame *frame)
{
  SCHRO_INFO("render picture %d", frame->frame_number);

  if (frame->params.num_refs > 0) {
    frame->motion->src1 = frame->ref_frame[0]->reconstructed_frame;
    if (frame->params.num_refs > 1) {
      frame->motion->src2 = frame->ref_frame[1]->reconstructed_frame;
    }

    SCHRO_ASSERT(schro_motion_verify (frame->motion));

    if ((frame->encoder->bits_per_picture &&
        frame->estimated_mc_bits > frame->encoder->bits_per_picture * frame->encoder->magic_mc_bailout_limit) ||
        frame->badblock_ratio > 0.5) {
      SCHRO_DEBUG("%d: MC bailout %d > %g", frame->frame_number,
          frame->estimated_mc_bits,
          frame->encoder->bits_per_picture*frame->encoder->magic_mc_bailout_limit);
      frame->picture_weight = frame->encoder->magic_bailout_weight;
      frame->params.num_refs = 0;
      frame->num_refs = 0;
    }
  }

  if (frame->params.num_refs > 0) {
    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);

    schro_motion_render (frame->motion, frame->prediction_frame);

    schro_frame_subtract (frame->iwt_frame, frame->prediction_frame);

    schro_frame_zero_extend (frame->iwt_frame,
        frame->params.video_format->width,
        schro_video_format_get_picture_height(frame->params.video_format));
  } else {
    schro_frame_convert (frame->iwt_frame, frame->filtered_frame);
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
  SchroBuffer *subbuffer;
  int picture_chroma_width, picture_chroma_height;
  int width, height;

  SCHRO_INFO("encode picture %d", frame->frame_number);

  frame->output_buffer = schro_buffer_new_and_alloc (frame->output_buffer_size);

  schro_video_format_get_iwt_alloc_size (&frame->encoder->video_format,
      &width, &height);
  frame->subband_size = width * height / 4;
  frame->subband_buffer = schro_buffer_new_and_alloc (frame->subband_size * sizeof(int16_t));

  schro_video_format_get_picture_chroma_size (&frame->encoder->video_format,
      &picture_chroma_width, &picture_chroma_height);

  frame->quant_data = schro_malloc (sizeof(int16_t) * frame->subband_size);

  frame->pack = schro_pack_new ();
  schro_pack_encode_init (frame->pack, frame->output_buffer);

  /* encode header */
  schro_encoder_encode_parse_info (frame->pack,
      SCHRO_PARSE_CODE_PICTURE(frame->is_ref, frame->params.num_refs,
        frame->params.is_lowdelay, frame->params.is_noarith));
  schro_encoder_encode_picture_header (frame);

  if (frame->params.num_refs > 0) {
    schro_pack_sync(frame->pack);
    schro_encoder_encode_picture_prediction_parameters (frame);
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

  frame->actual_residual_bits = -schro_pack_get_offset (frame->pack)*8;

  schro_pack_sync(frame->pack);
  if (frame->params.is_lowdelay) {
    schro_encoder_encode_lowdelay_transform_data (frame);
  } else {
    schro_encoder_encode_transform_data (frame);
  }

  schro_pack_flush (frame->pack);
  frame->actual_residual_bits += schro_pack_get_offset (frame->pack)*8;

  subbuffer = schro_buffer_new_subbuffer (frame->output_buffer, 0,
      schro_pack_get_offset (frame->pack));
  schro_buffer_unref (frame->output_buffer);
  frame->output_buffer = subbuffer;

  if (frame->subband_buffer) {
    schro_buffer_unref (frame->subband_buffer);
  }
  if (frame->quant_data) {
    schro_free (frame->quant_data);
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
  frame = schro_frame_new_and_alloc (NULL, frame_format,
      encoder_frame->encoder->video_format.width,
      schro_video_format_get_picture_height(&encoder_frame->encoder->video_format));
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
  SchroVideoFormat *video_format = frame->params.video_format;

  if (frame->encoder->enable_psnr) {
    double mse[3];

    schro_frame_mean_squared_error (frame->filtered_frame,
        frame->reconstructed_frame->frames[0], mse);

    frame->mean_squared_error_luma = mse[0] /
      (video_format->luma_excursion*video_format->luma_excursion);
    frame->mean_squared_error_chroma = 0.5 * (mse[1] + mse[2]) /
      (video_format->chroma_excursion*video_format->chroma_excursion);
  }

  if (frame->encoder->enable_ssim) {
    double mssim;

    mssim = schro_frame_ssim (frame->filtered_frame,
        frame->reconstructed_frame->frames[0]);
    schro_dump(SCHRO_DUMP_SSIM, "%d %g\n", frame->frame_number, mssim);
  }
}

static void
schro_encoder_encode_picture_prediction_parameters (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int index;

  /* block parameters */
  index = schro_params_get_block_params (params);
  schro_pack_encode_uint (frame->pack, index);
  if (index == 0) {
    schro_pack_encode_uint (frame->pack, params->xblen_luma);
    schro_pack_encode_uint (frame->pack, params->yblen_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
    schro_pack_encode_uint (frame->pack, params->xbsep_luma);
  }

  MARKER(frame->pack);

  /* mv precision */
  schro_pack_encode_uint (frame->pack, params->mv_precision);

  MARKER(frame->pack);

  /* global motion flag */
  schro_pack_encode_bit (frame->pack, params->have_global_motion);
  if (params->have_global_motion) {
    int i;
    for(i=0;i<params->num_refs;i++){
      SchroGlobalMotion *gm = params->global_motion + i;

      /* pan tilt */
      if (gm->b0 == 0 && gm->b1 == 0) {
        schro_pack_encode_bit (frame->pack, 0);
      } else {
        schro_pack_encode_bit (frame->pack, 1);
        schro_pack_encode_sint (frame->pack, gm->b0);
        schro_pack_encode_sint (frame->pack, gm->b1);
      }

      /* zoom rotate shear */
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

      /* perspective */
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
  SCHRO_ASSERT(params->picture_weight_2 == 1 || params->num_refs == 2);
  if (params->picture_weight_bits == 1 &&
      params->picture_weight_1 == 1 &&
      params->picture_weight_2 == 1) {
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
            x = mv->dx[ref];
            y = mv->dy[ref];

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

  schro_encoder_encode_parse_info (pack, SCHRO_PARSE_CODE_SEQUENCE_HEADER);
  
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
  schro_pack_encode_uint (pack, format->interlaced_coding);

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
  schro_pack_sync(frame->pack);
  schro_pack_encode_bits (frame->pack, 32, frame->frame_number);

  SCHRO_DEBUG("refs %d ref0 %d ref1 %d", frame->params.num_refs,
      frame->picture_number_ref[0], frame->picture_number_ref[1]);

  if (frame->params.num_refs > 0) {
    schro_pack_encode_sint (frame->pack,
        (int32_t)(frame->picture_number_ref[0] - frame->frame_number));
    if (frame->params.num_refs > 1) {
      schro_pack_encode_sint (frame->pack,
          (int32_t)(frame->picture_number_ref[1] - frame->frame_number));
    }
  }

  if (frame->is_ref) {
    if (frame->retired_picture_number != -1) {
      schro_pack_encode_sint (frame->pack,
          (int32_t)(frame->retired_picture_number - frame->frame_number));
    } else {
      schro_pack_encode_sint (frame->pack, 0);
    }
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
    if (schro_params_is_default_codeblock (params)) {
      schro_pack_encode_bit (pack, FALSE);
    } else {
      int i;

      schro_pack_encode_bit (pack, TRUE);
      for(i=0;i<params->transform_depth+1;i++){
        schro_pack_encode_uint (pack, params->horiz_codeblocks[i]);
        schro_pack_encode_uint (pack, params->vert_codeblocks[i]);
      }
      schro_pack_encode_uint (pack, params->codeblock_mode_index);
    }
  } else {
    schro_pack_encode_uint (pack, params->n_horiz_slices);
    schro_pack_encode_uint (pack, params->n_vert_slices);
    schro_pack_encode_uint (pack, params->slice_bytes_num);
    schro_pack_encode_uint (pack, params->slice_bytes_denom);

    if (schro_params_is_default_quant_matrix (params)) {
      schro_pack_encode_bit (pack, FALSE);
    } else {
      int i;
      schro_pack_encode_bit (pack, TRUE);
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
  static const int wavelet_extent[SCHRO_N_WAVELETS] = { 2, 1, 2, 0, 0, 4, 2 };
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
    schro_video_format_get_picture_luma_size (params->video_format, &w, &h);
  } else {
    schro_video_format_get_picture_chroma_size (params->video_format, &w, &h);
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

  if (index == 0) {
    horiz_codeblocks = params->horiz_codeblocks[0];
    vert_codeblocks = params->vert_codeblocks[0];
  } else {
    horiz_codeblocks = params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
    vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
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

  if (index == 0) {
    horiz_codeblocks = params->horiz_codeblocks[0];
    vert_codeblocks = params->vert_codeblocks[0];
  } else {
    horiz_codeblocks = params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
    vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT(position)+1];
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
      frame->estimated_residual_bits, schro_pack_get_offset(pack)*8);

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
  int iwt_width, iwt_height;
  int picture_width;
  int picture_height;

  encoder_frame = schro_malloc0 (sizeof(SchroEncoderFrame));
  encoder_frame->state = SCHRO_ENCODER_FRAME_STATE_NEW;
  encoder_frame->needed_state = 0xfff;
  encoder_frame->refcount = 1;

  frame_format = schro_params_get_frame_format (16,
      encoder->video_format.chroma_format);

  schro_video_format_get_iwt_alloc_size (&encoder->video_format,
      &iwt_width, &iwt_height);
  encoder_frame->iwt_frame = schro_frame_new_and_alloc (NULL, frame_format,
      iwt_width, iwt_height);
  
  schro_video_format_get_picture_luma_size (&encoder->video_format,
      &picture_width, &picture_height);
  encoder_frame->prediction_frame = schro_frame_new_and_alloc (NULL, frame_format,
      picture_width, picture_height);

  encoder_frame->inserted_buffers =
    schro_list_new_full ((SchroListFreeFunc)schro_buffer_unref, NULL);

  encoder_frame->retired_picture_number = -1;

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

    if (frame->tmpbuf) schro_free (frame->tmpbuf);
    schro_list_free (frame->inserted_buffers);

    schro_free (frame);
  }
}

#if 0
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
#endif

SchroEncoderFrame *
schro_encoder_reference_get (SchroEncoder *encoder,
    SchroPictureNumber frame_number)
{
  int i;
  for(i=0;i<SCHRO_LIMIT_REFERENCE_FRAMES;i++){
    if (encoder->reference_pictures[i] && 
        encoder->reference_pictures[i]->frame_number == frame_number) {
      return encoder->reference_pictures[i];
    }
  }
  return NULL;
}

/* settings */

#define ENUM(name,list,def) \
  { name , SCHRO_ENCODER_SETTING_TYPE_ENUM, 0, ARRAY_SIZE(list)-1, def, list }
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
  "lossless",
  "constant_lambda",
  "constant_error"
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
  "moo",
  "manos_sakrison"
};
static char *filtering_list[] = {
  "none",
  "center_weighted_median",
  "gaussian",
  "add_noise",
  "adaptive_gaussian"
};
static char *wavelet_list[] = {
  "desl_dubuc_9_7",
  "le_gall_5_3",
  "desl_dubuc_13_7",
  "haar_0",
  "haar_1",
  "fidelity",
  "daub_9_7"
};
static char *block_size_list[] = {
  "automatic",
  "small",
  "medium",
  "large"
};
static char *block_overlap_list[] = {
  "automatic",
  "none",
  "partial",
  "full"
};

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

static SchroEncoderSetting encoder_settings[] = {
  ENUM("rate_control", rate_control_list, 0),
  INT ("bitrate", 0, INT_MAX, 13824000),
  INT ("max_bitrate", 0, INT_MAX, 13824000),
  INT ("min_bitrate", 0, INT_MAX, 13824000),
  INT ("buffer_size", 0, INT_MAX, 0),
  INT ("buffer_level", 0, INT_MAX, 0),
  DOUB("noise_threshold", 0, 100.0, 25.0),
  ENUM("gop_structure", gop_structure_list, 0),
  INT("queue_depth", 1, SCHRO_LIMIT_FRAME_QUEUE_LENGTH, 20),
  ENUM("perceptual_weighting", perceptual_weighting_list, 0),
  DOUB("perceptual_distance", 0, 100.0, 4.0),
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
  ENUM("motion_block_size", block_size_list, 0),
  ENUM("motion_block_overlap", block_overlap_list, 0),
  BOOL("interlaced_coding", FALSE),
  BOOL("enable_internal_testing", FALSE),
  BOOL("enable_noarith", FALSE),
  BOOL("enable_md5", FALSE),
  BOOL("enable_fullscan_estimation", FALSE),
  BOOL("enable_hierarchical_estimation", TRUE),
  BOOL("enable_zero_estimation", FALSE),
  BOOL("enable_phasecorr_estimation", FALSE),
  BOOL("enable_bigblock_estimation", FALSE),
  INT ("horiz_slices", 1, INT_MAX, 16),
  INT ("vert_slices", 1, INT_MAX, 16),

  DOUB("magic_dc_metric_offset", 0.0, 1000.0, 0.0),
  DOUB("magic_subband0_lambda_scale", 0.0, 1000.0, 0.0),
  DOUB("magic_chroma_lambda_scale", 0.0, 1000.0, 0.0),
  DOUB("magic_nonref_lambda_scale", 0.0, 1000.0, 0.0),
  DOUB("magic_allocation_scale", 0.0, 1000.0, 0.0),
  DOUB("magic_keyframe_weight", 0.0, 1000.0, 0.0),
  DOUB("magic_scene_change_threshold", 0.0, 1000.0, 0.0),
  DOUB("magic_inter_p_weight", 0.0, 1000.0, 0.0),
  DOUB("magic_inter_b_weight", 0.0, 1000.0, 0.0),
  DOUB("magic_mc_bailout_limit", 0.0, 1000.0, 0.0),
  DOUB("magic_bailout_weight", 0.0, 1000.0, 0.0),
  DOUB("magic_error_power", 0.0, 1000.0, 0.0),
  DOUB("magic_mc_lambda", 0.0, 1000.0, 0.0),
  DOUB("magic_subgroup_length", 2.0, 10.0, 4.0),
  DOUB("magic_lambda", 0.0, 1000.0, 1.0),
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
  SCHRO_DEBUG("setting %s to %g", #x, value); \
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
  VAR_SET(buffer_size);
  VAR_SET(buffer_level);
  VAR_SET(noise_threshold);
  VAR_SET(gop_structure);
  VAR_SET(queue_depth);
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
  VAR_SET(motion_block_size);
  VAR_SET(motion_block_overlap);
  VAR_SET(enable_psnr);
  VAR_SET(enable_ssim);
  VAR_SET(enable_internal_testing);
  VAR_SET(enable_noarith);
  VAR_SET(enable_md5);
  VAR_SET(enable_fullscan_estimation);
  VAR_SET(enable_hierarchical_estimation);
  VAR_SET(enable_zero_estimation);
  VAR_SET(enable_phasecorr_estimation);
  VAR_SET(enable_bigblock_estimation);
  VAR_SET(horiz_slices);
  VAR_SET(vert_slices);
  //VAR_SET();
  
  VAR_SET(magic_dc_metric_offset);
  VAR_SET(magic_subband0_lambda_scale);
  VAR_SET(magic_chroma_lambda_scale);
  VAR_SET(magic_nonref_lambda_scale);
  VAR_SET(magic_allocation_scale);
  VAR_SET(magic_keyframe_weight);
  VAR_SET(magic_scene_change_threshold);
  VAR_SET(magic_inter_p_weight);
  VAR_SET(magic_inter_b_weight);
  VAR_SET(magic_mc_bailout_limit);
  VAR_SET(magic_bailout_weight);
  VAR_SET(magic_error_power);
  VAR_SET(magic_mc_lambda);
  VAR_SET(magic_subgroup_length);
  VAR_SET(magic_lambda);
}

double
schro_encoder_setting_get_double (SchroEncoder *encoder, const char *name)
{
  VAR_GET(rate_control);
  VAR_GET(bitrate);
  VAR_GET(max_bitrate);
  VAR_GET(min_bitrate);
  VAR_GET(buffer_size);
  VAR_GET(buffer_level);
  VAR_GET(noise_threshold);
  VAR_GET(gop_structure);
  VAR_GET(queue_depth);
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
  VAR_GET(motion_block_size);
  VAR_GET(motion_block_overlap);
  VAR_GET(enable_psnr);
  VAR_GET(enable_ssim);
  VAR_GET(enable_internal_testing);
  VAR_GET(enable_noarith);
  VAR_GET(enable_md5);
  VAR_GET(enable_fullscan_estimation);
  VAR_GET(enable_hierarchical_estimation);
  VAR_GET(enable_zero_estimation);
  VAR_GET(enable_phasecorr_estimation);
  VAR_GET(enable_bigblock_estimation);
  VAR_GET(horiz_slices);
  VAR_GET(vert_slices);
  //VAR_GET();

  VAR_GET(magic_dc_metric_offset);
  VAR_GET(magic_subband0_lambda_scale);
  VAR_GET(magic_chroma_lambda_scale);
  VAR_GET(magic_nonref_lambda_scale);
  VAR_GET(magic_allocation_scale);
  VAR_GET(magic_keyframe_weight);
  VAR_GET(magic_scene_change_threshold);
  VAR_GET(magic_inter_p_weight);
  VAR_GET(magic_inter_b_weight);
  VAR_GET(magic_mc_bailout_limit);
  VAR_GET(magic_bailout_weight);
  VAR_GET(magic_error_power);
  VAR_GET(magic_mc_lambda);
  VAR_GET(magic_subgroup_length);
  VAR_GET(magic_lambda);

  return 0;
}

