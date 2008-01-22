
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#define SCHRO_ARITH_DEFINE_INLINE
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <schroedinger/schrooil.h>
#include <string.h>
#include <stdio.h>

#if 0
/* Used for testing bitstream */
#define MARKER() do{ \
  SCHRO_ASSERT(schro_unpack_decode_uint(unpack) == 1234567); \
} while(0)
#else
#define MARKER()
#endif

#define SCHRO_SKIP_TIME_CONSTANT 0.1

typedef struct _SchroPictureSubbandContext SchroPictureSubbandContext;

struct _SchroPictureSubbandContext {
  int component;
  int index;
  int position;

  SchroFrameData *frame_data;
  SchroFrameData *parent_frame_data;

  int quant_index;
  int subband_length;
  SchroArith *arith;
  SchroUnpack unpack;
  int vert_codeblocks;
  int horiz_codeblocks;
  int have_zero_flags;
  int have_quant_offset;

  int ymin;
  int ymax;
  int xmin;
  int xmax;

  int quant_factor;
  int quant_offset;
  int16_t *line;
};

enum {
  SCHRO_DECODER_STATE_INIT = (1<<0),
  SCHRO_DECODER_STATE_REFERENCES = (1<<1),
  SCHRO_DECODER_STATE_MOTION_DECODE = (1<<2),
  SCHRO_DECODER_STATE_MOTION_RENDER = (1<<3),
  SCHRO_DECODER_STATE_RESIDUAL_DECODE = (1<<4),
  SCHRO_DECODER_STATE_WAVELET_TRANSFORM = (1<<5),
  SCHRO_DECODER_STATE_COMBINE = (1<<6),
  SCHRO_DECODER_STATE_UPSAMPLE = (1<<7),
  SCHRO_DECODER_STATE_DONE = (1<<8),
};

int _schro_decode_prediction_only;


static void schro_decoder_reference_add (SchroDecoder *decoder,
    SchroPicture *picture);
static SchroPicture * schro_decoder_reference_get (SchroDecoder *decoder,
    SchroPictureNumber frame_number);
static void schro_decoder_reference_retire (SchroDecoder *decoder,
    SchroPictureNumber frame_number);
static void schro_decoder_decode_subband (SchroPicture *picture,
    SchroPictureSubbandContext *ctx);
static int schro_decoder_async_schedule (SchroDecoder *decoder);

static void schro_decoder_error (SchroDecoder *decoder, const char *s);


SchroDecoder *
schro_decoder_new (void)
{
  SchroDecoder *decoder;

  decoder = schro_malloc0 (sizeof(SchroDecoder));

  decoder->skip_value = 1.0;
  decoder->skip_ratio = 1.0;

  decoder->reference_queue = schro_queue_new (SCHRO_LIMIT_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_picture_unref);
  decoder->output_queue = schro_queue_new (SCHRO_LIMIT_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_frame_unref);

  decoder->picture_queue = schro_queue_new (4,
      (SchroQueueFreeFunc)schro_picture_unref);
  decoder->queue_depth = 4;

  decoder->async = schro_async_new (0,
      (SchroAsyncScheduleFunc)schro_decoder_async_schedule, decoder);

  return decoder;
}

void
schro_decoder_free (SchroDecoder *decoder)
{
  if (decoder->async) {
    schro_async_free (decoder->async);
  }

  schro_queue_free (decoder->output_queue);
  schro_queue_free (decoder->reference_queue);
  schro_queue_free (decoder->picture_queue);

  if (decoder->error_message) schro_free (decoder->error_message);

  schro_free (decoder);
}

SchroPicture *
schro_picture_new (SchroDecoder *decoder)
{
  SchroPicture *picture;
  SchroFrameFormat frame_format;
  SchroVideoFormat *video_format = &decoder->video_format;
  int frame_width, frame_height;

  picture = schro_malloc0 (sizeof(SchroPicture));
  picture->refcount = 1;

  picture->decoder = decoder;

  picture->tmpbuf = schro_malloc(SCHRO_LIMIT_WIDTH * 2);
  picture->tmpbuf2 = schro_malloc(SCHRO_LIMIT_WIDTH * 2);

  picture->params.video_format = video_format;

  frame_format = schro_params_get_frame_format (16,
      video_format->chroma_format);
  frame_width = ROUND_UP_POW2(video_format->width,
      SCHRO_LIMIT_TRANSFORM_DEPTH + video_format->chroma_h_shift);
  frame_height = ROUND_UP_POW2(video_format->height,
      SCHRO_LIMIT_TRANSFORM_DEPTH + video_format->chroma_v_shift);

  picture->mc_tmp_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);
  picture->frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);

  frame_format = schro_params_get_frame_format (8,
      video_format->chroma_format);
  picture->planar_output_frame = schro_frame_new_and_alloc (frame_format,
      video_format->width, video_format->height);
  SCHRO_DEBUG("planar output frame %dx%d",
      video_format->width, video_format->height);

  return picture;
}

SchroPicture *
schro_picture_ref (SchroPicture *picture)
{
  picture->refcount++;
  return picture;
}

void
schro_picture_unref (SchroPicture *picture)
{
  SCHRO_ASSERT(picture->refcount > 0);
  picture->refcount--;
  if (picture->refcount == 0) {
    int i;
    int component;

    SCHRO_WARNING("freeing picture %p", picture);
    for(component=0;component<3;component++){
      for(i=0;i<SCHRO_LIMIT_SUBBANDS;i++) {
        if (picture->subband_buffer[component][i]) {
          schro_buffer_unref (picture->subband_buffer[component][i]);
        }
      }
    }
    for(i=0;i<9;i++){
      if (picture->motion_buffers[i]) {
        schro_buffer_unref (picture->motion_buffers[i]);
      }
    }
    if (picture->lowdelay_buffer) schro_buffer_unref (picture->lowdelay_buffer);

    if (picture->frame) schro_frame_unref (picture->frame);
    if (picture->mc_tmp_frame) schro_frame_unref (picture->mc_tmp_frame);
    if (picture->planar_output_frame) schro_frame_unref (picture->planar_output_frame);
    if (picture->output_picture) schro_frame_unref (picture->output_picture);
    if (picture->tmpbuf) schro_free (picture->tmpbuf);
    if (picture->tmpbuf2) schro_free (picture->tmpbuf2);
    if (picture->motion) schro_motion_free (picture->motion);
    if (picture->input_buffer) schro_buffer_unref (picture->input_buffer);
    if (picture->upsampled_frame) schro_upsampled_frame_free (picture->upsampled_frame);
    if (picture->ref0) schro_picture_unref (picture->ref0);
    if (picture->ref1) schro_picture_unref (picture->ref1);

    schro_free (picture);
  }
}

void
schro_decoder_reset (SchroDecoder *decoder)
{
  schro_queue_clear (decoder->picture_queue);
  schro_queue_clear (decoder->reference_queue);
  schro_queue_clear (decoder->output_queue);

  decoder->have_access_unit = FALSE;
  decoder->next_frame_number = 0;
  decoder->have_frame_number = FALSE;

}

SchroVideoFormat *
schro_decoder_get_video_format (SchroDecoder *decoder)
{
  SchroVideoFormat *format;

  format = schro_malloc(sizeof(SchroVideoFormat));
  memcpy (format, &decoder->video_format, sizeof(SchroVideoFormat));

  return format;
}

SchroPictureNumber
schro_decoder_get_picture_number (SchroDecoder *decoder)
{
  return decoder->next_frame_number;
}

void
schro_decoder_add_output_picture (SchroDecoder *decoder, SchroFrame *frame)
{
  schro_queue_add (decoder->output_queue, frame, 0);
}

void
schro_decoder_set_earliest_frame (SchroDecoder *decoder,
    SchroPictureNumber earliest_frame)
{
  decoder->earliest_frame = earliest_frame;
}

void
schro_decoder_set_skip_ratio (SchroDecoder *decoder, double ratio)
{
  if (ratio > 1.0) ratio = 1.0;
  if (ratio < 0.0) ratio = 0.0;
  decoder->skip_ratio = ratio;
}

int
schro_decoder_is_intra (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (SCHRO_PARSE_CODE_NUM_REFS(data[4] == 0)) return 1;

  return 1;
}

int
schro_decoder_is_parse_header (SchroBuffer *buffer)
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
schro_decoder_is_access_unit (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (data[4] == SCHRO_PARSE_CODE_SEQUENCE_HEADER) return 1;

  return 0;
}

int
schro_decoder_is_picture (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE(data[4])) return 1;

  return 0;
}

int
schro_decoder_is_end_sequence (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (data[4] == SCHRO_PARSE_CODE_END_OF_SEQUENCE) return 1;

  return 0;
}

int
schro_decoder_pull_is_ready_locked (SchroDecoder *decoder)
{
  SchroPicture *picture;

  picture = schro_queue_find (decoder->picture_queue,
      decoder->next_frame_number);
  if (picture && picture->state & SCHRO_DECODER_STATE_DONE) {
    return TRUE;
  }
  return FALSE;
}

SchroFrame *
schro_decoder_pull (SchroDecoder *decoder)
{
  SchroPicture *picture;
  SchroFrame *frame;

  SCHRO_DEBUG("searching for frame %d", decoder->next_frame_number);

  schro_async_lock (decoder->async);
  picture = schro_queue_find (decoder->picture_queue, decoder->next_frame_number);
  if (picture) {
    if (picture->state & SCHRO_DECODER_STATE_DONE) {
      schro_queue_remove (decoder->picture_queue, decoder->next_frame_number);
    } else {
      picture = NULL;
    }
  }
  schro_async_unlock (decoder->async);

  if (!picture) {
    return NULL;
  }

  decoder->next_frame_number++;

  frame = schro_frame_ref (picture->output_picture);
  schro_picture_unref (picture);

  return frame;
}

int
schro_decoder_push_ready (SchroDecoder *decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  ret = schro_queue_is_full (decoder->picture_queue);
  schro_async_unlock (decoder->async);

  return (ret == FALSE);
}

static int
schro_decoder_get_status_locked (SchroDecoder *decoder)
{
  if (schro_decoder_pull_is_ready_locked (decoder)) {
    return SCHRO_DECODER_OK;
  }
  if (decoder->have_access_unit &&
      schro_queue_is_empty (decoder->output_queue)) {
    return SCHRO_DECODER_NEED_FRAME;
  }
  if (!schro_queue_is_full (decoder->picture_queue) && !decoder->end_of_stream) {
    return SCHRO_DECODER_NEED_BITS;
  }
  if (decoder->end_of_stream &&
      schro_queue_find (decoder->picture_queue, decoder->next_frame_number) == NULL) {
    return SCHRO_DECODER_EOS;
  }

  return SCHRO_DECODER_WAIT;
}

int
schro_decoder_get_status (SchroDecoder *decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  ret = schro_decoder_get_status_locked (decoder);
  schro_async_unlock (decoder->async);

  return ret;
}

void
schro_decoder_dump (SchroDecoder *decoder)
{
  int i;
  for(i=0;i<decoder->picture_queue->n;i++){
    SchroPicture *picture = decoder->picture_queue->elements[i].data;

    SCHRO_ERROR("%d: %d %d %04x %04x %04x",
        i, picture->picture_number,
        picture->busy,
        picture->state,
        picture->needed_state,
        picture->working);
  }
}

int
schro_decoder_wait (SchroDecoder *decoder)
{
  int ret;

  schro_async_lock (decoder->async);
  while (1) {
    ret = schro_decoder_get_status_locked (decoder);
    if (ret != SCHRO_DECODER_WAIT) {
      break;
    }

    ret = schro_async_wait_locked (decoder->async);
    if (!ret) {
      SCHRO_ERROR("doh!");
      schro_decoder_dump (decoder);
      SCHRO_ASSERT(0);
    }
  }
  schro_async_unlock (decoder->async);

  return ret;
}

int
schro_decoder_push_end_of_stream (SchroDecoder *decoder)
{
  decoder->end_of_stream = TRUE;
  return SCHRO_DECODER_EOS;
}

int
schro_decoder_push (SchroDecoder *decoder, SchroBuffer *buffer)
{
  SCHRO_ASSERT(decoder->input_buffer == NULL);

  decoder->input_buffer = buffer;

  schro_unpack_init_with_data (&decoder->unpack,
      decoder->input_buffer->data,
      decoder->input_buffer->length, 1);
  schro_decoder_decode_parse_header(decoder);

  if (decoder->parse_code == SCHRO_PARSE_CODE_SEQUENCE_HEADER) {
    SCHRO_INFO ("decoding access unit");
    schro_decoder_parse_access_unit(decoder);

    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;

    if (decoder->have_access_unit) {
      return SCHRO_DECODER_OK;
    }
    decoder->have_access_unit = TRUE;
    return SCHRO_DECODER_FIRST_ACCESS_UNIT;
  }

  if (decoder->parse_code == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
    int code;

    code = schro_unpack_decode_bits (&decoder->unpack, 8);

    if (code == SCHRO_AUX_DATA_MD5_CHECKSUM) {
      int i;
      for(i=0;i<16;i++){
        decoder->md5_checksum[i] = schro_unpack_decode_bits (&decoder->unpack, 8);
      }
      decoder->has_md5 = TRUE;
    }

    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    
    return SCHRO_DECODER_OK;
  }

  if (SCHRO_PARSE_CODE_IS_PADDING(decoder->parse_code)) {
    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    return SCHRO_DECODER_OK;
  }

  if (SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE (decoder->parse_code)) {
    SCHRO_ERROR ("decoding end sequence");
    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    decoder->end_of_stream = TRUE;
    return SCHRO_DECODER_EOS;
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE(decoder->parse_code)) {

    if (!decoder->have_access_unit) {
      SCHRO_INFO ("no access unit -- dropping picture");
      schro_buffer_unref (decoder->input_buffer);
      decoder->input_buffer = NULL;
      return SCHRO_DECODER_OK;
    }

    return schro_decoder_iterate_picture (decoder);
  }

  return SCHRO_DECODER_ERROR;
}

int
schro_decoder_iterate_picture (SchroDecoder *decoder)
{
  SchroPicture *picture;
  SchroParams *params;

  picture = schro_picture_new (decoder);
  decoder->picture = picture;
  params = &picture->params;

  picture->input_buffer = decoder->input_buffer;
  decoder->input_buffer = NULL;

  params->num_refs = SCHRO_PARSE_CODE_NUM_REFS(decoder->parse_code);
  params->is_lowdelay = SCHRO_PARSE_CODE_IS_LOW_DELAY(decoder->parse_code);
  params->is_noarith = !SCHRO_PARSE_CODE_USING_AC(decoder->parse_code);
  picture->is_ref = SCHRO_PARSE_CODE_IS_REFERENCE(decoder->parse_code);

  if (decoder->has_md5) {
    picture->has_md5 = TRUE;
    memcpy (picture->md5_checksum, decoder->md5_checksum, 16);
    decoder->has_md5 = FALSE;
  }

  schro_decoder_parse_picture_header(decoder->picture);

  if (!decoder->have_frame_number) {
    if (params->num_refs > 0) {
      SCHRO_ERROR("expected I frame after access unit header");
    }
    decoder->next_frame_number = decoder->picture->picture_number;
    decoder->have_frame_number = TRUE;
    SCHRO_INFO("next frame number after seek %d", decoder->next_frame_number);
  }

  if (picture->is_ref) {
    schro_decoder_reference_retire (decoder,
        decoder->picture->retired_picture_number);
    schro_decoder_reference_add (decoder, picture);
  }
  schro_decoder_parse_picture (picture);

#if 0
  /* FIXME bring back picture skipping! */
  if (!picture->is_ref && decoder->skip_value > decoder->skip_ratio) {

    decoder->skip_value = (1-SCHRO_SKIP_TIME_CONSTANT) * decoder->skip_value;
    SCHRO_INFO("skipping frame %d", picture->picture_number);
    SCHRO_DEBUG("skip value %g ratio %g", decoder->skip_value, decoder->skip_ratio);

    picture->output_picture = schro_frame_new ();
    picture->output_picture->frame_number = decoder->picture->picture_number;

    SCHRO_DEBUG("adding %d to queue (skipped)", picture->picture_number);
    schro_queue_add (decoder->picture_queue, picture,
        picture->picture_number);

    return SCHRO_DECODER_OK;
  }
#endif

  decoder->skip_value = (1-SCHRO_SKIP_TIME_CONSTANT) * decoder->skip_value +
    SCHRO_SKIP_TIME_CONSTANT;
SCHRO_DEBUG("skip value %g ratio %g", decoder->skip_value, decoder->skip_ratio);

  picture->output_picture = schro_queue_pull (decoder->output_queue);
  SCHRO_ASSERT(picture->output_picture);

#if 0
  schro_decoder_decode_picture (picture);
#endif
#if 0
  if (skip) {
    schro_picture_unref (picture);
    decoder->picture = NULL;

    return SCHRO_DECODER_OK;
  }
#endif

  schro_async_lock (decoder->async);
  SCHRO_DEBUG("adding %d to queue", picture->picture_number);
  schro_queue_add (decoder->picture_queue, picture, picture->picture_number);
  schro_async_signal_scheduler (decoder->async);
  schro_async_unlock (decoder->async);

  return SCHRO_DECODER_OK;
}

int
schro_decoder_parse_picture (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;

  if (params->num_refs > 0) {
    SCHRO_DEBUG("inter");

    picture->ref0 = schro_decoder_reference_get (picture->decoder, picture->reference1);
    if (picture->ref0 == NULL) {
      SCHRO_ERROR("Could not find reference picture %d", picture->reference1);
    }
    schro_picture_ref (picture->ref0);

    picture->ref1 = NULL;
    if (params->num_refs > 1) {
      picture->ref1 = schro_decoder_reference_get (picture->decoder, picture->reference2);
      if (picture->ref1 == NULL) {
        SCHRO_ERROR("Could not find reference picture %d", picture->reference2);
      }
      schro_picture_ref (picture->ref1);
    }

    schro_unpack_byte_sync (unpack);
    schro_decoder_parse_picture_prediction_parameters (picture);

    schro_params_calculate_mc_sizes (params);

    schro_unpack_byte_sync (unpack);
    schro_decoder_parse_block_data (picture);
  }

  schro_unpack_byte_sync (unpack);
  picture->zero_residual = FALSE;
  if (params->num_refs > 0) {
    picture->zero_residual = schro_unpack_decode_bit (unpack);

    SCHRO_DEBUG ("zero residual %d", picture->zero_residual);
  }

  if (!picture->zero_residual) {
    schro_decoder_parse_transform_parameters (picture);
    schro_params_calculate_iwt_sizes (params);

    schro_unpack_byte_sync (unpack);
    if (params->is_lowdelay) {
      schro_decoder_parse_lowdelay_transform_data (picture);
    } else {
      schro_decoder_parse_transform_data (picture);
      schro_decoder_init_subband_frame_data_interleaved (picture);
    }
  }

  picture->needed_state |= SCHRO_DECODER_STATE_REFERENCES;
  picture->needed_state |= SCHRO_DECODER_STATE_MOTION_DECODE;
  picture->needed_state |= SCHRO_DECODER_STATE_MOTION_RENDER;
  picture->needed_state |= SCHRO_DECODER_STATE_RESIDUAL_DECODE;
  picture->needed_state |= SCHRO_DECODER_STATE_WAVELET_TRANSFORM;
  picture->needed_state |= SCHRO_DECODER_STATE_COMBINE;

  return TRUE;
}

void
schro_decoder_picture_complete (SchroPicture *picture)
{
  SCHRO_DEBUG ("picture complete");

  picture->state |= picture->working;
  picture->working = 0;
  if ((picture->needed_state & (~picture->state)) == 0) {
    picture->state |= SCHRO_DECODER_STATE_DONE;
  }
  picture->busy = FALSE;

}

static int 
schro_decoder_async_schedule (SchroDecoder *decoder)
{
  int i;

  SCHRO_DEBUG("schedule");

  while (schro_async_get_num_completed (decoder->async) > 0) {
    SchroPicture *picture;

    picture = schro_async_pull_locked (decoder->async);
    SCHRO_ASSERT(picture != NULL);

    schro_decoder_picture_complete (picture);
  }

  for(i=0;i<decoder->picture_queue->n;i++){
    SchroPicture *picture = decoder->picture_queue->elements[i].data;
    unsigned int todo;
    void *func = NULL;

    if (picture->busy) continue;

    todo = picture->needed_state & (~picture->state);
    SCHRO_DEBUG("picture %d todo %04x", picture->picture_number, todo);

    if (todo & SCHRO_DECODER_STATE_REFERENCES) {
      int j;
      int refs_ready = TRUE;

      for(j=0;j<picture->params.num_refs;j++){
        SchroPicture *refpic;

        refpic = (j==0) ? picture->ref0 : picture->ref1;

        if (refpic->busy || !(refpic->state & SCHRO_DECODER_STATE_DONE)) {
          refs_ready = FALSE;
          continue;
        }

        if (picture->params.mv_precision > 0 &&
            !(refpic->state & SCHRO_DECODER_STATE_UPSAMPLE)) {
          refpic->working = SCHRO_DECODER_STATE_UPSAMPLE;
          refpic->busy = TRUE;

          func = schro_decoder_x_upsample;
          schro_async_run_locked (decoder->async, func, refpic);

          return TRUE;
        }
      }
      if (refs_ready) {
        picture->state |= SCHRO_DECODER_STATE_REFERENCES;
      }
    }

    if (todo & SCHRO_DECODER_STATE_MOTION_DECODE &&
        picture->state & SCHRO_DECODER_STATE_REFERENCES) {
      func = schro_decoder_x_decode_motion;
      picture->working = SCHRO_DECODER_STATE_MOTION_DECODE;
    } else if (todo & SCHRO_DECODER_STATE_MOTION_RENDER &&
        picture->state & SCHRO_DECODER_STATE_MOTION_DECODE) {
      func = schro_decoder_x_render_motion;
      picture->working = SCHRO_DECODER_STATE_MOTION_RENDER;
    } else if (todo & SCHRO_DECODER_STATE_RESIDUAL_DECODE) {
      func = schro_decoder_x_decode_residual;
      picture->working = SCHRO_DECODER_STATE_RESIDUAL_DECODE;
    } else if (todo & SCHRO_DECODER_STATE_WAVELET_TRANSFORM &&
        picture->state & SCHRO_DECODER_STATE_RESIDUAL_DECODE) {
      func = schro_decoder_x_wavelet_transform;
      picture->working = SCHRO_DECODER_STATE_WAVELET_TRANSFORM;
    } else if (todo & SCHRO_DECODER_STATE_COMBINE &&
        picture->state & SCHRO_DECODER_STATE_WAVELET_TRANSFORM &&
        picture->state & SCHRO_DECODER_STATE_MOTION_RENDER) {
      func = schro_decoder_x_combine;
      picture->working = SCHRO_DECODER_STATE_COMBINE;
    }

    if (func) {
      picture->busy = TRUE;

      schro_async_run_locked (decoder->async, func, picture);

      return TRUE;
    }
  }

  return FALSE;
}

void
schro_decoder_decode_picture (SchroPicture *picture)
{
  schro_decoder_x_check_references (picture);
  schro_decoder_x_decode_motion (picture);
  schro_decoder_x_render_motion (picture);
  schro_decoder_x_decode_residual (picture);
  schro_decoder_x_wavelet_transform (picture);
  schro_decoder_x_combine (picture);
  schro_decoder_x_upsample (picture);
}

void
schro_decoder_x_check_references (SchroPicture *picture)
{
  //SchroParams *params = &picture->params;
  //SchroDecoder *decoder = picture->decoder;

}

void
schro_decoder_x_decode_motion (SchroPicture *picture)
{
  SchroParams *params = &picture->params;

  if (params->num_refs > 0) {
    picture->motion = schro_motion_new (params, picture->ref0->upsampled_frame,
        picture->ref1 ?  picture->ref1->upsampled_frame : NULL);
    schro_decoder_decode_block_data (picture);
  }
}

void
schro_decoder_x_render_motion (SchroPicture *picture)
{
  SchroParams *params = &picture->params;

  if (params->num_refs > 0) {
    SCHRO_WARNING("motion render with %p and %p", picture->ref0, picture->ref1);
    schro_motion_render (picture->motion, picture->mc_tmp_frame);
    SCHRO_WARNING("render done with %p and %p", picture->ref0, picture->ref1);
  }
}

void
schro_decoder_x_decode_residual (SchroPicture *picture)
{
  SchroParams *params = &picture->params;

  if (!picture->zero_residual) {
    if (params->is_lowdelay) {
      schro_decoder_decode_lowdelay_transform_data (picture);
    } else {
      schro_decoder_decode_transform_data (picture);
    }
  }
}

void
schro_decoder_x_wavelet_transform (SchroPicture *picture)
{
  if (!picture->zero_residual) {
    schro_frame_inverse_iwt_transform (picture->frame, &picture->params,
        picture->tmpbuf);
  }
}

void
schro_decoder_x_combine (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroDecoder *decoder = picture->decoder;
  SchroFrame *combined_frame;
  SchroFrame *output_frame;

  if (picture->zero_residual) {
    combined_frame = picture->mc_tmp_frame;
  } else {
    if (params->num_refs > 0) {
      schro_frame_add (picture->frame, picture->mc_tmp_frame);
    }
    combined_frame = picture->frame;
  }

  if (_schro_decode_prediction_only) {
    if (params->num_refs > 0 && !picture->is_ref) {
      output_frame = picture->mc_tmp_frame;
    } else {
      output_frame = combined_frame;
    }
  } else {
    output_frame = combined_frame;
  }

  if (SCHRO_FRAME_IS_PACKED(picture->output_picture->format)) {
    schro_frame_convert (picture->planar_output_frame, output_frame);
    schro_frame_convert (picture->output_picture, picture->planar_output_frame);
  } else {
    schro_frame_convert (picture->output_picture, output_frame);
  }

  if (picture->is_ref) {
    SchroFrame *ref;
    SchroFrameFormat frame_format;

    frame_format = schro_params_get_frame_format (8,
        params->video_format->chroma_format);
    
    ref = schro_frame_new_and_alloc (frame_format,
        decoder->video_format.width, decoder->video_format.height);
    schro_frame_convert (ref, combined_frame);
    picture->upsampled_frame = schro_upsampled_frame_new (ref);
  }

  if (picture->has_md5) {
    uint32_t state[4];

    schro_frame_md5 (picture->output_picture, state);
    if (memcmp (state, picture->md5_checksum, 16) != 0) {
      char a[65];
      char b[65];
      int i;

      for(i=0;i<16;i++){
        sprintf(a+2*i, "%02x", ((uint8_t *)state)[i]);
        sprintf(b+2*i, "%02x", picture->md5_checksum[i]);
      }
      SCHRO_ERROR("MD5 checksum mismatch (%s should be %s)", a, b);
    }
  }
}

void
schro_decoder_x_upsample (SchroPicture *picture)
{
  /* FIXME */
}

void
schro_decoder_decode_parse_header (SchroDecoder *decoder)
{
  SchroUnpack *unpack = &decoder->unpack;
  int v1, v2, v3, v4;
  
  v1 = schro_unpack_decode_bits (unpack, 8);
  v2 = schro_unpack_decode_bits (unpack, 8);
  v3 = schro_unpack_decode_bits (unpack, 8);
  v4 = schro_unpack_decode_bits (unpack, 8);
  SCHRO_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    SCHRO_ERROR ("expected parse header");
    return;
  }

  decoder->parse_code = schro_unpack_decode_bits (unpack, 8);
  SCHRO_DEBUG ("parse code %02x", decoder->parse_code);

  decoder->next_parse_offset = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

static int
schro_decoder_check_version (int major, int minor)
{
  if (major == 0 && minor == 20071203) return TRUE;
  if (major == 1 && minor == 0) return TRUE;
  if (major == 2 && minor == 0) return TRUE;

  return FALSE;
}

void
schro_decoder_parse_access_unit (SchroDecoder *decoder)
{
  int bit;
  int index;
  SchroVideoFormat *format = &decoder->video_format;
  SchroUnpack *unpack = &decoder->unpack;

  SCHRO_DEBUG("decoding access unit");

  /* parse parameters */
  decoder->major_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("major_version = %d", decoder->major_version);
  decoder->minor_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("minor_version = %d", decoder->minor_version);
  decoder->profile = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("profile = %d", decoder->profile);
  decoder->level = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("level = %d", decoder->level);

  if (!schro_decoder_check_version (decoder->major_version, decoder->minor_version)) {
    SCHRO_ERROR("Stream version number %d:%d not handled, expecting 0:20071203, 1:0, or 2:0",
        decoder->major_version, decoder->minor_version);
  }
  if (decoder->profile != 0 || decoder->level != 0) {
    SCHRO_ERROR("Expecting profile/level 0:0, got %d:%d",
        decoder->profile, decoder->level);
  }

  /* base video format */
  index = schro_unpack_decode_uint (unpack);
  schro_video_format_set_std_video_format (format, index);

  /* source parameters */
  /* frame dimensions */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->width = schro_unpack_decode_uint (unpack);
    format->height = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG("size = %d x %d", format->width, format->height);

  /* chroma format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->chroma_format = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG("chroma_format %d", format->chroma_format);

  /* scan format */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->interlaced = schro_unpack_decode_bit (unpack);
    if (format->interlaced) {
      format->top_field_first = schro_unpack_decode_bit (unpack);
    }
  }
  SCHRO_DEBUG("interlaced %d top_field_first %d",
      format->interlaced, format->top_field_first);

  MARKER();

  /* frame rate */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    int index;
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->frame_rate_numerator = schro_unpack_decode_uint (unpack);
      format->frame_rate_denominator = schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_frame_rate (format, index);
    }
  }
  SCHRO_DEBUG("frame rate %d/%d", format->frame_rate_numerator,
      format->frame_rate_denominator);

  MARKER();

  /* aspect ratio */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    int index;
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->aspect_ratio_numerator =
        schro_unpack_decode_uint (unpack);
      format->aspect_ratio_denominator =
        schro_unpack_decode_uint (unpack);
    } else {
      schro_video_format_set_std_aspect_ratio (format, index);
    }
  }
  SCHRO_DEBUG("aspect ratio %d/%d", format->aspect_ratio_numerator,
      format->aspect_ratio_denominator);

  MARKER();

  /* clean area */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    format->clean_width = schro_unpack_decode_uint (unpack);
    format->clean_height = schro_unpack_decode_uint (unpack);
    format->left_offset = schro_unpack_decode_uint (unpack);
    format->top_offset = schro_unpack_decode_uint (unpack);
  }
  SCHRO_DEBUG("clean offset %d %d", format->left_offset,
      format->top_offset);
  SCHRO_DEBUG("clean size %d %d", format->clean_width,
      format->clean_height);

  MARKER();

  /* signal range */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    int index;
    index = schro_unpack_decode_uint (unpack);
    if (index == 0) {
      format->luma_offset = schro_unpack_decode_uint (unpack);
      format->luma_excursion = schro_unpack_decode_uint (unpack);
      format->chroma_offset = schro_unpack_decode_uint (unpack);
      format->chroma_excursion =
        schro_unpack_decode_uint (unpack);
    } else {
      if (index <= SCHRO_SIGNAL_RANGE_12BIT_VIDEO) {
        schro_video_format_set_std_signal_range (format, index);
      } else {
        schro_decoder_error (decoder, "signal range index out of range");
      }
    }
  }
  SCHRO_DEBUG("luma offset %d excursion %d", format->luma_offset,
      format->luma_excursion);
  SCHRO_DEBUG("chroma offset %d excursion %d", format->chroma_offset,
      format->chroma_excursion);

  MARKER();

  /* colour spec */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    int index;
    index = schro_unpack_decode_uint (unpack);
    if (index <= SCHRO_COLOUR_SPEC_CINEMA) {
      schro_video_format_set_std_colour_spec (format, index);
    } else {
      schro_decoder_error (decoder, "colour spec index out of range");
    }
    if (index == 0) {
      /* colour primaries */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_primaries = schro_unpack_decode_uint (unpack);
      }
      /* colour matrix */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->colour_matrix = schro_unpack_decode_uint (unpack);
      }
      /* transfer function */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        format->transfer_function = schro_unpack_decode_uint (unpack);
      }
    }
  }

  decoder->interlaced_coding = schro_unpack_decode_uint (unpack);
  if (decoder->interlaced_coding != 0) {
    SCHRO_ERROR("Decoder doesn't handle interlaced coding");
  }

  MARKER();

  schro_video_format_validate (format);
}

void
schro_decoder_parse_picture_header (SchroPicture *picture)
{
  SchroUnpack *unpack = &picture->decoder->unpack;
  SchroParams *params = &picture->params;

  schro_unpack_byte_sync(unpack);

  picture->picture_number = schro_unpack_decode_bits (unpack, 32);
  SCHRO_DEBUG("picture number %d", picture->picture_number);

  if (params->num_refs > 0) {
    picture->reference1 = picture->picture_number +
      schro_unpack_decode_sint (unpack);
    SCHRO_DEBUG("ref1 %d", picture->reference1);
  }

  if (params->num_refs > 1) {
    picture->reference2 = picture->picture_number +
      schro_unpack_decode_sint (unpack);
    SCHRO_DEBUG("ref2 %d", picture->reference2);
  }

  if (picture->is_ref) {
    picture->retired_picture_number = picture->picture_number +
      schro_unpack_decode_sint (unpack);
  }
}

void
schro_decoder_parse_picture_prediction_parameters (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;
  int bit;
  int index;

  /* block parameters */
  index = schro_unpack_decode_uint (unpack);
  if (index == 0) {
    params->xblen_luma = schro_unpack_decode_uint (unpack);
    params->yblen_luma = schro_unpack_decode_uint (unpack);
    params->xbsep_luma = schro_unpack_decode_uint (unpack);
    params->ybsep_luma = schro_unpack_decode_uint (unpack);
  } else {
    schro_params_set_block_params (params, index);
  }
  SCHRO_DEBUG("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  MARKER();

  /* mv precision */
  params->mv_precision = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("mv_precision %d", params->mv_precision);

  MARKER();

  /* global motion */
  params->have_global_motion = schro_unpack_decode_bit (unpack);
  if (params->have_global_motion) {
    int i;

    for (i=0;i<params->num_refs;i++) {
      SchroGlobalMotion *gm = params->global_motion + i;

      /* pan/tilt */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->b0 = schro_unpack_decode_sint (unpack);
        gm->b1 = schro_unpack_decode_sint (unpack);
      } else {
        gm->b0 = 0;
        gm->b1 = 0;
      }

      /* zoom/rotate/shear */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->a_exp = schro_unpack_decode_uint (unpack);
        gm->a00 = schro_unpack_decode_sint (unpack);
        gm->a01 = schro_unpack_decode_sint (unpack);
        gm->a10 = schro_unpack_decode_sint (unpack);
        gm->a11 = schro_unpack_decode_sint (unpack);
      } else {
        gm->a_exp = 0;
        gm->a00 = 1;
        gm->a01 = 0;
        gm->a10 = 0;
        gm->a11 = 1;
      }

      /* perspective */
      bit = schro_unpack_decode_bit (unpack);
      if (bit) {
        gm->c_exp = schro_unpack_decode_uint (unpack);
        gm->c0 = schro_unpack_decode_sint (unpack);
        gm->c1 = schro_unpack_decode_sint (unpack);
      } else {
        gm->c_exp = 0;
        gm->c0 = 0;
        gm->c1 = 0;
      }

      SCHRO_DEBUG("ref %d pan %d %d matrix %d %d %d %d perspective %d %d",
          i, gm->b0, gm->b1, gm->a00, gm->a01, gm->a10, gm->a11,
          gm->c0, gm->c1);
    }
  }

  MARKER();

  /* picture prediction mode */
  params->picture_pred_mode = schro_unpack_decode_uint (unpack);
  if (params->picture_pred_mode != 0) {
    schro_decoder_error (picture->decoder, "picture prediction mode != 0");
  }

  /* reference picture weights */
  params->picture_weight_bits = 1;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    params->picture_weight_bits = schro_unpack_decode_uint (unpack);
    params->picture_weight_1 = schro_unpack_decode_sint (unpack);
    if (params->num_refs > 1) {
      params->picture_weight_2 = schro_unpack_decode_sint (unpack);
    }
  }

  MARKER();
}

enum {
  SCHRO_DECODER_ARITH_SUPERBLOCK,
  SCHRO_DECODER_ARITH_PRED_MODE,
  SCHRO_DECODER_ARITH_VECTOR_REF1_X,
  SCHRO_DECODER_ARITH_VECTOR_REF1_Y,
  SCHRO_DECODER_ARITH_VECTOR_REF2_X,
  SCHRO_DECODER_ARITH_VECTOR_REF2_Y,
  SCHRO_DECODER_ARITH_DC_0,
  SCHRO_DECODER_ARITH_DC_1,
  SCHRO_DECODER_ARITH_DC_2
};

void
schro_decoder_parse_block_data (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;
  int i;

  for(i=0;i<9;i++){
    int length;

    if (params->num_refs < 2 && (i == SCHRO_DECODER_ARITH_VECTOR_REF2_X ||
          i == SCHRO_DECODER_ARITH_VECTOR_REF2_Y)) {
      picture->motion_buffers[i] = NULL;
      continue;
    }

    length = schro_unpack_decode_uint (unpack);
    schro_unpack_byte_sync (unpack);
    picture->motion_buffers[i] =
      schro_buffer_new_subbuffer (picture->input_buffer,
        schro_unpack_get_bits_read (unpack)/8, length);
    schro_unpack_skip_bits (unpack, length*8);
  }
}

void
schro_decoder_decode_block_data (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroArith *arith[9];
  SchroUnpack unpack[9];
  int i, j;

  memset(picture->motion->motion_vectors, 0,
      sizeof(SchroMotionVector)*params->y_num_blocks*params->x_num_blocks);

  for(i=0;i<9;i++){
    if (params->num_refs < 2 && (i == SCHRO_DECODER_ARITH_VECTOR_REF2_X ||
          i == SCHRO_DECODER_ARITH_VECTOR_REF2_Y)) {
      arith[i] = NULL;
      continue;
    }

    if (!params->is_noarith) {
      arith[i] = schro_arith_new ();
      schro_arith_decode_init (arith[i], picture->motion_buffers[i]);
    } else {
      schro_unpack_init_with_data (unpack + i,
          picture->motion_buffers[i]->data,
          picture->motion_buffers[i]->length, 1);
    }
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      schro_decoder_decode_macroblock(picture, arith, unpack, i, j);
    }
  }

  for(i=0;i<9;i++) {
    if (!params->is_noarith) {
      if (arith[i] == NULL) continue;

      if (arith[i]->offset < arith[i]->buffer->length) {
        SCHRO_WARNING("arith decoding %d didn't consume buffer (%d < %d)", i,
            arith[i]->offset, arith[i]->buffer->length);
      }
      if (arith[i]->offset > arith[i]->buffer->length + 6) {
        SCHRO_ERROR("arith decoding %d overran buffer (%d > %d)", i,
            arith[i]->offset, arith[i]->buffer->length);
      }
      schro_arith_free (arith[i]);
    } else {
      /* FIXME complain about buffer over/underrun */
    }
  }
}

void
schro_decoder_decode_macroblock(SchroPicture *picture, SchroArith **arith,
    SchroUnpack *unpack, int i, int j)
{
  SchroParams *params = &picture->params;
  SchroMotion *motion = picture->motion;
  SchroMotionVector *mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  int k,l;
  int split_prediction;

  split_prediction = schro_motion_split_prediction (motion, i, j);
  if (!params->is_noarith) {
    mv->split = (split_prediction +
        _schro_arith_decode_uint (arith[SCHRO_DECODER_ARITH_SUPERBLOCK],
          SCHRO_CTX_SB_F1, SCHRO_CTX_SB_DATA))%3;
  } else {
    mv->split = (split_prediction +
        schro_unpack_decode_uint (unpack + SCHRO_DECODER_ARITH_SUPERBLOCK))%3;
  }

  switch (mv->split) {
    case 0:
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j);
      mv[1] = mv[0];
      mv[2] = mv[0];
      mv[3] = mv[0];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 2*params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 3*params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 1:
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i + 2, j);
      mv[3] = mv[2];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));

      mv += 2*params->x_num_blocks;
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i, j + 2);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (picture, arith, unpack,
          motion->motion_vectors, i + 2, j + 2);
      mv[3] = mv[2];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 2:
      for (l=0;l<4;l++) {
        for (k=0;k<4;k++) {
          schro_decoder_decode_prediction_unit (picture, arith, unpack,
              motion->motion_vectors, i + k, j + l);
        }
      }
      break;
    default:
      SCHRO_ERROR("mv->split == %d, split_prediction %d", mv->split, split_prediction);
      SCHRO_ASSERT(0);
  }
}

void
schro_decoder_decode_prediction_unit(SchroPicture *picture, SchroArith **arith,
    SchroUnpack *unpack, SchroMotionVector *motion_vectors, int x, int y)
{
  SchroParams *params = &picture->params;
  SchroMotion *motion = picture->motion;
  SchroMotionVector *mv = &motion_vectors[y*params->x_num_blocks + x];

  mv->pred_mode = schro_motion_get_mode_prediction (motion,
      x, y);
  if (!params->is_noarith) {
    mv->pred_mode ^= 
      _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
          SCHRO_CTX_BLOCK_MODE_REF1);
  } else {
    mv->pred_mode ^= 
      schro_unpack_decode_bit (unpack + SCHRO_DECODER_ARITH_PRED_MODE);
  }
  if (params->num_refs > 1) {
    if (!params->is_noarith) {
      mv->pred_mode ^=
        _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
            SCHRO_CTX_BLOCK_MODE_REF2) << 1;
    } else {
      mv->pred_mode ^= 
        schro_unpack_decode_bit (unpack + SCHRO_DECODER_ARITH_PRED_MODE) << 1;
    }
  }

  if (mv->pred_mode == 0) {
    int pred[3];
    SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

    schro_motion_dc_prediction (motion, x, y, pred);

    if (!params->is_noarith) {
      mvdc->dc[0] = pred[0] + _schro_arith_decode_sint (
          arith[SCHRO_DECODER_ARITH_DC_0],
          SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
          SCHRO_CTX_LUMA_DC_SIGN);
      mvdc->dc[1] = pred[1] + _schro_arith_decode_sint (
          arith[SCHRO_DECODER_ARITH_DC_1],
          SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
          SCHRO_CTX_CHROMA1_DC_SIGN);
      mvdc->dc[2] = pred[2] + _schro_arith_decode_sint (
          arith[SCHRO_DECODER_ARITH_DC_2],
          SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
          SCHRO_CTX_CHROMA2_DC_SIGN);
    } else {
      mvdc->dc[0] = pred[0] +
        schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_0);
      mvdc->dc[1] = pred[1] +
        schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_1);
      mvdc->dc[2] = pred[2] +
        schro_unpack_decode_sint (unpack + SCHRO_DECODER_ARITH_DC_2);
    }
  } else {
    int pred_x, pred_y;

    if (params->have_global_motion) {
      int pred;
      pred = schro_motion_get_global_prediction (motion, x, y);
      if (!params->is_noarith) {
        mv->using_global = pred ^ _schro_arith_decode_bit (
            arith[SCHRO_DECODER_ARITH_PRED_MODE], SCHRO_CTX_GLOBAL_BLOCK);
      } else {
        mv->using_global = pred ^ schro_unpack_decode_bit (
            unpack + SCHRO_DECODER_ARITH_PRED_MODE);
      }
    } else {
      mv->using_global = FALSE;
    }
    if (!mv->using_global) {
      if (mv->pred_mode & 1) {
        schro_motion_vector_prediction (motion, x, y,
            &pred_x, &pred_y, 1);

        if (!params->is_noarith) {
          mv->x1 = pred_x + _schro_arith_decode_sint (
                arith[SCHRO_DECODER_ARITH_VECTOR_REF1_X],
                SCHRO_CTX_MV_REF1_H_CONT_BIN1, SCHRO_CTX_MV_REF1_H_VALUE,
                SCHRO_CTX_MV_REF1_H_SIGN);
          mv->y1 = pred_y + _schro_arith_decode_sint (
                arith[SCHRO_DECODER_ARITH_VECTOR_REF1_Y],
                SCHRO_CTX_MV_REF1_V_CONT_BIN1, SCHRO_CTX_MV_REF1_V_VALUE,
                SCHRO_CTX_MV_REF1_V_SIGN);
        } else {
          mv->x1 = pred_x + schro_unpack_decode_sint (
                unpack + SCHRO_DECODER_ARITH_VECTOR_REF1_X);
          mv->y1 = pred_y + schro_unpack_decode_sint (
                unpack + SCHRO_DECODER_ARITH_VECTOR_REF1_Y);
        }
      }
      if (mv->pred_mode & 2) {
        schro_motion_vector_prediction (motion, x, y,
            &pred_x, &pred_y, 2);

        if (!params->is_noarith) {
          mv->x2 = pred_x + _schro_arith_decode_sint (
                arith[SCHRO_DECODER_ARITH_VECTOR_REF2_X],
                SCHRO_CTX_MV_REF2_H_CONT_BIN1, SCHRO_CTX_MV_REF2_H_VALUE,
                SCHRO_CTX_MV_REF2_H_SIGN);
          mv->y2 = pred_y + _schro_arith_decode_sint (
                arith[SCHRO_DECODER_ARITH_VECTOR_REF2_Y],
                SCHRO_CTX_MV_REF2_V_CONT_BIN1, SCHRO_CTX_MV_REF2_V_VALUE,
                SCHRO_CTX_MV_REF2_V_SIGN);
        } else {
          mv->x2 = pred_x + schro_unpack_decode_sint (
                unpack + SCHRO_DECODER_ARITH_VECTOR_REF2_X);
          mv->y2 = pred_y + schro_unpack_decode_sint (
                unpack + SCHRO_DECODER_ARITH_VECTOR_REF2_Y);
        }
      }
    } else {
      mv->x1 = 0;
      mv->y1 = 0;
      mv->x2 = 0;
      mv->y2 = 0;
    }
  }
}

void
schro_decoder_parse_transform_parameters (SchroPicture *picture)
{
  int bit;
  int i;
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;

  /* transform */
  params->wavelet_filter_index = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  params->transform_depth = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG ("transform depth %d", params->transform_depth);

  if (!params->is_lowdelay) {
    /* codeblock parameters */
    params->codeblock_mode_index = 0;
    for(i=0;i<params->transform_depth + 1;i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }

    bit = schro_unpack_decode_bit (unpack);
    if (bit) {
      int i;
      for(i=0;i<params->transform_depth + 1;i++) {
        params->horiz_codeblocks[i] = schro_unpack_decode_uint (unpack);
        params->vert_codeblocks[i] = schro_unpack_decode_uint (unpack);
      }
      params->codeblock_mode_index = schro_unpack_decode_uint (unpack);
    }
  } else {
    /* slice parameters */
    params->n_horiz_slices = schro_unpack_decode_uint(unpack);
    params->n_vert_slices = schro_unpack_decode_uint(unpack);

    params->slice_bytes_num = schro_unpack_decode_uint(unpack);
    params->slice_bytes_denom = schro_unpack_decode_uint(unpack);

    /* quant matrix */
    bit = schro_unpack_decode_bit (unpack);
    if (bit) {
      params->quant_matrix[0] = schro_unpack_decode_uint (unpack);
      for(i=0;i<params->transform_depth;i++){
        params->quant_matrix[1+3*i] = schro_unpack_decode_uint (unpack);
        params->quant_matrix[2+3*i] = schro_unpack_decode_uint (unpack);
        params->quant_matrix[3+3*i] = schro_unpack_decode_uint (unpack);
      }
    } else {
      schro_params_set_default_quant_matrix (params);
    }
  }
}

void
schro_decoder_init_subband_frame_data_interleaved (SchroPicture *picture)
{
  int i;
  int component;
  SchroFrameData *comp;
  SchroFrameData *fd;
  SchroParams *params = &picture->params;
  int shift;
  int position;

  for(component=0;component<3;component++){
    comp = &picture->frame->components[component];
    for(i=0;i<1+3*params->transform_depth;i++) {
      position = schro_subband_get_position (i);

      fd = &picture->subband_data[component][i];

      shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);

      fd->format = picture->frame->format;
      fd->h_shift = comp->h_shift + shift;
      fd->v_shift = comp->v_shift + shift;
      fd->stride = comp->stride << shift;
      if (component == 0) {
        fd->width = params->iwt_luma_width >> shift;
        fd->height = params->iwt_luma_height >> shift;
      } else {
        fd->width = params->iwt_chroma_width >> shift;
        fd->height = params->iwt_chroma_height >> shift;
      }

      fd->data = comp->data;
      if (position & 2) {
        fd->data = OFFSET(fd->data, fd->stride>>1);
      }
      if (position & 1) {
        fd->data = OFFSET(fd->data, fd->width*sizeof(int16_t));
      }
    }
  }
}

void
schro_decoder_parse_lowdelay_transform_data (SchroPicture *picture)
{
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;
  int length;

  length = (params->slice_bytes_num * params->n_horiz_slices *
      params->n_vert_slices) / params->slice_bytes_denom;
  picture->lowdelay_buffer = schro_buffer_new_subbuffer (
      picture->input_buffer,
      schro_unpack_get_bits_read (unpack)/8,
      length);
  schro_unpack_skip_bits (unpack, length*8);
}

void
schro_decoder_parse_transform_data (SchroPicture *picture)
{
  int i;
  int component;
  SchroParams *params = &picture->params;
  SchroUnpack *unpack = &picture->decoder->unpack;
  int subband_length;

  for(component=0;component<3;component++){
    for(i=0;i<1+3*params->transform_depth;i++) {
      schro_unpack_byte_sync (unpack);

      subband_length = schro_unpack_decode_uint (unpack);

      SCHRO_DEBUG("subband %d %d length %d", component, i,
          subband_length);

      if (subband_length == 0) {
        SCHRO_DEBUG("subband is zero");
        schro_unpack_byte_sync (unpack);

        picture->subband_quant_index[component][i] = 0;
        picture->subband_length[component][i] = 0;
        picture->subband_buffer[component][i] = NULL;
      } else {
        int quant_index;

        quant_index = schro_unpack_decode_uint (unpack);
        SCHRO_DEBUG("quant index %d", quant_index);

        /* FIXME check quant_index */
        SCHRO_MILD_ASSERT(quant_index >= 0);
        SCHRO_MILD_ASSERT(quant_index <= 60);

        schro_unpack_byte_sync (unpack);

        picture->subband_quant_index[component][i] = quant_index;
        picture->subband_length[component][i] = subband_length;
        picture->subband_buffer[component][i] = schro_buffer_new_subbuffer (
            picture->input_buffer,
            schro_unpack_get_bits_read (unpack)/8,
            subband_length);
        schro_unpack_skip_bits (unpack, subband_length*8);
      }
    }
  }
}

void
schro_decoder_decode_transform_data (SchroPicture *picture)
{
  int i;
  int component;
  SchroParams *params = &picture->params;
  SchroPictureSubbandContext context = { 0 }, *ctx = &context;
  int skip_subbands;
  
  /* FIXME some day, hook this up into automatic degraded decoding */
  skip_subbands = 0;

  for(component=0;component<3;component++){
    for(i=0;i<1+3*params->transform_depth - skip_subbands;i++) {
      ctx->component = component;
      ctx->index = i;
      ctx->position = schro_subband_get_position(i);

      schro_decoder_decode_subband (picture, ctx);
    }
  }
}

static void
codeblock_line_decode_generic (SchroPictureSubbandContext *ctx,
    int16_t *line, int j, const int16_t *parent_data, const int16_t *prev)
{
  int i;

  for(i=ctx->xmin;i<ctx->xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i>>1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j>0) nhood_or |= prev[i];
    if (i>0) nhood_or |= line[i-1];
    if (i>0 && j>0) nhood_or |= prev[i-1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(ctx->position)) {
      if (i > 0) previous_value = line[i-1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(ctx->position)) {
      if (j > 0) previous_value = prev[i];
    }

#define STUFF \
  do { \
    int cont_context, sign_context, value_context; \
    \
    if (parent == 0) { \
      cont_context = nhood_or ? SCHRO_CTX_ZPNN_F1 : SCHRO_CTX_ZPZN_F1; \
    } else { \
      cont_context = nhood_or ? SCHRO_CTX_NPNN_F1 : SCHRO_CTX_NPZN_F1; \
    } \
     \
    if (previous_value < 0) { \
      sign_context = SCHRO_CTX_SIGN_NEG; \
    } else { \
      sign_context = (previous_value > 0) ? SCHRO_CTX_SIGN_POS : \
        SCHRO_CTX_SIGN_ZERO; \
    } \
 \
    value_context = SCHRO_CTX_COEFF_DATA; \
 \
    v = _schro_arith_decode_uint (ctx->arith, cont_context, \
        value_context); \
    if (v) { \
      v = (ctx->quant_offset + ctx->quant_factor * v + 2)>>2; \
      if (_schro_arith_decode_bit (ctx->arith, sign_context)) { \
        v = -v; \
      } \
      line[i] = v; \
    } else { \
      line[i] = 0; \
    } \
  } while(0)

    STUFF;
  }
}

static void
codeblock_line_decode_noarith (SchroPictureSubbandContext *ctx,
    int16_t *line)
{
  int i;

  for(i=ctx->xmin;i<ctx->xmax;i++){
    line[i] = schro_dequantise (schro_unpack_decode_sint (&ctx->unpack),
        ctx->quant_factor, ctx->quant_offset);
  }
}

#if 0
static void
codeblock_line_decode_deep (SchroPictureSubbandContext *ctx,
    int32_t *line, int j, const int32_t *parent_data, const int32_t *prev)
{
  int i;

  for(i=ctx->xmin;i<ctx->xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i>>1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j>0) nhood_or |= prev[i];
    if (i>0) nhood_or |= line[i-1];
    if (i>0 && j>0) nhood_or |= prev[i-1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(ctx->position)) {
      if (i > 0) previous_value = line[i-1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(ctx->position)) {
      if (j > 0) previous_value = prev[i];
    }

    STUFF;
  }
}

static void
codeblock_line_decode_deep_parent (SchroPictureSubbandContext *ctx,
    int16_t *line, int j, const int32_t *parent_data, const int16_t *prev)
{
  int i;

  for(i=ctx->xmin;i<ctx->xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    if (parent_data) {
      parent = parent_data[(i>>1)];
    } else {
      parent = 0;
    }

    nhood_or = 0;
    if (j>0) nhood_or |= prev[i];
    if (i>0) nhood_or |= line[i-1];
    if (i>0 && j>0) nhood_or |= prev[i-1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(ctx->position)) {
      if (i > 0) previous_value = line[i-1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(ctx->position)) {
      if (j > 0) previous_value = prev[i];
    }

    STUFF;
  }
}
#endif


static void
codeblock_line_decode_p_horiz (SchroPictureSubbandContext *ctx,
    int16_t *line, int j, const int16_t *parent_data, const int16_t *prev)
{
  int i = ctx->xmin;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  if (i == 0) {
    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    i++;
  }
  for(;i<ctx->xmax;i++){
    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= line[i-1];
    nhood_or |= prev[i-1];

    previous_value = line[i-1];

    STUFF;
  }
}

static void
codeblock_line_decode_p_vert (SchroPictureSubbandContext *ctx,
    int16_t *line, int j, const int16_t *parent_data, const int16_t *prev)
{
  int i = ctx->xmin;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  if (i == 0) {
    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = prev[i];

    STUFF;
    i++;
  }
  for(;i<ctx->xmax;i++){
    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= line[i-1];
    nhood_or |= prev[i-1];

    previous_value = prev[i];

    STUFF;
  }
}

static void
codeblock_line_decode_p_diag (SchroPictureSubbandContext *ctx,
    int16_t *line, int j,
    const int16_t *parent_data,
    const int16_t *prev)
{
  int i;
  int v;
  int parent;
  int nhood_or;
  int previous_value;

  i = ctx->xmin;
  if (i == 0) {
    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    i++;
  }
  for(;i<ctx->xmax;i++){
    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= line[i-1];
    nhood_or |= prev[i-1];
    previous_value = 0;

    STUFF;
  }
}

void
schro_decoder_subband_dc_predict (SchroFrameData *fd)
{
  int16_t *prev_line;
  int16_t *line;
  int i,j;
  int pred_value;

  line = SCHRO_FRAME_DATA_GET_LINE(fd, 0);
  for(i=1;i<fd->width;i++){
    pred_value = line[i-1];
    line[i] += pred_value;
  }
  
  for(j=1;j<fd->height;j++){
    line = SCHRO_FRAME_DATA_GET_LINE(fd, j);
    prev_line = SCHRO_FRAME_DATA_GET_LINE(fd, j-1);

    pred_value = prev_line[0];
    line[0] += pred_value;

    for(i=1;i<fd->width;i++){
      pred_value = schro_divide(line[i-1] + prev_line[i] +
          prev_line[i-1] + 1,3);
      line[i] += pred_value;
    }
  }

}

static void
schro_decoder_setup_codeblocks (SchroPicture *picture,
    SchroPictureSubbandContext *ctx)
{
  SchroParams *params = &picture->params;

  if (ctx->position == 0) {
    ctx->vert_codeblocks = params->vert_codeblocks[0];
    ctx->horiz_codeblocks = params->horiz_codeblocks[0];
  } else {
    ctx->vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT(ctx->position)+1];
    ctx->horiz_codeblocks = params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT(ctx->position)+1];
  }
  if ((ctx->horiz_codeblocks > 1 || ctx->vert_codeblocks > 1) && ctx->position > 0) {
    ctx->have_zero_flags = TRUE;
  } else {
    ctx->have_zero_flags = FALSE;
  }
  if (ctx->horiz_codeblocks > 1 || ctx->vert_codeblocks > 1) {
    if (params->codeblock_mode_index == 1) {
      ctx->have_quant_offset = TRUE;
    } else {
      ctx->have_quant_offset = FALSE;
    }
  } else {
    ctx->have_quant_offset = FALSE;
  }
}

static void
schro_decoder_zero_block (SchroPictureSubbandContext *ctx,
    int x1, int y1, int x2, int y2)
{
  int j;
  int16_t *line;

  SCHRO_DEBUG("subband is zero");
  for(j=y1;j<y2;j++){
    line = SCHRO_FRAME_DATA_GET_LINE (ctx->frame_data, j);
    oil_splat_s16_ns (line + x1, schro_zero, x2 - x1);
  }
}

static void
schro_decoder_decode_codeblock (SchroPicture *picture,
    SchroPictureSubbandContext *ctx)
{
  SchroParams *params = &picture->params;
  int j;

  if (ctx->have_zero_flags) {
    int bit;

    /* zero codeblock */
    if (params->is_noarith) {
      bit = schro_unpack_decode_bit (&ctx->unpack);
    } else {
      bit = _schro_arith_decode_bit (ctx->arith, SCHRO_CTX_ZERO_CODEBLOCK);
    }
    if (bit) {
      schro_decoder_zero_block (ctx, ctx->xmin, ctx->ymin,
          ctx->xmax, ctx->ymax);
      return;
    }
  }

  if (ctx->have_quant_offset) {
    if (params->is_noarith) {
      ctx->quant_index += schro_unpack_decode_sint (&ctx->unpack);
    } else {
      ctx->quant_index += _schro_arith_decode_sint (ctx->arith,
          SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
          SCHRO_CTX_QUANTISER_SIGN);
    }

    /* FIXME check quant_index */
    SCHRO_MILD_ASSERT(ctx->quant_index >= 0);
    SCHRO_MILD_ASSERT(ctx->quant_index <= 60);
  }

  ctx->quant_factor = schro_table_quant[ctx->quant_index];
  if (params->num_refs > 0) {
    ctx->quant_offset = schro_table_offset_3_8[ctx->quant_index];
  } else {
    ctx->quant_offset = schro_table_offset_1_2[ctx->quant_index];
  }

  for(j=ctx->ymin;j<ctx->ymax;j++){
    int16_t *p = SCHRO_FRAME_DATA_GET_LINE(ctx->frame_data,j);
    const int16_t *parent_line;
    const int16_t *prev_line;

    if (ctx->position >= 4) {
      parent_line = SCHRO_FRAME_DATA_GET_LINE(ctx->parent_frame_data, (j>>1));
    } else {
      parent_line = NULL;
    }
    if (j==0) {
      prev_line = schro_zero;
    } else {
      prev_line = SCHRO_FRAME_DATA_GET_LINE (ctx->frame_data, (j-1));
    }
    if (params->is_noarith) {
      codeblock_line_decode_noarith (ctx, p);
    } else if (ctx->position >= 4) {
      if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(ctx->position)) {
        codeblock_line_decode_p_horiz (ctx, p, j, parent_line, prev_line);
      } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(ctx->position)) {
        codeblock_line_decode_p_vert (ctx, p, j, parent_line, prev_line);
      } else {
        codeblock_line_decode_p_diag (ctx, p, j, parent_line, prev_line);
      }
    } else {
      codeblock_line_decode_generic (ctx, p, j, parent_line, prev_line);
    }
  }
}

void
schro_decoder_decode_subband (SchroPicture *picture,
    SchroPictureSubbandContext *ctx)
{
  SchroParams *params = &picture->params;
  int x,y;

  ctx->subband_length = picture->subband_length[ctx->component][ctx->index];
  ctx->quant_index = picture->subband_quant_index[ctx->component][ctx->index];

  ctx->frame_data = &picture->subband_data[ctx->component][ctx->index];
  if (ctx->position >= 4) {
    ctx->parent_frame_data = &picture->subband_data[ctx->component][ctx->index-3];
  }

  if (picture->subband_length[ctx->component][ctx->index] == 0) {
    schro_decoder_zero_block (ctx, 0, 0,
        ctx->frame_data->width, ctx->frame_data->height);
    return;
  }

  if (!params->is_noarith) {
    ctx->arith = schro_arith_new ();
    schro_arith_decode_init (ctx->arith,
        picture->subband_buffer[ctx->component][ctx->index]);
  } else {
    schro_unpack_init_with_data (&ctx->unpack,
        picture->subband_buffer[ctx->component][ctx->index]->data,
        picture->subband_buffer[ctx->component][ctx->index]->length, 1);
  }

  schro_decoder_setup_codeblocks (picture, ctx);

  for(y=0;y<ctx->vert_codeblocks;y++){
    ctx->ymin = (ctx->frame_data->height*y)/ctx->vert_codeblocks;
    ctx->ymax = (ctx->frame_data->height*(y+1))/ctx->vert_codeblocks;

    for(x=0;x<ctx->horiz_codeblocks;x++){

      ctx->xmin = (ctx->frame_data->width*x)/ctx->horiz_codeblocks;
      ctx->xmax = (ctx->frame_data->width*(x+1))/ctx->horiz_codeblocks;
      
      schro_decoder_decode_codeblock (picture, ctx);
    }
  }
  if (!params->is_noarith) {
    schro_arith_decode_flush (ctx->arith);
    if (ctx->arith->offset < ctx->subband_length) {
      SCHRO_ERROR("arith decoding didn't consume buffer (%d < %d)",
          ctx->arith->offset, ctx->subband_length);
    }
    if (ctx->arith->offset > ctx->subband_length + 4) {
      SCHRO_ERROR("arith decoding overran buffer (%d > %d)",
          ctx->arith->offset, ctx->subband_length);
    }
    schro_arith_free (ctx->arith);
  } else {
    /* FIXME check noarith decoding */
  }

  if (ctx->position == 0 && picture->params.num_refs == 0) {
    schro_decoder_subband_dc_predict (ctx->frame_data);
  }
}

/* reference pool */

static void
schro_decoder_reference_add (SchroDecoder *decoder, SchroPicture *picture)
{
  SCHRO_DEBUG("adding %d", picture->picture_number);

  if (schro_queue_is_full(decoder->reference_queue)) {
    schro_queue_pop (decoder->reference_queue);
  }
  schro_queue_add (decoder->reference_queue, schro_picture_ref(picture),
      picture->picture_number);
}

static SchroPicture *
schro_decoder_reference_get (SchroDecoder *decoder,
    SchroPictureNumber picture_number)
{
  SCHRO_DEBUG("getting %d", picture_number);
  return schro_queue_find (decoder->reference_queue, picture_number);
}

static void
schro_decoder_reference_retire (SchroDecoder *decoder,
    SchroPictureNumber picture_number)
{
  SCHRO_DEBUG("retiring %d", picture_number);
  schro_queue_delete (decoder->reference_queue, picture_number);
}

static void
schro_decoder_error (SchroDecoder *decoder, const char *s)
{
  SCHRO_DEBUG("decoder error");
  decoder->error = TRUE;
  if (!decoder->error_message) {
    decoder->error_message = strdup(s);
  }
}

