
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <schroedinger/schrooil.h>
#include <string.h>

int _schro_decode_prediction_only;

static void schro_decoder_decode_macroblock(SchroDecoder *decoder,
    SchroArith **arith, int i, int j);
static void schro_decoder_decode_prediction_unit(SchroDecoder *decoder,
    SchroArith **arith, SchroMotionVector *motion_vectors, int x, int y);
static void schro_decoder_init (SchroDecoder *decoder);

static void schro_decoder_reference_add (SchroDecoder *decoder,
    SchroUpsampledFrame *frame, SchroPictureNumber picture_number);
static SchroUpsampledFrame * schro_decoder_reference_get (SchroDecoder *decoder,
    SchroPictureNumber frame_number);
static void schro_decoder_reference_retire (SchroDecoder *decoder,
    SchroPictureNumber frame_number);

static void schro_decoder_error (SchroDecoder *decoder, const char *s);

int16_t schro_zero[SCHRO_LIMIT_WIDTH];

SchroDecoder *
schro_decoder_new (void)
{
  SchroDecoder *decoder;

  decoder = malloc(sizeof(SchroDecoder));
  memset (decoder, 0, sizeof(SchroDecoder));

  decoder->tmpbuf = malloc(SCHRO_LIMIT_WIDTH * 2);
  decoder->tmpbuf2 = malloc(SCHRO_LIMIT_WIDTH * 2);

  decoder->params.video_format = &decoder->video_format;

  decoder->skip_value = 1.0;
  decoder->skip_ratio = 1.0;

  decoder->reference_queue = schro_queue_new (SCHRO_MAX_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_upsampled_frame_free);
  decoder->frame_queue = schro_queue_new (SCHRO_MAX_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_frame_unref);
  decoder->output_queue = schro_queue_new (SCHRO_MAX_REFERENCE_FRAMES,
      (SchroQueueFreeFunc)schro_frame_unref);

  return decoder;
}

void
schro_decoder_free (SchroDecoder *decoder)
{
  if (decoder->frame) {
    schro_frame_unref (decoder->frame);
  }
  schro_queue_free (decoder->output_queue);
  schro_queue_free (decoder->reference_queue);
  schro_queue_free (decoder->frame_queue);

  if (decoder->motion_field) schro_motion_field_free (decoder->motion_field);
  if (decoder->mc_tmp_frame) schro_frame_unref (decoder->mc_tmp_frame);
  if (decoder->planar_output_frame) schro_frame_unref (decoder->planar_output_frame);

  if (decoder->tmpbuf) free (decoder->tmpbuf);
  if (decoder->tmpbuf2) free (decoder->tmpbuf2);
  if (decoder->error_message) free (decoder->error_message);

  free (decoder);
}

void
schro_decoder_reset (SchroDecoder *decoder)
{
  schro_queue_clear (decoder->frame_queue);
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

  format = malloc(sizeof(SchroVideoFormat));
  memcpy (format, &decoder->video_format, sizeof(SchroVideoFormat));

  return format;
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

  if (data[4] == SCHRO_PARSE_CODE_ACCESS_UNIT) return 1;

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

  if (data[4] == SCHRO_PARSE_CODE_END_SEQUENCE) return 1;

  return 0;
}

SchroFrame *
schro_decoder_pull (SchroDecoder *decoder)
{
  SchroFrame *ret;

  SCHRO_DEBUG("searching for frame %d", decoder->next_frame_number);
  ret = schro_queue_remove (decoder->frame_queue, decoder->next_frame_number);
  if (ret) {
    decoder->next_frame_number++;
  }
  return ret;
}

void
schro_decoder_push (SchroDecoder *decoder, SchroBuffer *buffer)
{
  SCHRO_ASSERT(decoder->input_buffer == NULL);

  decoder->input_buffer = buffer;
}

int
schro_decoder_iterate (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  int i;
  SchroFrame *output_picture;
  
  if (decoder->input_buffer == NULL) {
    return SCHRO_DECODER_NEED_BITS;
  }

  decoder->bits = schro_bits_new ();
  schro_bits_decode_init (decoder->bits, decoder->input_buffer);

  schro_decoder_decode_parse_header(decoder);

  if (decoder->code == SCHRO_PARSE_CODE_ACCESS_UNIT) {
    SCHRO_INFO ("decoding access unit");
    schro_decoder_decode_access_unit(decoder);

    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    schro_bits_free (decoder->bits);

    if (decoder->have_access_unit) {
      return SCHRO_DECODER_OK;
    }
    schro_decoder_init (decoder);
    decoder->have_access_unit = TRUE;
    return SCHRO_DECODER_FIRST_ACCESS_UNIT;
  }

  if (decoder->code == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    schro_bits_free (decoder->bits);
    
    return SCHRO_DECODER_OK;
  }

  if (schro_decoder_is_end_sequence (decoder->input_buffer)) {
    SCHRO_INFO ("decoding end sequence");
    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    schro_bits_free (decoder->bits);
    return SCHRO_DECODER_EOS;
  }

  if (!decoder->have_access_unit) {
    SCHRO_INFO ("no access unit -- dropping frame");
    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    schro_bits_free (decoder->bits);
    return SCHRO_DECODER_OK;
  }

  if (schro_queue_is_empty (decoder->output_queue)) {
    schro_bits_free (decoder->bits);
    return SCHRO_DECODER_NEED_FRAME;
  }

  schro_decoder_decode_picture_header(decoder);

  params->num_refs = SCHRO_PARSE_CODE_NUM_REFS(decoder->code);

  schro_params_init (params, decoder->video_format.index);

  if (!decoder->have_frame_number) {
    if (SCHRO_PARSE_CODE_NUM_REFS (decoder->code) > 0) {
      SCHRO_ERROR("expected I frame after access unit header");
    }
    decoder->next_frame_number = decoder->picture_number;
    decoder->have_frame_number = TRUE;
    SCHRO_INFO("next frame number after seek %d", decoder->next_frame_number);
  }

  if (SCHRO_PARSE_CODE_IS_NON_REFERENCE (decoder->code) &&
       decoder->skip_value > decoder->skip_ratio) {

    decoder->skip_value = 0.8 * decoder->skip_value;
    SCHRO_INFO("skipping frame %d", decoder->picture_number);
    SCHRO_DEBUG("skip value %g ratio %g", decoder->skip_value, decoder->skip_ratio);

    for(i=0;i<decoder->n_retire;i++){
      schro_decoder_reference_retire (decoder, decoder->retire_list[i]);
    }

    schro_buffer_unref (decoder->input_buffer);
    decoder->input_buffer = NULL;
    schro_bits_free (decoder->bits);

    output_picture = schro_frame_new ();
    output_picture->frame_number = decoder->picture_number;

    SCHRO_DEBUG("adding %d to queue (skipped)", output_picture->frame_number);
    schro_queue_add (decoder->frame_queue, output_picture,
        decoder->picture_number);

    return SCHRO_DECODER_OK;
  }

  decoder->skip_value = 0.8 * decoder->skip_value + 0.2;
SCHRO_DEBUG("skip value %g ratio %g", decoder->skip_value, decoder->skip_ratio);

  output_picture = schro_queue_pull (decoder->output_queue);

  if (SCHRO_PARSE_CODE_NUM_REFS(decoder->code) > 0) {
    int skip = 0;

    SCHRO_DEBUG("inter");

    schro_bits_sync (decoder->bits);
    schro_decoder_decode_frame_prediction (decoder);
    schro_params_calculate_mc_sizes (params);

    if (decoder->motion_field == NULL) {
      decoder->motion_field = schro_motion_field_new (params->x_num_blocks,
          params->y_num_blocks);
    }

    decoder->ref0 = schro_decoder_reference_get (decoder, decoder->reference1);
    if (decoder->ref0 == NULL) {
      SCHRO_INFO("Could not find reference picture %d", decoder->reference1);
      skip = 1;
    }

    if (decoder->n_refs > 1) {
      decoder->ref1 = schro_decoder_reference_get (decoder, decoder->reference2);
      if (decoder->ref1 == NULL) {
        SCHRO_INFO("Could not find reference picture %d", decoder->reference2);
        skip = 1;
      }
    }

    if (skip) {
      for(i=0;i<decoder->n_retire;i++){
        schro_decoder_reference_retire (decoder, decoder->retire_list[i]);
      }

      schro_buffer_unref (decoder->input_buffer);
      decoder->input_buffer = NULL;
      schro_bits_free (decoder->bits);

      return SCHRO_DECODER_OK;
    }

    schro_bits_sync (decoder->bits);
    schro_decoder_decode_prediction_data (decoder);
    {
      SchroMotion motion;

      if (params->mv_precision > 0) {
        schro_upsampled_frame_upsample (decoder->ref0);
        if (decoder->ref1) {
          schro_upsampled_frame_upsample (decoder->ref1);
        }
      }

      motion.src1 = decoder->ref0;
      if (decoder->ref1) {
        motion.src2 = decoder->ref1;
      }

      motion.motion_vectors = decoder->motion_field->motion_vectors;
      motion.params = &decoder->params;
      schro_frame_copy_with_motion (decoder->mc_tmp_frame, &motion);
    }
  }

  schro_bits_sync (decoder->bits);
  decoder->zero_residual = FALSE;
  if (params->num_refs > 0) {
    decoder->zero_residual = schro_bits_decode_bit (decoder->bits);

    SCHRO_DEBUG ("zero residual %d", decoder->zero_residual);
  }

  if (!decoder->zero_residual) {
    schro_decoder_decode_transform_parameters (decoder);
    schro_params_calculate_iwt_sizes (params);

    schro_bits_sync (decoder->bits);
    schro_params_init_subbands (params, decoder->subbands,
        decoder->frame->components[0].stride,
        decoder->frame->components[1].stride);
    schro_decoder_decode_transform_data (decoder);

    schro_frame_inverse_iwt_transform (decoder->frame, &decoder->params,
        decoder->tmpbuf);
  }

  if (decoder->zero_residual) {
    if (SCHRO_FRAME_IS_PACKED(output_picture->format)) {
      schro_frame_convert (decoder->planar_output_frame, decoder->mc_tmp_frame);
      schro_frame_convert (output_picture, decoder->planar_output_frame);
    } else {
      schro_frame_convert (output_picture, decoder->mc_tmp_frame);
    }
  } else if (!_schro_decode_prediction_only) {
    if (SCHRO_PARSE_CODE_IS_INTER(decoder->code)) {
      schro_frame_add (decoder->frame, decoder->mc_tmp_frame);
    }

    if (SCHRO_FRAME_IS_PACKED(output_picture->format)) {
      schro_frame_convert (decoder->planar_output_frame, decoder->frame);
      schro_frame_convert (output_picture, decoder->planar_output_frame);
    } else {
      schro_frame_convert (output_picture, decoder->frame);
    }
  } else {
    SchroFrame *frame;
    if (SCHRO_PARSE_CODE_IS_INTER(decoder->code)) {
      frame = decoder->mc_tmp_frame;
    } else {
      frame = decoder->frame;
    }
    if (SCHRO_FRAME_IS_PACKED(output_picture->format)) {
      schro_frame_convert (decoder->planar_output_frame, frame);
      schro_frame_convert (output_picture, decoder->planar_output_frame);
    } else {
      schro_frame_convert (output_picture, frame);
    }

    if (SCHRO_PARSE_CODE_IS_INTER(decoder->code)) {
      schro_frame_add (decoder->frame, decoder->mc_tmp_frame);
    }
  }
  output_picture->frame_number = decoder->picture_number;

  if (SCHRO_PARSE_CODE_IS_REFERENCE(decoder->code)) {
    SchroFrame *ref;
    SchroFrameFormat frame_format;

    switch (params->video_format->chroma_format) {
      case SCHRO_CHROMA_420:
        frame_format = SCHRO_FRAME_FORMAT_U8_420;
        break;
      case SCHRO_CHROMA_422:
        frame_format = SCHRO_FRAME_FORMAT_U8_422;
        break;
      case SCHRO_CHROMA_444:
        frame_format = SCHRO_FRAME_FORMAT_U8_444;
        break;
      default:
        SCHRO_ASSERT(0);
    }
    ref = schro_frame_new_and_alloc (frame_format,
        decoder->video_format.width, decoder->video_format.height);
    schro_frame_convert (ref, decoder->frame);
    ref->frame_number = decoder->picture_number;
    schro_decoder_reference_add (decoder, schro_upsampled_frame_new(ref),
        decoder->picture_number);
  }

  for(i=0;i<decoder->n_retire;i++){
    schro_decoder_reference_retire (decoder, decoder->retire_list[i]);
  }

  schro_buffer_unref (decoder->input_buffer);
  decoder->input_buffer = NULL;
  schro_bits_free (decoder->bits);

  SCHRO_DEBUG("adding %d to queue", output_picture->frame_number);
  schro_queue_add (decoder->frame_queue, output_picture,
      output_picture->frame_number);

  return SCHRO_DECODER_OK;
}

static void
schro_decoder_init (SchroDecoder *decoder)
{
  SchroFrameFormat frame_format;
  SchroVideoFormat *video_format = &decoder->video_format;
  int frame_width, frame_height;

  frame_format = schro_params_get_frame_format (16,
      video_format->chroma_format);
  frame_width = ROUND_UP_POW2(video_format->width,
      SCHRO_MAX_TRANSFORM_DEPTH + video_format->chroma_h_shift);
  frame_height = ROUND_UP_POW2(video_format->height,
      SCHRO_MAX_TRANSFORM_DEPTH + video_format->chroma_v_shift);

  decoder->mc_tmp_frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);
  decoder->frame = schro_frame_new_and_alloc (frame_format,
      frame_width, frame_height);

  frame_format = schro_params_get_frame_format (8,
      video_format->chroma_format);
  decoder->planar_output_frame = schro_frame_new_and_alloc (frame_format,
      video_format->width, video_format->height);
  SCHRO_DEBUG("planar output frame %dx%d",
      video_format->width, video_format->height);
}

#if 0
void
schro_decoder_iwt_transform (SchroDecoder *decoder, int component)
{
  SchroParams *params = &decoder->params;
  int16_t *frame_data;
  int height;
  int width;
  int level;
  int stride;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }

  frame_data = (int16_t *)decoder->frame->components[component].data;
  stride = decoder->frame->components[component].stride;
  for(level=params->transform_depth-1;level>=0;level--) {
    int w;
    int h;
    int s;

    w = width >> level;
    h = height >> level;
    s = stride << level;

    schro_wavelet_inverse_transform_2d (params->wavelet_filter_index,
        frame_data, s, w, h, decoder->tmpbuf);
  }

}
#endif

void
schro_decoder_decode_parse_header (SchroDecoder *decoder)
{
  int v1, v2, v3, v4;
  
  v1 = schro_bits_decode_bits (decoder->bits, 8);
  v2 = schro_bits_decode_bits (decoder->bits, 8);
  v3 = schro_bits_decode_bits (decoder->bits, 8);
  v4 = schro_bits_decode_bits (decoder->bits, 8);
  SCHRO_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    SCHRO_ERROR ("expected parse header");
    return;
  }

  decoder->code = schro_bits_decode_bits (decoder->bits, 8);
  SCHRO_DEBUG ("parse code %02x", decoder->code);

  decoder->n_refs = SCHRO_PARSE_CODE_NUM_REFS(decoder->code);
  SCHRO_DEBUG("n_refs %d", decoder->n_refs);

  decoder->next_parse_offset = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

void
schro_decoder_decode_access_unit (SchroDecoder *decoder)
{
  int bit;
  int index;
  SchroVideoFormat *format = &decoder->video_format;

  SCHRO_DEBUG("decoding access unit");
  /* parse parameters */
  decoder->au_picture_number = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG("au picture number = %d", decoder->au_picture_number);
  decoder->major_version = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("major_version = %d", decoder->major_version);
  decoder->minor_version = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("minor_version = %d", decoder->minor_version);
  decoder->profile = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("profile = %d", decoder->profile);
  decoder->level = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("level = %d", decoder->level);

  if (decoder->major_version != 0 || decoder->minor_version != 11) {
    SCHRO_ERROR("Expecting version number 0.11, got %d.%d",
        decoder->major_version, decoder->minor_version);
    //SCHRO_MILD_ASSERT(0);
  }
  if (decoder->profile != 0 || decoder->level != 0) {
    SCHRO_ERROR("Expecting profile/level 0.0, got %d.%d",
        decoder->profile, decoder->level);
    SCHRO_MILD_ASSERT(0);
  }

  /* sequence parameters */
  index = schro_bits_decode_uint (decoder->bits);
  schro_params_set_video_format (format, index);

  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    format->width = schro_bits_decode_uint (decoder->bits);
    format->height = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("size = %d x %d", format->width, format->height);

  /* chroma format */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    format->chroma_format = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("chroma_format %d", format->chroma_format);

  /* video depth */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    format->video_depth = schro_bits_decode_uint (decoder->bits);
  }

  /* source parameters */
  /* scan format */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    format->interlaced = schro_bits_decode_bit (decoder->bits);
    if (format->interlaced) {
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        format->top_field_first = schro_bits_decode_bit (decoder->bits);
      }
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        format->sequential_fields = schro_bits_decode_bit (decoder->bits);
      }
    }
  }
  SCHRO_DEBUG("interlaced %d top_field_first %d sequential_fields %d",
      format->interlaced, format->top_field_first,
      format->sequential_fields);

  /* frame rate */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      format->frame_rate_numerator = schro_bits_decode_uint (decoder->bits);
      format->frame_rate_denominator = schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_frame_rate (format, index);
    }
  }
  SCHRO_DEBUG("frame rate %d/%d", format->frame_rate_numerator,
      format->frame_rate_denominator);

  /* aspect ratio */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      format->aspect_ratio_numerator =
        schro_bits_decode_uint (decoder->bits);
      format->aspect_ratio_denominator =
        schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_aspect_ratio (format, index);
    }
  }
  SCHRO_DEBUG("aspect ratio %d/%d", format->aspect_ratio_numerator,
      format->aspect_ratio_denominator);

  /* clean area */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    format->clean_width = schro_bits_decode_uint (decoder->bits);
    format->clean_height = schro_bits_decode_uint (decoder->bits);
    format->left_offset = schro_bits_decode_uint (decoder->bits);
    format->top_offset = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("clean offset %d %d", format->left_offset,
      format->top_offset);
  SCHRO_DEBUG("clean size %d %d", format->clean_width,
      format->clean_height);

  /* signal range */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      format->luma_offset = schro_bits_decode_uint (decoder->bits);
      format->luma_excursion = schro_bits_decode_uint (decoder->bits);
      format->chroma_offset = schro_bits_decode_uint (decoder->bits);
      format->chroma_excursion =
        schro_bits_decode_uint (decoder->bits);
    } else {
      if (index <= SCHRO_PARAMS_MAX_SIGNAL_RANGE) {
        schro_params_set_signal_range (format, index);
      } else {
        schro_decoder_error (decoder, "signal range index out of range");
      }
    }
  }
  SCHRO_DEBUG("luma offset %d excursion %d", format->luma_offset,
      format->luma_excursion);
  SCHRO_DEBUG("chroma offset %d excursion %d", format->chroma_offset,
      format->chroma_excursion);

  /* colour spec */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      /* colour primaries */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        format->colour_primaries = schro_bits_decode_uint (decoder->bits);
      }
      /* colour matrix */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        format->colour_matrix = schro_bits_decode_uint (decoder->bits);
      }
      /* transfer function */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        format->transfer_function = schro_bits_decode_uint (decoder->bits);
      }
    } else {
      if (index <= SCHRO_PARAMS_MAX_COLOUR_SPEC) {
        schro_params_set_colour_spec (format, index);
      } else {
        schro_decoder_error (decoder, "colour spec index out of range");
      }
    }
  }

  schro_params_validate (format);
}

void
schro_decoder_decode_picture_header (SchroDecoder *decoder)
{
  int i;

  schro_bits_sync(decoder->bits);

  decoder->picture_number = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG("picture number %d", decoder->picture_number);

  if (decoder->n_refs > 0) {
    decoder->reference1 = decoder->picture_number +
      schro_bits_decode_sint (decoder->bits);
    SCHRO_DEBUG("ref1 %d", decoder->reference1);
  }

  if (decoder->n_refs > 1) {
    decoder->reference2 = decoder->picture_number +
      schro_bits_decode_sint (decoder->bits);
    SCHRO_DEBUG("ref2 %d", decoder->reference2);
  }

  decoder->n_retire = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("n_retire %d", decoder->n_retire);

  SCHRO_ASSERT(decoder->n_retire <= SCHRO_MAX_REFERENCE_FRAMES);

  for(i=0;i<decoder->n_retire;i++){
    int offset;
    offset = schro_bits_decode_sint (decoder->bits);
    decoder->retire_list[i] = decoder->picture_number + offset;
    SCHRO_DEBUG("retire %d", decoder->picture_number + offset);
  }
}

void
schro_decoder_decode_frame_prediction (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  int bit;
  int index;

  /* block params flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      params->xblen_luma = schro_bits_decode_uint (decoder->bits);
      params->yblen_luma = schro_bits_decode_uint (decoder->bits);
      params->xbsep_luma = schro_bits_decode_uint (decoder->bits);
      params->ybsep_luma = schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_block_params (params, index);
    }
  }
  SCHRO_DEBUG("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  /* mv precision flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->mv_precision = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("mv_precision %d", params->mv_precision);

  /* global motion flag */
  params->have_global_motion = schro_bits_decode_bit (decoder->bits);
  if (params->have_global_motion) {
    int i;

    for (i=0;i<params->num_refs;i++) {
      SchroGlobalMotion *gm = params->global_motion + i;

      /* pan/tilt */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        gm->b0 = schro_bits_decode_sint (decoder->bits);
        gm->b1 = schro_bits_decode_sint (decoder->bits);
      } else {
        gm->b0 = 0;
        gm->b1 = 0;
      }

      /* matrix */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        gm->a_exp = schro_bits_decode_uint (decoder->bits);
        gm->a00 = schro_bits_decode_sint (decoder->bits);
        gm->a01 = schro_bits_decode_sint (decoder->bits);
        gm->a10 = schro_bits_decode_sint (decoder->bits);
        gm->a11 = schro_bits_decode_sint (decoder->bits);
      } else {
        gm->a_exp = 0;
        gm->a00 = 1;
        gm->a01 = 0;
        gm->a10 = 0;
        gm->a11 = 1;
      }

      /* perspective */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        gm->c_exp = schro_bits_decode_uint (decoder->bits);
        gm->c0 = schro_bits_decode_sint (decoder->bits);
        gm->c1 = schro_bits_decode_sint (decoder->bits);
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

  /* picture prediction mode */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->picture_pred_mode = schro_bits_decode_uint (decoder->bits);
  }

  /* non-default picture weights */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->picture_weight_bits = schro_bits_decode_uint (decoder->bits);
    if (params->num_refs > 0) {
      params->picture_weight_1 = schro_bits_decode_sint (decoder->bits);
    }
    if (params->num_refs > 1) {
      params->picture_weight_2 = schro_bits_decode_sint (decoder->bits);
    }
  }
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
schro_decoder_decode_prediction_data (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  SchroArith *arith[9];
  int i, j;

  schro_bits_sync (decoder->bits);

  memset(decoder->motion_field->motion_vectors, 0,
      sizeof(SchroMotionVector)*params->y_num_blocks*params->x_num_blocks);

  for(i=0;i<9;i++){
    SchroBuffer *buffer;
    int length;

    if (params->num_refs < 2 && (i == SCHRO_DECODER_ARITH_VECTOR_REF2_X ||
          i == SCHRO_DECODER_ARITH_VECTOR_REF2_Y)) {
      arith[i] = NULL;
      continue;
    }
    length = schro_bits_decode_uint (decoder->bits);
    schro_bits_sync (decoder->bits);
    buffer = schro_buffer_new_subbuffer (decoder->bits->buffer,
        schro_bits_get_offset (decoder->bits), length);

    arith[i] = schro_arith_new ();
    schro_arith_decode_init (arith[i], buffer);

    schro_bits_skip (decoder->bits, length);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      schro_decoder_decode_macroblock(decoder, arith, i, j);
    }
  }

  for(i=0;i<9;i++) {
    if (arith[i] == NULL) continue;

    if (arith[i]->offset < arith[i]->buffer->length) {
      SCHRO_WARNING("arith decoding %d didn't consume buffer (%d < %d)", i,
          arith[i]->offset, arith[i]->buffer->length);
    }
    if (arith[i]->offset > arith[i]->buffer->length + 6) {
      SCHRO_ERROR("arith decoding %d overran buffer (%d > %d)", i,
          arith[i]->offset, arith[i]->buffer->length);
    }
    schro_buffer_unref (arith[i]->buffer);
    schro_arith_free (arith[i]);
  }
}

static void
schro_decoder_decode_macroblock(SchroDecoder *decoder, SchroArith **arith,
    int i, int j)
{
  SchroParams *params = &decoder->params;
  SchroMotionVector *mv = &decoder->motion_field->motion_vectors[j*params->x_num_blocks + i];
  int k,l;
  int split_prediction;
  int a;

  split_prediction = schro_motion_split_prediction (decoder->motion_field->motion_vectors,
      params, i, j);
#if 0
  mv->split = (split_prediction + _schro_arith_decode_uint (arith,
        SCHRO_CTX_SB_F1, SCHRO_CTX_SB_DATA))%3;
#endif
  a = _schro_arith_decode_uint (arith[SCHRO_DECODER_ARITH_SUPERBLOCK],
        SCHRO_CTX_SB_F1, SCHRO_CTX_SB_DATA);
  mv->split = (split_prediction + a)%3;

  switch (mv->split) {
    case 0:
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_field->motion_vectors, i, j);
      mv[1] = mv[0];
      mv[2] = mv[0];
      mv[3] = mv[0];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 2*params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 3*params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 1:
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_field->motion_vectors, i, j);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_field->motion_vectors, i + 2, j);
      mv[3] = mv[2];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));

      mv += 2*params->x_num_blocks;
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_field->motion_vectors, i, j + 2);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_field->motion_vectors, i + 2, j + 2);
      mv[3] = mv[2];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 2:
      for (l=0;l<4;l++) {
        for (k=0;k<4;k++) {
          schro_decoder_decode_prediction_unit (decoder, arith,
              decoder->motion_field->motion_vectors, i + k, j + l);
        }
      }
      break;
    default:
      SCHRO_ERROR("mv->split == %d, split_prediction %d, a %d", mv->split, split_prediction, a);
      SCHRO_ASSERT(0);
  }
}

static void
schro_decoder_decode_prediction_unit(SchroDecoder *decoder, SchroArith **arith,
    SchroMotionVector *motion_vectors, int x, int y)
{
  SchroParams *params = &decoder->params;
  SchroMotionVector *mv = &motion_vectors[y*params->x_num_blocks + x];

  mv->pred_mode = schro_motion_get_mode_prediction (decoder->motion_field,
      x, y);
  mv->pred_mode ^= 
    _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
        SCHRO_CTX_BLOCK_MODE_REF1);
  if (params->num_refs > 1) {
    mv->pred_mode ^=
      _schro_arith_decode_bit (arith[SCHRO_DECODER_ARITH_PRED_MODE],
          SCHRO_CTX_BLOCK_MODE_REF2) << 1;
  }

  if (mv->pred_mode == 0) {
    int pred[3];
    SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

    schro_motion_dc_prediction (motion_vectors, &decoder->params, x, y, pred);

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
    int pred_x, pred_y;

    if (params->have_global_motion) {
      int pred;
      schro_motion_field_get_global_prediction (decoder->motion_field,
          x, y, &pred);
      mv->using_global = pred ^ _schro_arith_decode_bit (
          arith[SCHRO_DECODER_ARITH_PRED_MODE], SCHRO_CTX_GLOBAL_BLOCK);
    } else {
      mv->using_global = FALSE;
    }
    if (!mv->using_global) {
      if (mv->pred_mode & 1) {
        schro_motion_vector_prediction (motion_vectors, &decoder->params, x, y,
            &pred_x, &pred_y, 1);

        mv->x1 = pred_x + (_schro_arith_decode_sint (
              arith[SCHRO_DECODER_ARITH_VECTOR_REF1_X],
              SCHRO_CTX_MV_REF1_H_CONT_BIN1, SCHRO_CTX_MV_REF1_H_VALUE,
              SCHRO_CTX_MV_REF1_H_SIGN)<<(3-params->mv_precision));
        mv->y1 = pred_y + (_schro_arith_decode_sint (
              arith[SCHRO_DECODER_ARITH_VECTOR_REF1_Y],
              SCHRO_CTX_MV_REF1_V_CONT_BIN1, SCHRO_CTX_MV_REF1_V_VALUE,
              SCHRO_CTX_MV_REF1_V_SIGN)<<(3-params->mv_precision));
      }
      if (mv->pred_mode & 2) {
        schro_motion_vector_prediction (motion_vectors, &decoder->params, x, y,
            &pred_x, &pred_y, 2);

        mv->x2 = pred_x + (_schro_arith_decode_sint (
              arith[SCHRO_DECODER_ARITH_VECTOR_REF2_X],
              SCHRO_CTX_MV_REF2_H_CONT_BIN1, SCHRO_CTX_MV_REF2_H_VALUE,
              SCHRO_CTX_MV_REF2_H_SIGN)<<(3-params->mv_precision));
        mv->y2 = pred_y + (_schro_arith_decode_sint (
              arith[SCHRO_DECODER_ARITH_VECTOR_REF2_Y],
              SCHRO_CTX_MV_REF2_V_CONT_BIN1, SCHRO_CTX_MV_REF2_V_VALUE,
              SCHRO_CTX_MV_REF2_V_SIGN)<<(3-params->mv_precision));
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
schro_decoder_decode_transform_parameters (SchroDecoder *decoder)
{
  int bit;
  SchroParams *params = &decoder->params;

  params->wavelet_filter_index = SCHRO_WAVELET_DESL_9_3;
  params->transform_depth = 4;

  /* transform */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->wavelet_filter_index = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->transform_depth = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG ("transform depth %d", params->transform_depth);

  /* spatial partitioning */
  params->spatial_partition_flag = schro_bits_decode_bit (decoder->bits);
  SCHRO_DEBUG ("spatial_partitioning %d", params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    params->nondefault_partition_flag = schro_bits_decode_bit (decoder->bits);
    if (params->nondefault_partition_flag) {
      int i;
      for(i=0;i<params->transform_depth + 1;i++) {
        params->horiz_codeblocks[i] = schro_bits_decode_uint (decoder->bits);
        params->vert_codeblocks[i] = schro_bits_decode_uint (decoder->bits);
      }
    }
    params->codeblock_mode_index = schro_bits_decode_uint (decoder->bits);
  }
}

void
schro_decoder_decode_transform_data (SchroDecoder *decoder)
{
  int i;
  int component;
  SchroParams *params = &decoder->params;

  for(component=0;component<3;component++){
    for(i=0;i<1+3*params->transform_depth;i++) {
      if (i != 0) schro_bits_sync (decoder->bits);
      schro_decoder_decode_subband (decoder, component, i);
    }
  }
}

#if 0
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
#endif

void
codeblock_line_decode_generic (int16_t *p, int16_t *prev_line, int j, int xmin, int xmax,
    int16_t *parent_data, int position,
    SchroArith *arith, int quant_factor,
    int quant_offset)
{
  int i;

  p += xmin;
  for(i=xmin;i<xmax;i++){
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
    if (j>0) nhood_or |= prev_line[i];
    if (i>0) nhood_or |= p[-1];
    if (i>0 && j>0) nhood_or |= prev_line[i-1];

    previous_value = 0;
    if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(position)) {
      if (i > 0) previous_value = p[-1];
    } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(position)) {
      if (j > 0) previous_value = prev_line[i];
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
    v = _schro_arith_decode_uint (arith, cont_context, \
        value_context); \
    if (v) { \
      v = (quant_offset + quant_factor * v + 2)>>2; \
      if (_schro_arith_decode_bit (arith, sign_context)) { \
        v = -v; \
      } \
      p[0] = v; \
    } else { \
      p[0] = 0; \
    } \
    p++; \
  } while(0)

    STUFF;
  }
}

void
codeblock_line_decode_p_horiz (int16_t *p, int j, int xmin, int xmax,
    int16_t *parent_data, SchroArith *arith, int quant_factor,
    int quant_offset, int16_t *prev)
{
  int i;

  p += xmin;
  if (xmin == 0) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    i = 0;
    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    xmin++;
  }
  for(i=xmin;i<xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= p[-1];
    nhood_or |= prev[i-1];

    previous_value = 0;
    previous_value = p[-1];

    STUFF;
  }
}

void
codeblock_line_decode_p_vert (int16_t *p, int j, int xmin, int xmax,
    int16_t *parent_data, SchroArith *arith, int quant_factor,
    int quant_offset, int16_t *prev)
{
  int i;

  p += xmin;
  if (xmin == 0) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    i = 0;

    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = prev[i];

    STUFF;
    xmin++;
  }
  for(i=xmin;i<xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= p[-1];
    nhood_or |= prev[i-1];

    previous_value = prev[i];

    STUFF;
  }
}

void
codeblock_line_decode_p_diag (int16_t *p, int j, int xmin, int xmax,
    int16_t *parent_data, SchroArith *arith, int quant_factor,
    int quant_offset, int16_t *prev)
{
  int i;

  p += xmin;
  if (xmin == 0) {
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    i = 0;
    parent = parent_data[(i>>1)];
    nhood_or = prev[i];
    previous_value = 0;

    STUFF;
    xmin++;
  }
  for(i=xmin;i<xmax;i++){
    int v;
    int parent;
    int nhood_or;
    int previous_value;

    parent = parent_data[(i>>1)];

    nhood_or = prev[i];
    nhood_or |= p[-1];
    nhood_or |= prev[i-1];
    previous_value = 0;

    STUFF;
  }
}

void
schro_decoder_subband_dc_predict (int16_t *data, int stride, int width,
    int height)
{
  int16_t *prev_line;
  int16_t *line;
  int i,j;
  int pred_value;

  line = OFFSET(data, 0);
  for(i=1;i<width;i++){
    pred_value = line[i-1];
    line[i] += pred_value;
  }
  
  for(j=1;j<height;j++){
    line = OFFSET(data, j*stride);
    prev_line = OFFSET(data, (j-1)*stride);

    pred_value = prev_line[0];
    line[0] += pred_value;

    for(i=1;i<width;i++){
      pred_value = schro_divide(line[i-1] + prev_line[i] +
          prev_line[i-1] + 1,3);
      line[i] += pred_value;
    }
  }

}

void
schro_decoder_get_subband (SchroFrame *frame, int component, int position,
    SchroParams *params,
    int16_t **data, int *stride, int *width, int *height)
{
  int shift;
  SchroFrameComponent *comp = &frame->components[component];

  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);

  *stride = comp->stride << shift;
  if (component == 0) {
    *width = params->iwt_luma_width >> shift;
    *height = params->iwt_luma_height >> shift;
  } else {
    *width = params->iwt_chroma_width >> shift;
    *height = params->iwt_chroma_height >> shift;
  }

  *data = comp->data;
  if (position & 2) {
    *data = OFFSET(*data, (*stride)>>1);
  }
  if (position & 1) {
    *data = OFFSET(*data, (*width)*sizeof(int16_t));
  }
}

void
schro_decoder_decode_subband (SchroDecoder *decoder, int component, int index)
{
  SchroParams *params = &decoder->params;
  SchroSubband *subband = decoder->subbands + index;
  SchroSubband *parent_subband = NULL;
  int16_t *data;
  int16_t *line;
  int16_t *parent_data = NULL;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int subband_length;
  int scale_factor;
  int height;
  int width;
  int stride;
  //int offset;
  int x,y;

#if 0
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

  data = (int16_t *)decoder->frame->components[component].data + offset;

  SCHRO_ERROR("subband index=%d %d x %d at offset %d stride=%d data=%p",
      index, width, height, offset, stride, data);
#endif

  schro_decoder_get_subband (decoder->frame, component, subband->position,
      params, &data, &stride, &width, &height);

  SCHRO_DEBUG("subband index=%d %d x %d stride=%d data=%p",
      index, width, height, stride, data);

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

  subband_length = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("subband %d %d length %d", component, index, subband_length);

  if (subband_length > 0) {
    int i,j;
    SchroBuffer *buffer;
    SchroArith *arith;
    int vert_codeblocks;
    int horiz_codeblocks;
    int have_zero_flags;
    int have_quant_offset;

    quant_index = schro_bits_decode_uint (decoder->bits);
    SCHRO_DEBUG("quant index %d", quant_index);
    if ((unsigned int)quant_index > 60) {
      SCHRO_ERROR("quant_index too big (%u > 60)", quant_index);
      return;
    }

    scale_factor = 1<<(params->transform_depth - SCHRO_SUBBAND_SHIFT(subband->position));

    schro_bits_sync (decoder->bits);

    buffer = schro_buffer_new_subbuffer (decoder->bits->buffer,
        schro_bits_get_offset(decoder->bits), subband_length);

    arith = schro_arith_new ();
    schro_arith_decode_init (arith, buffer);

    if (params->spatial_partition_flag) {
      if (index == 0) {
        vert_codeblocks = params->vert_codeblocks[0];
        horiz_codeblocks = params->horiz_codeblocks[0];
      } else {
        vert_codeblocks = params->vert_codeblocks[SCHRO_SUBBAND_SHIFT(subband->position)+1];
        horiz_codeblocks = params->horiz_codeblocks[SCHRO_SUBBAND_SHIFT(subband->position)+1];
      }
    } else {
      vert_codeblocks = 1;
      horiz_codeblocks = 1;
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
        int zero_codeblock;

        if (have_zero_flags) {
          zero_codeblock = _schro_arith_decode_bit (arith,
              SCHRO_CTX_ZERO_CODEBLOCK);
          if (zero_codeblock) {
            for(j=ymin;j<ymax;j++){
              line = OFFSET(data, j*stride);
              for(i=xmin;i<xmax;i++){
                line[i] = 0;
              }
            }
            continue;
          }
        }

        if (have_quant_offset) {
          quant_index += _schro_arith_decode_sint (arith,
              SCHRO_CTX_QUANTISER_CONT, SCHRO_CTX_QUANTISER_VALUE,
              SCHRO_CTX_QUANTISER_SIGN);
        }
        quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
        if (params->num_refs > 0) {
          quant_offset = schro_table_offset_3_8[CLAMP(quant_index,0,60)];
        } else {
          quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];
        }
        //SCHRO_DEBUG("quant factor %d offset %d", quant_factor, quant_offset);

    for(j=ymin;j<ymax;j++){
      int16_t *p = OFFSET(data,j*stride);
      int16_t *parent_line;
      int16_t *prev_line;
      int x,x2;
      if (subband->has_parent) {
        parent_line = OFFSET(parent_data, (j>>1)*(stride<<1));
      } else {
        parent_line = NULL;
      }
      if (j==0) {
        prev_line = schro_zero;
      } else {
        prev_line = OFFSET(data, (j-1)*stride);
      }
      x = xmin;
      while(x < xmax) {
        x2 = xmax;
        if (subband->has_parent) {
          if (SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(subband->position)) {
            codeblock_line_decode_p_horiz (p, j, x, x2, parent_line,
                arith, quant_factor, quant_offset, prev_line);
          } else if (SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(subband->position)) {
            codeblock_line_decode_p_vert (p, j, x, x2, parent_line,
                arith, quant_factor, quant_offset, prev_line);
          } else {
            codeblock_line_decode_p_diag (p, j, x, x2, parent_line,
                arith, quant_factor, quant_offset, prev_line);
          }
        } else {
          codeblock_line_decode_generic (p, prev_line, j, x, x2, parent_line,
              subband->position,
              arith, quant_factor, quant_offset);
        }
        x = x2;
      }
    }
      }
    }
#if 0
    if (arith->offset < buffer->length) {
      SCHRO_ERROR("arith decoding didn't consume buffer (%d < %d)",
          arith->offset, buffer->length);
    }
#endif
#if 0
    if (arith->offset > buffer->length + 6) {
      SCHRO_ERROR("arith decoding overran buffer (%d > %d)",
          arith->offset, buffer->length);
    }
#endif
    schro_arith_free (arith);
    schro_buffer_unref (buffer);

    schro_bits_skip (decoder->bits, subband_length);

    if (index == 0 && decoder->n_refs == 0) {
      schro_decoder_subband_dc_predict (data, stride, width, height);
    }
  } else {
    int j;

    SCHRO_DEBUG("subband is zero");
    for(j=0;j<height;j++){
      line = OFFSET(data, j*stride);
      oil_splat_s16_ns (line, schro_zero, width);
    }
  }
}



/* reference pool */

static void
schro_decoder_reference_add (SchroDecoder *decoder, SchroUpsampledFrame *frame,
    SchroPictureNumber picture_number)
{
  SCHRO_DEBUG("adding %d", picture_number);

  if (schro_queue_is_full(decoder->reference_queue)) {
    schro_queue_pop (decoder->reference_queue);
  }
  schro_queue_add (decoder->reference_queue, frame, picture_number);
}

static SchroUpsampledFrame *
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
  decoder->error = TRUE;
  if (!decoder->error_message) {
    decoder->error_message = strdup(s);
  }
}

