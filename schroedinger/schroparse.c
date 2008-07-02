
#include "config.h"

#include <schroedinger/schroparse.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schrounpack.h>
#include <schroedinger/schrodecoder.h>

int
schro_parse_decode_sequence_header (uint8_t *data, int length,
    SchroVideoFormat *format)
{
  int bit;
  int index;
  SchroUnpack u;
  SchroUnpack *unpack = &u;
  int major_version, minor_version;
  int profile, level;
  
  SCHRO_DEBUG("decoding sequence header");

  schro_unpack_init_with_data (unpack, data, length, 1);

  /* parse parameters */
  major_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("major_version = %d", major_version);
  minor_version = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("minor_version = %d", minor_version);
  profile = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("profile = %d", profile);
  level = schro_unpack_decode_uint (unpack);
  SCHRO_DEBUG("level = %d", level);
  
#if 0
  if (!schro_decoder_check_version (major_version, minor_version)) {
    SCHRO_WARNING("Stream version number %d:%d not handled, expecting 0:20071203, 1:0, 2:0, or 2:1",
        major_version, minor_version);
  }
#endif
  if (profile != 0 || level != 0) {
    SCHRO_WARNING("Expecting profile/level 0:0, got %d:%d",
        profile, level);
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

  /* frame rate */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
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

  /* aspect ratio */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
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

  /* signal range */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
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
        return FALSE;
      }
    }
  }
  SCHRO_DEBUG("luma offset %d excursion %d", format->luma_offset,
      format->luma_excursion);
  SCHRO_DEBUG("chroma offset %d excursion %d", format->chroma_offset,
      format->chroma_excursion);

  /* colour spec */
  bit = schro_unpack_decode_bit (unpack);
  if (bit) {
    index = schro_unpack_decode_uint (unpack);
    if (index <= SCHRO_COLOUR_SPEC_CINEMA) {
      schro_video_format_set_std_colour_spec (format, index);
    } else {
      return FALSE;
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

  format->interlaced_coding = schro_unpack_decode_uint (unpack);

  schro_video_format_validate (format);

  return TRUE;
}

