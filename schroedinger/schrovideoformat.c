
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>


/*
 * schro_video_format_validate:
 * @format: pointer to a SchroVideoFormat structure
 *
 * Checks the video format structure pointed to by @format for
 * inconsistencies.
 *
 * Returns: TRUE if the contents of @format is valid
 */
int
schro_video_format_validate (SchroVideoFormat *format)
{
  if (format->aspect_ratio_numerator == 0) {
    SCHRO_ERROR("aspect_ratio_numerator is 0");
    format->aspect_ratio_numerator = 1;
  }
  if (format->aspect_ratio_denominator == 0) {
    SCHRO_ERROR("aspect_ratio_denominator is 0");
    format->aspect_ratio_denominator = 1;
  }

  switch (format->chroma_format) {
    case SCHRO_CHROMA_444:
      format->chroma_width = format->width;
      format->chroma_height = format->height;
      format->chroma_h_shift = 0;
      format->chroma_v_shift = 0;
      break;
    case SCHRO_CHROMA_422:
      format->chroma_width = ROUND_UP_SHIFT(format->width,1);
      format->chroma_height = format->height;
      format->chroma_h_shift = 1;
      format->chroma_v_shift = 0;
      break;
    case SCHRO_CHROMA_420:
      format->chroma_width = ROUND_UP_SHIFT(format->width,1);
      format->chroma_height = ROUND_UP_SHIFT(format->height,1);
      format->chroma_h_shift = 1;
      format->chroma_v_shift = 1;
      break;
  }

  return 1;
}

static SchroVideoFormat
schro_video_formats[] = {
  { 0, /* custom */
    640, 480, SCHRO_CHROMA_420,
    FALSE, FALSE,
    24000, 1001, 1, 1,
    640, 480, 0, 0,
    0, 255, 128, 255,
    0, 0, 0 },
  { 1, /* QSIF525 */
    176, 120, SCHRO_CHROMA_420,
    FALSE, FALSE,
    15000, 1001, 10, 11,
    176, 120, 0, 0,
    0, 255, 128, 255,
    1, 1, 0 },
  { 2, /* QCIF */
    176, 144, SCHRO_CHROMA_420,
    FALSE, TRUE,
    25, 2, 12, 11,
    176, 144, 0, 0,
    0, 255, 128, 255,
    2, 1, 0 },
  { 3, /* SIF525 */
    352, 240, SCHRO_CHROMA_420,
    FALSE, FALSE,
    15000, 1001, 10, 11,
    352, 240, 0, 0,
    0, 255, 128, 255,
    1, 1, 0 },
  { 4, /* CIF */
    352, 288, SCHRO_CHROMA_420,
    FALSE, TRUE,
    25, 2, 12, 11,
    352, 288, 0, 0,
    0, 255, 128, 255,
    2, 1, 0 },
  { 5, /* 4SIF525 */
    704, 480, SCHRO_CHROMA_420,
    FALSE, FALSE,
    15000, 1001, 10, 11,
    704, 480, 0, 0,
    0, 255, 128, 255,
    1, 1, 0 },
  { 6, /* 4CIF */
    704, 576, SCHRO_CHROMA_420,
    FALSE, TRUE,
    25, 2, 12, 11,
    704, 576, 0, 0,
    0, 255, 128, 255,
    2, 1, 0 },
  { 7, /* SD480I-60 */
    720, 480, SCHRO_CHROMA_422,
    TRUE, FALSE,
    30000, 1001, 10, 11,
    704, 480, 8, 0,
    64, 876, 512, 896,
    1, 1, 0 },
  { 8, /* SD576I-50 */
    720, 576, SCHRO_CHROMA_422,
    TRUE, TRUE,
    25, 1, 12, 11,
    704, 576, 8, 0,
    64, 876, 512, 896,
    2, 1, 0 },
  { 9, /* HD720P-60 */
    1280, 720, SCHRO_CHROMA_422,
    FALSE, TRUE,
    60000, 1001, 1, 1,
    1280, 720, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 10, /* HD720P-50 */
    1280, 720, SCHRO_CHROMA_422,
    FALSE, TRUE,
    50, 1, 1, 1,
    1280, 720, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 11, /* HD1080I-60 */
    1920, 1080, SCHRO_CHROMA_422,
    TRUE, TRUE,
    30000, 1001, 1, 1,
    1920, 1080, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 12, /* HD1080I-50 */
    1920, 1080, SCHRO_CHROMA_422,
    TRUE, TRUE,
    25, 1, 1, 1,
    1920, 1080, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 13, /* HD1080P-60 */
    1920, 1080, SCHRO_CHROMA_422,
    FALSE, TRUE,
    60000, 1001, 1, 1,
    1920, 1080, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 14, /* HD1080P-50 */
    1920, 1080, SCHRO_CHROMA_422,
    FALSE, TRUE,
    50, 1, 1, 1,
    1920, 1080, 0, 0,
    64, 876, 512, 896,
    0, 0, 0 },
  { 15, /* DC2K */
    2048, 1080, SCHRO_CHROMA_444,
    FALSE, TRUE,
    24, 1, 1, 1,
    2048, 1080, 0, 0,
    256, 3504, 2048, 3584,
    3, 0, 0 },
  { 16, /* DC4K */
    4096, 2160, SCHRO_CHROMA_444,
    FALSE, TRUE,
    24, 1, 1, 1,
    2048, 1536, 0, 0,
    256, 3504, 2048, 3584,
    3, 0, 0 },
};

/**
 * schro_video_format_set_std_video_format:
 * @format:
 * @index:
 *
 * Initializes the video format structure pointed to by @format to
 * the standard Dirac video formats specified by @index.
 */
void
schro_video_format_set_std_video_format (SchroVideoFormat *format,
    SchroVideoFormatEnum index)
{
  if (index < 0 || index >= ARRAY_SIZE(schro_video_formats)) {
    SCHRO_ERROR("illegal video format index");
    return;
  }

  memcpy (format, schro_video_formats + index, sizeof(SchroVideoFormat));
}

static int
schro_video_format_get_video_format_metric (SchroVideoFormat *format, int i)
{
  SchroVideoFormat *std_format;
  int metric = 0;

  std_format = schro_video_formats + i;

  if (format->width != std_format->width) {
    metric++;
  }
  if (format->height != std_format->height) {
    metric++;
  }
  if (format->interlaced != std_format->interlaced) {
    metric++;
  }
  if (format->top_field_first != std_format->top_field_first) {
    metric++;
  }
  if (format->frame_rate_numerator != std_format->frame_rate_numerator) {
    metric++;
  }
  if (format->frame_rate_denominator != std_format->frame_rate_denominator) {
    metric++;
  }
  if (format->aspect_ratio_numerator != std_format->aspect_ratio_numerator) {
    metric++;
  }
  if (format->aspect_ratio_denominator != std_format->aspect_ratio_denominator) {
    metric++;
  }
  if (format->left_offset != std_format->left_offset) {
    metric++;
  }
  if (format->top_offset != std_format->top_offset) {
    metric++;
  }
  if (format->clean_width != std_format->clean_width) {
    metric++;
  }
  if (format->clean_height != std_format->clean_height) {
    metric++;
  }
  if (format->colour_matrix != std_format->colour_matrix) {
    metric++;
  }
  if (format->colour_primaries != std_format->colour_primaries) {
    metric++;
  }
  if (format->transfer_function != std_format->transfer_function) {
    metric++;
  }

  return metric;
}

/**
 * schro_video_format_get_std_video_format:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac streams, video formats are encoded by specifying a standard
 * format, and then modifying that to get the desired video format.  This
 * function guesses a standard format to use as a starting point for
 * encoding the video format pointed to by @format.
 *
 * FIXME: the function that guesses the best format is poor
 *
 * Returns: an index to the optimal standard format
 */
SchroVideoFormatEnum
schro_video_format_get_std_video_format (SchroVideoFormat *format)
{
  int metric;
  int min_index;
  int min_metric;
  int i;

  min_index = 0;
  min_metric = schro_video_format_get_video_format_metric (format, 0);
  for(i=1;i<ARRAY_SIZE (schro_video_formats); i++) {
    metric = schro_video_format_get_video_format_metric (format, i);
    if (metric < min_metric) {
      min_index = i;
      min_metric = metric;
    }
  }
  return min_index;
}

typedef struct _SchroFrameRate SchroFrameRate;
struct _SchroFrameRate {
  int numerator;
  int denominator;
};

static SchroFrameRate
schro_frame_rates[] = {
  { 0, 0 },
  { 24000, 1001 },
  { 24, 1 },
  { 25, 1 },
  { 30000, 1001 },
  { 30, 1 },
  { 50, 1 },
  { 60000, 1001 },
  { 60, 1 },
  { 15000, 1001 },
  { 25, 2 }
};

/**
 * schro_video_format_set_std_frame_rate:
 * @format:
 * @index:
 *
 * Sets the frame rate of the video format structure pointed to by
 * @format to the Dirac standard frame specified by @index.
 */
void schro_video_format_set_std_frame_rate (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_frame_rates)) {
    SCHRO_ERROR("illegal frame rate index");
    return;
  }

  format->frame_rate_numerator = schro_frame_rates[index].numerator;
  format->frame_rate_denominator = schro_frame_rates[index].denominator;
}

/**
 * schro_video_format_get_std_frame_rate:
 * @format:
 *
 * In Dirac bitstreams, frame rates can be one of several standard
 * frame rates, encoded as an index, or the numerator and denominator
 * of the framerate can be encoded directly.  This function looks up
 * the frame rate contained in the video format structure @format in
 * the list of standard frame rates.  If the frame rate is a standard
 * frame rate, the corresponding index is returned, otherwise 0 is
 * returned.
 *
 * Returns: index to a standard Dirac frame rate, or 0 if the frame rate
 * is custom.
 */
int schro_video_format_get_std_frame_rate (SchroVideoFormat *format)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_frame_rates);i++){
    if (format->frame_rate_numerator == schro_frame_rates[i].numerator &&
        format->frame_rate_denominator == schro_frame_rates[i].denominator) {
      return i;
    }
  }

  return 0;
}

typedef struct _SchroPixelAspectRatio SchroPixelAspectRatio;
struct _SchroPixelAspectRatio {
  int numerator;
  int denominator;
};

static const SchroPixelAspectRatio
schro_aspect_ratios[] = {
  { 0, 0 },
  { 1, 1 },
  { 10, 11 },
  { 12, 11 },
  { 40, 33 },
  { 16, 11 },
  { 4, 3 }
};

/*
 * schro_video_format_set_std_aspect_ratio:
 * @format: pointer to a SchroVideoFormat structure
 * @index: index to a standard aspect ratio
 *
 * Sets the pixel aspect ratio of the video format structure pointed to
 * by @format to the standard pixel aspect ratio indicated by @index.
 */
void schro_video_format_set_std_aspect_ratio (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_aspect_ratios)) {
    SCHRO_ERROR("illegal pixel aspect ratio index");
    return;
  }

  format->aspect_ratio_numerator = schro_aspect_ratios[index].numerator;
  format->aspect_ratio_denominator = schro_aspect_ratios[index].denominator;

}

/*
 * schro_video_format_get_std_aspect_ratio:
 * @format: pointer to a SchroVideoFormat structure
 *
 * In Dirac bitstreams, pixel aspect ratios can be one of several standard
 * pixel aspect ratios, encoded as an index, or the numerator and denominator
 * of the pixel aspect ratio can be encoded directly.  This function looks up
 * the pixel aspect ratio contained in the video format structure @format in
 * the list of standard pixel aspect ratios.  If the pixel aspect ratio is
 * a standard pixel aspect ratio, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard pixel aspect ratio, or 0 if there is no
 * corresponding standard pixel aspect ratio.
 */
int schro_video_format_get_std_aspect_ratio (SchroVideoFormat *format)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_aspect_ratios);i++){
    if (format->aspect_ratio_numerator ==
        schro_aspect_ratios[i].numerator &&
        format->aspect_ratio_denominator ==
        schro_aspect_ratios[i].denominator) {
      return i;
    }
  }

  return 0;
}

typedef struct _SchroSignalRangeStruct SchroSignalRangeStruct;
struct _SchroSignalRangeStruct {
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
};

static const SchroSignalRangeStruct schro_signal_ranges[] = {
  { 0, 0, 0, 0 },
  { 0, 255, 128, 255 },
  { 16, 219, 128, 224 },
  { 64, 876, 512, 896 },
  { 256, 3504, 2048, 3584 }
};

/**
 * schro_video_format_set_std_signal_range:
 * @format:
 * @index:
 *
 * Sets the signal range of the video format structure to one of the
 * standard values indicated by @index.
 */
void schro_video_format_set_std_signal_range (SchroVideoFormat *format,
    SchroSignalRange i)
{
  if (i < 1 || i >= ARRAY_SIZE(schro_signal_ranges)) {
    SCHRO_ERROR("illegal signal range index");
    return;
  }

  format->luma_offset = schro_signal_ranges[i].luma_offset;
  format->luma_excursion = schro_signal_ranges[i].luma_excursion;
  format->chroma_excursion = schro_signal_ranges[i].chroma_excursion;
  format->chroma_excursion = schro_signal_ranges[i].chroma_excursion;
}

/**
 * schro_video_format_get_std_signal_range:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac bitstreams, signal ranges can be one of several standard
 * signal ranges, encoded as an index, or the extents of the signal
 * range can be encoded directly.  This function looks up
 * the signal range contained in the video format structure @format in
 * the list of standard signal ranges.  If the signal range is
 * a standard signal range, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard signal range, or 0 if there is no
 * corresponding standard signal range.
 */
SchroSignalRange
schro_video_format_get_std_signal_range (SchroVideoFormat *format)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_signal_ranges);i++){
    if (format->luma_offset == schro_signal_ranges[i].luma_offset &&
        format->luma_excursion == schro_signal_ranges[i].luma_excursion &&
        format->chroma_excursion == schro_signal_ranges[i].chroma_excursion &&
        format->chroma_excursion == schro_signal_ranges[i].chroma_excursion) {
      return i;
    }
  }

  return 0;

}

typedef struct _SchroColourSpecStruct SchroColourSpecStruct;
struct _SchroColourSpecStruct {
  int colour_primaries;
  int colour_matrix;
  int transfer_function;
};

static const SchroColourSpecStruct schro_colour_specs[] = {
  { /* Custom */
    SCHRO_COLOUR_PRIMARY_HDTV,
    SCHRO_COLOUR_MATRIX_HDTV,
    SCHRO_TRANSFER_CHAR_TV_GAMMA
  },
  { /* SDTV 525 */
    SCHRO_COLOUR_PRIMARY_SDTV_525,
    SCHRO_COLOUR_MATRIX_SDTV,
    SCHRO_TRANSFER_CHAR_TV_GAMMA
  },
  { /* SDTV 625 */
    SCHRO_COLOUR_PRIMARY_SDTV_625,
    SCHRO_COLOUR_MATRIX_SDTV,
    SCHRO_TRANSFER_CHAR_TV_GAMMA
  },
  { /* HDTV */
    SCHRO_COLOUR_PRIMARY_HDTV,
    SCHRO_COLOUR_MATRIX_HDTV,
    SCHRO_TRANSFER_CHAR_TV_GAMMA
  },
  { /* Cinema */
    SCHRO_COLOUR_PRIMARY_CINEMA,
    SCHRO_COLOUR_MATRIX_HDTV,
    SCHRO_TRANSFER_CHAR_TV_GAMMA
  }
};

/**
 * schro_video_format_set_std_colour_spec:
 * @format: pointer to SchroVideoFormat structure
 * @index: index to standard colour specification
 *
 * Sets the colour specification of the video format structure to one of the
 * standard values indicated by @index.
 */
void schro_video_format_set_std_colour_spec (SchroVideoFormat *format,
    SchroColourSpec i)
{
  if (i < 0 || i >= ARRAY_SIZE(schro_colour_specs)) {
    SCHRO_ERROR("illegal signal range index");
    return;
  }

  format->colour_primaries = schro_colour_specs[i].colour_primaries;
  format->colour_matrix = schro_colour_specs[i].colour_matrix;
  format->transfer_function = schro_colour_specs[i].transfer_function;
}

/**
 * schro_video_format_get_std_colour_spec:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac bitstreams, colour specifications can be one of several standard
 * colour specifications, encoded as an index, or the individual parts of
 * the colour specication can be encoded.  This function looks up
 * the colour specification contained in the video format structure @format in
 * the list of standard colour specifications.  If the colour specification is
 * a standard colour specification, the corresponding index is returned,
 * otherwise 0 is returned.
 *
 * Returns: index to standard colour specification, or 0 if there is no
 * corresponding standard colour specification.
 */
SchroColourSpec
schro_video_format_get_std_colour_spec (SchroVideoFormat *format)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_colour_specs);i++){
    if (format->colour_primaries == schro_colour_specs[i].colour_primaries &&
        format->colour_matrix == schro_colour_specs[i].colour_matrix &&
        format->transfer_function == schro_colour_specs[i].transfer_function) {
      return i;
    }
  }

  return 0;
}

