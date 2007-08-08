
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>


void
schro_params_init (SchroParams *params, int video_format)
{
  int i;

  params->transform_depth = 4;

  if (params->num_refs == 0) {
    if (video_format < 11) {
      params->wavelet_filter_index = SCHRO_WAVELET_DESL_9_3;
    } else {
      params->wavelet_filter_index = SCHRO_WAVELET_FIDELITY;
    }
  } else {
    if (video_format < 11) {
      params->wavelet_filter_index = SCHRO_WAVELET_5_3;
    } else {
      params->wavelet_filter_index = SCHRO_WAVELET_DESL_9_3;
    }
  }

  switch(video_format) {
    case 0: case 3: case 4: case 5: case 6: case 7: case 8:
      params->xbsep_luma = 8;
      params->xblen_luma = 12;
      params->ybsep_luma = 8;
      params->yblen_luma = 12;
      break;
    case 1: case 2:
      params->xbsep_luma = 4;
      params->xblen_luma = 8;
      params->ybsep_luma = 4;
      params->yblen_luma = 8;
      break;
    case 9:
      params->xbsep_luma = 12;
      params->xblen_luma = 16;
      params->ybsep_luma = 12;
      params->yblen_luma = 16;
      break;
    case 10: case 11: case 12:
      params->xbsep_luma = 16;
      params->xblen_luma = 24;
      params->ybsep_luma = 16;
      params->yblen_luma = 24;
      break;
    default:
      SCHRO_ERROR("schro_params_init called with video_format_index %d",
          video_format);
      SCHRO_ASSERT(0);
  }

  params->mv_precision = 2;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;
  params->picture_weight_bits = 1;

  if (params->num_refs == 0) {
    for(i=0;i<3;i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    for(i=3;i<SCHRO_MAX_TRANSFORM_DEPTH+1;i++){
      params->horiz_codeblocks[i] = 4;
      params->vert_codeblocks[i] = 3;
    }
  } else {
    for(i=0;i<2;i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    params->horiz_codeblocks[2] = 8;
    params->vert_codeblocks[2] = 6;
    for(i=3;i<SCHRO_MAX_TRANSFORM_DEPTH+1;i++){
      params->horiz_codeblocks[i] = 12;
      params->vert_codeblocks[i] = 8;
    }
  }

  /* other initializations */

  params->spatial_partition_flag = TRUE;
  params->nondefault_partition_flag = FALSE;
  params->codeblock_mode_index = 1;
  params->have_global_motion = FALSE;
  params->picture_pred_mode = 0;

}

/*
 * schro_params_validate:
 * @format: pointer to a SchroVideoFormat structure
 *
 * Checks the video format structure pointed to by @format for
 * inconsistencies.
 *
 * Returns: TRUE if the contents of @format is valid
 */
int
schro_params_validate (SchroVideoFormat *format)
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

/**
 * schro_params_calculate_iwt_sizes:
 * @params: pointer to @SchroParams structure
 *
 * Calculates the size of the array used for wavelet transformation
 * using the current video format and transformation depth in the
 * @params structure.  The @params structure is updated with the new
 * values.
 *
 * The structure fields changed are: iwt_chroma_width, iwt_chroma_height,
 * iwt_luma_width, iwt_luma_height.
 */
void
schro_params_calculate_iwt_sizes (SchroParams *params)
{
  SchroVideoFormat *video_format = params->video_format;

  params->iwt_chroma_width =
    ROUND_UP_POW2(video_format->chroma_width,params->transform_depth);
  params->iwt_chroma_height =
    ROUND_UP_POW2(video_format->chroma_height, params->transform_depth);
  SCHRO_DEBUG ("iwt chroma size %d x %d", params->iwt_chroma_width,
      params->iwt_chroma_height);

  params->iwt_luma_width =
    ROUND_UP_POW2(video_format->width,params->transform_depth);
  params->iwt_luma_height =
    ROUND_UP_POW2(video_format->height,params->transform_depth);
  SCHRO_DEBUG ("iwt luma size %d x %d", params->iwt_luma_width,
      params->iwt_luma_height);
}

/**
 * schro_params_calculate_mc_sizes:
 * @params: pointer to @SchroParams structure
 *
 * Calculates the size of the array used for motion compensation
 * using the current video format and motion compensation paramters
 * in the @params structure.  The @params structure is updated with
 * the new values.
 *
 * The structure fields changed are: x_num_blocks, y_num_blocks,
 * mc_luma_width, mc_luma_height, mc_chroma_width, mc_chroma_height.
 */
void
schro_params_calculate_mc_sizes (SchroParams *params)
{
  SchroVideoFormat *video_format = params->video_format;

  params->x_num_blocks =
    4 * DIVIDE_ROUND_UP(video_format->width, 4*params->xbsep_luma);
  params->y_num_blocks =
    4 * DIVIDE_ROUND_UP(video_format->height, 4*params->ybsep_luma);

  SCHRO_DEBUG("picture %dx%d, num_blocks %dx%d", video_format->width,
      video_format->height, params->x_num_blocks, params->y_num_blocks);

  params->mc_luma_width = params->x_num_blocks * params->xbsep_luma;
  params->mc_luma_height = params->y_num_blocks * params->ybsep_luma;
  params->mc_chroma_width =
    ROUND_UP_SHIFT(params->mc_luma_width, video_format->chroma_h_shift);
  params->mc_chroma_height =
    ROUND_UP_SHIFT(params->mc_luma_height, video_format->chroma_v_shift);

  SCHRO_DEBUG("mc_luma %dx%d, mc_chroma %dx%d",
      params->mc_luma_width, params->mc_luma_height,
      params->mc_chroma_width, params->mc_chroma_height);
}

static SchroVideoFormat
schro_video_formats[] = {
  { 0, /* custom */
    640, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    30, 1, 1, 1,
    640, 480, 0, 0,
    0, 255, 128, 254,
    0, 0, 0 },
  { 1, /* QSIF */
    176, 120, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    176, 120, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { 2, /* QCIF */
    176, 144, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    176, 144, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { 3, /* SIF */
    352, 240, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    352, 240, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { 4, /* CIF */
    352, 288, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    352, 288, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { 5, /* 4SIF */
    704, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    704, 480, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { 6, /* 4CIF */
    704, 576, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    704, 576, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { 7, /* SD480 */
    720, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24000, 1001, 10, 11,
    720, 480, 0, 0,
    16, 235, 128, 244,
    1, 1, 0 },
  { 8, /* SD576 */
    720, 576, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 1, 12, 11,
    720, 576, 0, 0,
    16, 235, 128, 244,
    2, 1, 0 },
  { 9, /* HD720 */
    1280, 720, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    1280, 720, 0, 0,
    16, 235, 128, 244,
    0, 0, 0 },
  { 10, /* HD1080 */
    1920, 1080, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    1920, 1080, 0, 0,
    16, 235, 128, 244,
    0, 0, 0 },
  { 11, /* 2KCinema */
    2048, 1556, SCHRO_CHROMA_444, 16,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    2048, 1536, 
    0, 65535, 32768, 65534,
    3, 2, 3 },
  { 12, /* 4KCinema */
    4096, 3072, SCHRO_CHROMA_444, 16,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    2048, 1536, 
    0, 65535, 32768, 65534,
    3, 2, 3 },
};

/**
 * schro_params_set_video_format:
 * @format:
 * @index:
 *
 * Initializes the video format structure pointed to by @format to
 * the standard Dirac video formats specified by @index.
 */
void
schro_params_set_video_format (SchroVideoFormat *format,
    SchroVideoFormatEnum index)
{
  if (index < 0 || index >= ARRAY_SIZE(schro_video_formats)) {
    SCHRO_ERROR("illegal video format index");
    return;
  }

  memcpy (format, schro_video_formats + index, sizeof(SchroVideoFormat));
}

static int
schro_params_get_video_format_metric (SchroVideoFormat *format, int i)
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
 * schro_params_get_video_format:
 * @format: pointer to SchroVideoFormat structure
 *
 * In Dirac streams, video formats are encoded by specifying a standard
 * format, and then modifying that to get the desired video format.  This
 * function guesses a standard format to use as a starting point for
 * encoding the video format pointed to by @format.
 *
 * FIXME: should rename this function to schro_params_get_std_video_format.
 *
 * FIXME: the function that guesses the best format is poor
 *
 * Returns: an index to the optimal standard format
 */
SchroVideoFormatEnum
schro_params_get_video_format (SchroVideoFormat *format)
{
  int metric;
  int min_index;
  int min_metric;
  int i;

  min_index = 0;
  min_metric = schro_params_get_video_format_metric (format, 0);
  for(i=1;i<ARRAY_SIZE (schro_video_formats); i++) {
    metric = schro_params_get_video_format_metric (format, i);
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
  { 12, 1 },
  { 25, 2 },
  { 15, 1 },
  { 24000, 1001 },
  { 24, 1 },
  { 25, 1 },
  { 30000, 1001 },
  { 30, 1 },
  { 50, 1 },
  { 60000, 1001 },
  { 60, 1 }
};

/**
 * schro_params_set_frame_rate:
 * @format:
 * @index:
 *
 * Sets the frame rate of the video format structure pointed to by
 * @format to the Dirac standard frame specified by @index.
 */
void schro_params_set_frame_rate (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_frame_rates)) {
    SCHRO_ERROR("illegal frame rate index");
    return;
  }

  format->frame_rate_numerator = schro_frame_rates[index].numerator;
  format->frame_rate_denominator = schro_frame_rates[index].denominator;
}

/**
 * schro_params_get_frame_rate:
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
int schro_params_get_frame_rate (SchroVideoFormat *format)
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
  { 59, 54 }
};

/*
 * schro_params_set_aspect_ratio:
 * @format: pointer to a SchroVideoFormat structure
 * @index: index to a standard aspect ratio
 *
 * Sets the pixel aspect ratio of the video format structure pointed to
 * by @format to the standard pixel aspect ratio indicated by @index.
 */
void schro_params_set_aspect_ratio (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_aspect_ratios)) {
    SCHRO_ERROR("illegal pixel aspect ratio index");
    return;
  }

  format->aspect_ratio_numerator = schro_aspect_ratios[index].numerator;
  format->aspect_ratio_denominator = schro_aspect_ratios[index].denominator;

}

/*
 * schro_params_get_aspect_ratio:
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
int schro_params_get_aspect_ratio (SchroVideoFormat *format)
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

typedef struct _SchroSignalRange SchroSignalRange;
struct _SchroSignalRange {
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
};

static const SchroSignalRange schro_signal_ranges[] = {
  { 0, 0, 0, 0 },
  { 0, 255, 128, 255 },
  { 16, 235, 128, 224 },
  { 64, 876, 512, 896 }
};

/**
 * schro_params_set_signal_range:
 * @format:
 * @index:
 *
 * Sets the signal range of the video format structure to one of the
 * standard values indicated by @index.
 */
void schro_params_set_signal_range (SchroVideoFormat *format, int i)
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
 * schro_params_get_signal_range:
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
int schro_params_get_signal_range (SchroVideoFormat *format)
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

typedef struct _SchroColourSpec SchroColourSpec;
struct _SchroColourSpec {
  int colour_primaries;
  int colour_matrix;
  int transfer_function;
};

static const SchroColourSpec schro_colour_specs[] = {
  { 0, 0, 0 },
  { 1, 1, 0 },
  { 2, 1, 0 },
  { 3, 2, 3 }
};

/**
 * schro_params_set_colour_spec:
 * @format: pointer to SchroVideoFormat structure
 * @index: index to standard colour specification
 *
 * Sets the colour specification of the video format structure to one of the
 * standard values indicated by @index.
 */
void schro_params_set_colour_spec (SchroVideoFormat *format, int i)
{
  if (i < 1 || i >= ARRAY_SIZE(schro_colour_specs)) {
    SCHRO_ERROR("illegal signal range index");
    return;
  }

  format->colour_primaries = schro_colour_specs[i].colour_primaries;
  format->colour_matrix = schro_colour_specs[i].colour_matrix;
  format->transfer_function = schro_colour_specs[i].transfer_function;
}

/**
 * schro_params_get_colour_spec:
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
int schro_params_get_colour_spec (SchroVideoFormat *format)
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

typedef struct _SchroBlockParams SchroBlockParams;
struct _SchroBlockParams {
  int xblen_luma;
  int yblen_luma;
  int xbsep_luma;
  int ybsep_luma;
};

static SchroBlockParams
schro_block_params[] = {
  { 0, 0, 0, 0 },
  { 8, 8, 4, 4 },
  { 12, 12, 8, 8 },
  { 16, 16, 12, 12 },
  { 24, 24, 16, 16 }
};

/**
 * schro_params_set_block_params:
 * @params: pointer to SchroParams structure
 * @index: index to standard block parameters
 *
 * Sets the block parameters for motion compensation in the parameters
 * structure pointed to by @params to the
 * standard block parameters given by @index.
 */
void
schro_params_set_block_params (SchroParams *params, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_block_params)) {
    SCHRO_ERROR("illegal block params index");
    return;
  }

  params->xblen_luma = schro_block_params[index].xblen_luma;
  params->yblen_luma = schro_block_params[index].yblen_luma;
  params->xbsep_luma = schro_block_params[index].xbsep_luma;
  params->ybsep_luma = schro_block_params[index].ybsep_luma;
}

int
schro_params_get_block_params (SchroParams *params)
{
  int i;
  for(i=1;i<ARRAY_SIZE(schro_block_params);i++){
    if (schro_block_params[i].xblen_luma == params->xblen_luma && 
        schro_block_params[i].xbsep_luma == params->xbsep_luma &&
        schro_block_params[i].yblen_luma == params->yblen_luma &&
        schro_block_params[i].ybsep_luma == params->ybsep_luma) {
      return i;
    }
  }
  return 0;
}

/**
 * schro_params_set_default_codeblock:
 * @params: pointer to SchroParams structure
 *
 * Sets the codeblock parameters in the parameters structure pointed to
 * by @params to the defaults.
 */
void
schro_params_set_default_codeblock (SchroParams *params)
{
  int i;

  params->spatial_partition_flag = TRUE;
  params->nondefault_partition_flag = FALSE;

  if (params->num_refs == 0) {
    for(i=0;i<3;i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    for(i=3;i<SCHRO_MAX_TRANSFORM_DEPTH+1;i++){
      params->horiz_codeblocks[i] = 4;
      params->vert_codeblocks[i] = 3;
    }
  } else {
    for(i=0;i<2;i++) {
      params->horiz_codeblocks[i] = 1;
      params->vert_codeblocks[i] = 1;
    }
    params->horiz_codeblocks[2] = 8;
    params->vert_codeblocks[2] = 6;
    for(i=3;i<SCHRO_MAX_TRANSFORM_DEPTH+1;i++){
      params->horiz_codeblocks[i] = 12;
      params->vert_codeblocks[i] = 8;
    }
  }
  params->codeblock_mode_index = 1;

}

/**
 * schro_params_init_subbands:
 * @params: pointer to SchroParams structure
 * @subbands: pointer to array of SchroSubband structures
 *
 * Initializes the array of subband structures based on the values in the
 * @params structure.
 *
 */
void
schro_params_init_subbands (SchroParams *params, SchroSubband *subbands,
    int luma_frame_stride, int chroma_frame_stride)
{
  int i;
  int w;
  int h;
  int stride;
  int chroma_w;
  int chroma_h;
  int chroma_stride;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = luma_frame_stride << params->transform_depth;
  chroma_w = params->iwt_chroma_width >> params->transform_depth;
  chroma_h = params->iwt_chroma_height >> params->transform_depth;
  chroma_stride = chroma_frame_stride << params->transform_depth;

  subbands[0].position = 0;
  subbands[0].has_parent = 0;

  for(i=0; i<params->transform_depth; i++) {
    /* hl */
    subbands[1+3*i].position = 1 | (i<<2);
    subbands[1+3*i].has_parent = (i>0);

    /* lh */
    subbands[2+3*i].position = 2 | (i<<2);
    subbands[2+3*i].has_parent = (i>0);

    /* hh */
    subbands[3+3*i].position = 3 | (i<<2);
    subbands[3+3*i].has_parent = (i>0);

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
schro_subband_get_frame_component (SchroFrameComponent *dest,
    SchroFrameComponent *full_frame, int position)
{
  int shift = (position>>2) + 1;

  dest->stride = full_frame->stride << shift;
  dest->width = full_frame->width >> shift;
  dest->height = full_frame->height >> shift;

  if (position & 2) {
    dest->data = OFFSET(full_frame->data, (dest->stride>>1) * sizeof(int16_t));
  }
  if (position & 1) {
    dest->data = OFFSET(full_frame->data, dest->width * sizeof(int16_t));
  }

}

void
schro_subband_get (SchroFrame *frame, int component, int position,
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

int
schro_params_get_frame_format (int depth, SchroChromaFormat chroma_format)
{
  if (depth == 8) {
    switch (chroma_format) {
      case SCHRO_CHROMA_444:
        return SCHRO_FRAME_FORMAT_U8_444;
      case SCHRO_CHROMA_422:
        return SCHRO_FRAME_FORMAT_U8_422;
      case SCHRO_CHROMA_420:
        return SCHRO_FRAME_FORMAT_U8_420;
    }
  } else if (depth == 16) {
    switch (chroma_format) {
      case SCHRO_CHROMA_444:
        return SCHRO_FRAME_FORMAT_S16_444;
      case SCHRO_CHROMA_422:
        return SCHRO_FRAME_FORMAT_S16_422;
      case SCHRO_CHROMA_420:
        return SCHRO_FRAME_FORMAT_S16_420;
    }
  }

  SCHRO_ASSERT(0);
}

