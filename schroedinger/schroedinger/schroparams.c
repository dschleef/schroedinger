
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>


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
      format->chroma_h_scale = 1;
      format->chroma_v_scale = 1;
      break;
    case SCHRO_CHROMA_422:
      format->chroma_width = ROUND_UP_SHIFT(format->width,1);
      format->chroma_height = format->height;
      format->chroma_h_scale = 2;
      format->chroma_v_scale = 1;
      break;
    case SCHRO_CHROMA_420:
      format->chroma_width = ROUND_UP_SHIFT(format->width,1);
      format->chroma_height = ROUND_UP_SHIFT(format->height,1);
      format->chroma_h_scale = 2;
      format->chroma_v_scale = 2;
      break;
  }

  return 1;
}

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
  params->mc_chroma_width = params->mc_luma_width / video_format->chroma_h_scale;
  params->mc_chroma_height = params->mc_luma_height / video_format->chroma_v_scale;

  SCHRO_DEBUG("mc_luma %dx%d, mc_chroma %dx%d",
      params->mc_luma_width, params->mc_luma_height,
      params->mc_chroma_width, params->mc_chroma_height);
}

static SchroVideoFormat
schro_video_formats[] = {
  { /* custom */
    640, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    30, 1, 1, 1,
    640, 480, 0, 0,
    0, 255, 128, 254,
    0, 0, 0 },
  { /* QSIF */
    176, 120, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    176, 120, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { /* QCIF */
    176, 144, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    176, 144, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { /* SIF */
    352, 240, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    352, 240, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { /* CIF */
    352, 288, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    352, 288, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { /* 4SIF */
    704, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    15000, 1001, 10, 11,
    704, 480, 0, 0,
    0, 255, 128, 254,
    1, 1, 0 },
  { /* 4CIF */
    704, 576, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 2, 12, 11,
    704, 576, 0, 0,
    0, 255, 128, 254,
    2, 1, 0 },
  { /* SD480 */
    720, 480, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24000, 1001, 10, 11,
    720, 480, 0, 0,
    16, 235, 128, 244,
    1, 1, 0 },
  { /* SD576 */
    720, 576, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    25, 1, 12, 11,
    720, 576, 0, 0,
    16, 235, 128, 244,
    2, 1, 0 },
  { /* HD720 */
    1280, 720, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    1280, 720, 0, 0,
    16, 235, 128, 244,
    0, 0, 0 },
  { /* HD1080 */
    1920, 1080, SCHRO_CHROMA_420, 8,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    1920, 1080, 0, 0,
    16, 235, 128, 244,
    0, 0, 0 },
  { /* 2KCinema */
    2048, 1556, SCHRO_CHROMA_444, 16,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    2048, 1536, 
    0, 65535, 32768, 65534,
    3, 2, 3 },
  { /* 4KCinema */
    4096, 3072, SCHRO_CHROMA_444, 16,
    FALSE, TRUE, FALSE,
    24, 1, 1, 1,
    2048, 1536, 
    0, 65535, 32768, 65534,
    3, 2, 3 },
};

void schro_params_set_video_format (SchroVideoFormat *format, int index)
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
  if (format->interlaced_source != std_format->interlaced_source) {
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

int schro_params_get_video_format (SchroVideoFormat *format)
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


void schro_params_set_frame_rate (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_frame_rates)) {
    SCHRO_ERROR("illegal frame rate index");
    return;
  }

  format->frame_rate_numerator = schro_frame_rates[index].numerator;
  format->frame_rate_denominator = schro_frame_rates[index].denominator;
}

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

void schro_params_set_aspect_ratio (SchroVideoFormat *format, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_aspect_ratios)) {
    SCHRO_ERROR("illegal pixel aspect ratio index");
    return;
  }

  format->aspect_ratio_numerator = schro_aspect_ratios[index].numerator;
  format->aspect_ratio_denominator = schro_aspect_ratios[index].denominator;

}

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
  { 8, 8, 4, 4 },
  { 12, 12, 8, 8 },
  { 18, 16, 10, 12 },
  { 24, 24, 16, 16 }
};

void
schro_params_set_block_params (SchroParams *params, int index)
{
  if (index < 1 || index >= ARRAY_SIZE(schro_block_params) + 1) {
    SCHRO_ERROR("illegal block params index");
    return;
  }

  params->xblen_luma = schro_block_params[index-1].xblen_luma;
  params->yblen_luma = schro_block_params[index-1].yblen_luma;
  params->xbsep_luma = schro_block_params[index-1].xbsep_luma;
  params->ybsep_luma = schro_block_params[index-1].ybsep_luma;
}

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
    for(i=3;i<8;i++){
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
    for(i=3;i<8;i++){
      params->horiz_codeblocks[i] = 12;
      params->vert_codeblocks[i] = 8;
    }
  }
  params->codeblock_mode_index = 1;

}

void
schro_params_init_subbands (SchroParams *params, SchroSubband *subbands)
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
  stride = sizeof(int16_t)*(params->iwt_luma_width << params->transform_depth);
  chroma_w = params->iwt_chroma_width >> params->transform_depth;
  chroma_h = params->iwt_chroma_height >> params->transform_depth;
  chroma_stride = sizeof(int16_t)*(params->iwt_chroma_width << params->transform_depth);

  subbands[0].x = 0;
  subbands[0].y = 0;
  subbands[0].w = w;
  subbands[0].h = h;
  subbands[0].offset = 0;
  subbands[0].stride = stride;
  subbands[0].chroma_w = chroma_w;
  subbands[0].chroma_h = chroma_h;
  subbands[0].chroma_offset = 0;
  subbands[0].chroma_stride = chroma_stride;
  subbands[0].has_parent = 0;
  subbands[0].scale_factor_shift = 0;
  subbands[0].horizontally_oriented = 0;
  subbands[0].vertically_oriented = 0;

  for(i=0; i<params->transform_depth; i++) {
    /* hl */
    subbands[1+3*i].x = 1;
    subbands[1+3*i].y = 0;
    subbands[1+3*i].w = w;
    subbands[1+3*i].h = h;
    subbands[1+3*i].offset = w;
    subbands[1+3*i].stride = stride;
    subbands[1+3*i].chroma_w = chroma_w;
    subbands[1+3*i].chroma_h = chroma_h;
    subbands[1+3*i].chroma_offset = chroma_w;
    subbands[1+3*i].chroma_stride = chroma_stride;
    subbands[1+3*i].has_parent = (i>0);
    subbands[1+3*i].scale_factor_shift = i;
    subbands[1+3*i].horizontally_oriented = 0;
    subbands[1+3*i].vertically_oriented = 1;

    /* lh */
    subbands[2+3*i].x = 0;
    subbands[2+3*i].y = 1;
    subbands[2+3*i].w = w;
    subbands[2+3*i].h = h;
    subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    subbands[2+3*i].stride = stride;
    subbands[2+3*i].chroma_w = chroma_w;
    subbands[2+3*i].chroma_h = chroma_h;
    subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    subbands[2+3*i].chroma_stride = chroma_stride;
    subbands[2+3*i].has_parent = (i>0);
    subbands[2+3*i].scale_factor_shift = i;
    subbands[2+3*i].horizontally_oriented = 1;
    subbands[2+3*i].vertically_oriented = 0;

    /* hh */
    subbands[3+3*i].x = 1;
    subbands[3+3*i].y = 1;
    subbands[3+3*i].w = w;
    subbands[3+3*i].h = h;
    subbands[3+3*i].offset = w + (stride/2/sizeof(int16_t));
    subbands[3+3*i].stride = stride;
    subbands[3+3*i].chroma_w = chroma_w;
    subbands[3+3*i].chroma_h = chroma_h;
    subbands[3+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    subbands[3+3*i].chroma_stride = chroma_stride;
    subbands[3+3*i].has_parent = (i>0);
    subbands[3+3*i].scale_factor_shift = i;
    subbands[3+3*i].horizontally_oriented = 0;
    subbands[3+3*i].vertically_oriented = 0;

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}
