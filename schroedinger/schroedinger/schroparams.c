
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>


#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))

static int
round_up_pow2 (int x, int p)
{
  int y = (1<<p) - 1;
  x += y;
  x &= ~y;
  return x;
}

#define DIVIDE_ROUND_UP(a,b) (((a) + (b) - 1)/(b))

void
schro_params_calculate_mc_sizes (SchroParams *params)
{
  params->x_num_mb =
    DIVIDE_ROUND_UP(params->width, 4*params->xbsep_luma);
  params->y_num_mb =
    DIVIDE_ROUND_UP(params->height, 4*params->ybsep_luma);

  params->x_num_blocks = 4 * params->x_num_mb;
  params->y_num_blocks = 4 * params->y_num_mb;
  params->mc_luma_width = 4 * params->x_num_mb * params->xbsep_luma;
  params->mc_luma_height = 4 * params->y_num_mb * params->ybsep_luma;
  params->mc_chroma_width = params->mc_luma_width / params->chroma_h_scale;
  params->mc_chroma_height = params->mc_luma_width / params->chroma_v_scale;
}

void
schro_params_calculate_iwt_sizes (SchroParams *params)
{
  SCHRO_DEBUG ("chroma size %d x %d", params->chroma_width,
      params->chroma_height);
  if (params->is_intra) {
    params->iwt_chroma_width =
      round_up_pow2(params->chroma_width,params->transform_depth);
    params->iwt_chroma_height =
      round_up_pow2(params->chroma_height, params->transform_depth);
  } else {
    params->iwt_chroma_width =
      round_up_pow2(params->mc_chroma_width, params->transform_depth);
    params->iwt_chroma_height =
      round_up_pow2(params->mc_chroma_height, params->transform_depth);
  }
  SCHRO_DEBUG ("iwt chroma size %d x %d", params->iwt_chroma_width,
      params->iwt_chroma_height);
  params->iwt_luma_width =
    params->iwt_chroma_width * params->chroma_h_scale;
  params->iwt_luma_height =
    params->iwt_chroma_height * params->chroma_v_scale;
  SCHRO_DEBUG ("iwt luma size %d x %d", params->iwt_luma_width,
      params->iwt_luma_height);
}

typedef struct _SchroVideoFormat SchroVideoFormat;
struct _SchroVideoFormat {
  char *name;
  int width;
  int height;
  int interlace;
  int top_field_first;
  int frame_rate_numerator;
  int frame_rate_denominator;
  int pixel_aspect_ratio_numerator;
  int pixel_aspect_ratio_denominator;
  int clean_tl_x;
  int clean_tl_y;
  int clean_height;
  int clean_width;
  int colour_matrix_index;
  int signal_range_index;
  int colour_primaries_index;
  int transfer_char_index;
  int block_params_index;
};

static SchroVideoFormat
schro_video_formats[] = {
  { "custom", 640, 480, FALSE, TRUE, 30, 1, 1, 1, 0, 0, 640, 480,
    1, 1, 1, 0, 2 },
  { "QSIF", 176, 120, FALSE, TRUE, 15, 1, 10, 11, 0, 0, 176, 120,
    1, 1, 1, 0, 1 },
  { "QCIF", 176, 144, FALSE, TRUE, 25, 2, 59, 54, 0, 0, 176, 144,
    1, 1, 2, 0, 1 },
  { "SIF", 352, 240, FALSE, TRUE, 15, 1, 10, 11, 0, 0, 352, 240,
    1, 1, 1, 0, 2 },
  { "CIF", 352, 288, FALSE, TRUE, 25, 2, 59, 54, 0, 0, 352, 288,
    1, 1, 2, 0, 2 },
  { "SD (NTSC)", 704, 480, TRUE, TRUE, 30000, 1001, 10, 11, 0, 0, 704, 480,
    1, 2, 1, 0, 2 },
  { "SD (PAL)", 704, 576, TRUE, TRUE, 25, 1, 59, 54, 0, 0, 704, 576,
    1, 2, 2, 0, 2 },
  { "SD (525 Digital)", 720, 480, TRUE, TRUE, 30000, 1001, 10, 11, 0, 0, 720, 480,
    1, 2, 1, 0, 2 },
  { "SD (625 Digital)", 720, 576, TRUE, TRUE, 30, 1, 59, 54, 0, 0, 720, 576,
    1, 2, 2, 0, 2 },
  { "HD 720", 1280, 720, FALSE, TRUE, 24, 1, 1, 1, 0, 0, 1280, 720,
    2, 2, 3, 0, 3 },
  { "HD 1080", 1920, 1080, FALSE, TRUE, 24, 1, 1, 1, 0, 0, 1920, 1080,
    2, 2, 3, 0, 4 },
  { "Advanced Video Format", 1920, 1080, FALSE, TRUE, 50, 1, 1, 1, 0, 0, 1920, 1080,
    3, 5, 3, 1, 4 }
};

void schro_params_set_video_format (SchroParams *params, int index)
{
  SchroVideoFormat *format;

  if (index < 0 || index >= ARRAY_SIZE(schro_video_formats)) {
    SCHRO_ERROR("illegal video format index");
    return;
  }

  format = schro_video_formats + index;

  params->width = format->width;
  params->height = format->height;
  params->interlace = format->interlace;
  params->top_field_first = format->top_field_first;
  params->frame_rate_numerator = format->frame_rate_numerator;
  params->frame_rate_denominator = format->frame_rate_denominator;
  params->pixel_aspect_ratio_numerator = format->pixel_aspect_ratio_numerator;
  params->pixel_aspect_ratio_denominator =
    format->pixel_aspect_ratio_denominator;
  params->clean_tl_x = format->clean_tl_x;
  params->clean_tl_x = format->clean_tl_y;
  params->clean_width = format->clean_width;
  params->clean_height = format->clean_height;
  params->colour_matrix_index = format->colour_matrix_index;
  params->signal_range_index = format->signal_range_index;
  params->colour_primaries_index = format->colour_primaries_index;
  params->transfer_char_index = format->transfer_char_index;

  schro_params_set_block_params (params, format->block_params_index);
}

static int
schro_params_get_video_format_metric (SchroParams *params, int i)
{
  SchroVideoFormat *format;
  int metric = 0;

  format = schro_video_formats + i;

  if (params->width != format->width) {
    metric++;
  }
  if (params->height != format->height) {
    metric++;
  }
  if (params->interlace != format->interlace) {
    metric++;
  }
  if (params->top_field_first != format->top_field_first) {
    metric++;
  }
  if (params->frame_rate_numerator != format->frame_rate_numerator) {
    metric++;
  }
  if (params->frame_rate_denominator != format->frame_rate_denominator) {
    metric++;
  }
  if (params->pixel_aspect_ratio_numerator != format->pixel_aspect_ratio_numerator) {
    metric++;
  }
  if (params->pixel_aspect_ratio_denominator != format->pixel_aspect_ratio_denominator) {
    metric++;
  }
  if (params->clean_tl_x != format->clean_tl_x) {
    metric++;
  }
  if (params->clean_tl_x != format->clean_tl_y) {
    metric++;
  }
  if (params->clean_width != format->clean_width) {
    metric++;
  }
  if (params->clean_height != format->clean_height) {
    metric++;
  }
  if (params->colour_matrix_index != format->colour_matrix_index) {
    metric++;
  }
  if (params->signal_range_index != format->signal_range_index) {
    metric++;
  }
  if (params->colour_primaries_index != format->colour_primaries_index) {
    metric++;
  }
  if (params->transfer_char_index != format->transfer_char_index) {
    metric++;
  }

  return metric;
}

int schro_params_get_video_format (SchroParams *params)
{
  int metric;
  int min_index;
  int min_metric;
  int i;

  min_index = 0;
  min_metric = schro_params_get_video_format_metric (params, 0);
  for(i=1;i<ARRAY_SIZE (schro_video_formats); i++) {
    metric = schro_params_get_video_format_metric (params, i);
    if (metric < min_metric) {
      min_index = i;
      min_metric = metric;
    }
  }
  return min_index;
}

typedef struct _SchroChromaFormat SchroChromaFormat;
struct _SchroChromaFormat {
  int chroma_h_scale;
  int chroma_v_scale;
  int have_chroma;
};

static SchroChromaFormat
schro_chroma_formats[] = {
  { 2, 2, TRUE },
  { 2, 1, TRUE },
  { 4, 4, TRUE },
  { 1, 1, TRUE },
  { 1, 1, FALSE }
};

void schro_params_set_chroma_format (SchroParams *params, int index)
{
  if (index < 0 || index >= ARRAY_SIZE(schro_chroma_formats)) {
    SCHRO_ERROR("illegal chroma format index");
    return;
  }

  params->chroma_h_scale = schro_chroma_formats[index].chroma_h_scale;
  params->chroma_v_scale = schro_chroma_formats[index].chroma_v_scale;
  params->have_chroma = schro_chroma_formats[index].have_chroma;

}

int schro_params_get_chroma_format (SchroParams *params)
{
  int i;

  for(i=0;i<ARRAY_SIZE(schro_chroma_formats);i++){
    if (params->chroma_h_scale == schro_chroma_formats[i].chroma_h_scale &&
        params->chroma_v_scale == schro_chroma_formats[i].chroma_v_scale &&
        params->have_chroma == schro_chroma_formats[i].have_chroma) {
      return i;
    }
  }

  SCHRO_ERROR("illegal chroma format");

  return -1;
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


void schro_params_set_frame_rate (SchroParams *params, int index)
{
  if (index < 0 || index >= ARRAY_SIZE(schro_frame_rates)) {
    SCHRO_ERROR("illegal frame rate index");
    return;
  }

  params->frame_rate_numerator = schro_frame_rates[index].numerator;
  params->frame_rate_denominator = schro_frame_rates[index].denominator;
}

int schro_params_get_frame_rate (SchroParams *params)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_frame_rates);i++){
    if (params->frame_rate_numerator == schro_frame_rates[i].numerator &&
        params->frame_rate_denominator == schro_frame_rates[i].denominator) {
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

static SchroPixelAspectRatio
schro_pixel_aspect_ratios[] = {
  { 0, 0 },
  { 1, 1 },
  { 10, 11 },
  { 59, 54 }
};

void schro_params_set_pixel_aspect_ratio (SchroParams *params, int index)
{
  if (index < 0 || index >= ARRAY_SIZE(schro_pixel_aspect_ratios)) {
    SCHRO_ERROR("illegal pixel aspect ratio index");
    return;
  }

  params->pixel_aspect_ratio_numerator =
    schro_pixel_aspect_ratios[index].numerator;
  params->pixel_aspect_ratio_denominator =
    schro_pixel_aspect_ratios[index].denominator;

}

int schro_params_get_pixel_aspect_ratio (SchroParams *params)
{
  int i;

  for(i=1;i<ARRAY_SIZE(schro_pixel_aspect_ratios);i++){
    if (params->pixel_aspect_ratio_numerator ==
        schro_pixel_aspect_ratios[i].numerator &&
        params->pixel_aspect_ratio_denominator ==
        schro_pixel_aspect_ratios[i].denominator) {
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



