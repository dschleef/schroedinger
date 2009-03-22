
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <string.h>
#include <schroedinger/schrooil.h>

int _schro_motion_ref = FALSE;

#ifdef ENABLE_MOTION_REF
static int
get_pixel (SchroMotion *motion, int k, SchroUpsampledFrame *upframe,
    int x, int y, int dx, int dy);

int
schro_motion_pixel_predict_block (SchroMotion *motion, int x, int y, int k,
    int i, int j);
#endif


SchroMotion *
schro_motion_new (SchroParams *params, SchroUpsampledFrame *ref1,
    SchroUpsampledFrame *ref2)
{
  SchroMotion *motion;

  motion = schro_malloc0 (sizeof(SchroMotion));

  motion->params = params;
  motion->src1 = ref1;
  motion->src2 = ref2;

  motion->motion_vectors = schro_malloc0 (
      sizeof(SchroMotionVector)*params->x_num_blocks*params->y_num_blocks);

  motion->tmpdata = schro_malloc (64*64*3);

  return motion;
}

void
schro_motion_free (SchroMotion *motion)
{
  schro_free (motion->tmpdata);
  schro_free (motion->motion_vectors);
  schro_free (motion);
}

#ifdef ENABLE_MOTION_REF
static void
schro_motion_get_global_vector (SchroMotion *motion, int ref, int x, int y,
    int *dx, int *dy)
{
  SchroParams *params = motion->params;
  SchroGlobalMotion *gm = params->global_motion + ref;
  int alpha, beta;
  int scale;

  alpha = gm->a_exp;
  beta = gm->c_exp;

  scale = (1<<beta) - (gm->c0 * x + gm->c1 * y);

  *dx = scale * (gm->a00 * x + gm->a01 * y + (1<<alpha) * gm->b0);
  *dy = scale * (gm->a10 * x + gm->a11 * y + (1<<alpha) * gm->b1);

  *dx >>= (alpha+beta);
  *dy >>= (alpha+beta);
}


static int
schro_motion_pixel_predict (SchroMotion *motion, int x, int y, int k)
{
  int i,j;
  int value;

  i = (x + motion->xoffset) / motion->xbsep - 1;
  j = (y + motion->yoffset) / motion->ybsep - 1;

  value = schro_motion_pixel_predict_block (motion, x, y, k, i, j);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i + 1, j);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i, j + 1);
  value += schro_motion_pixel_predict_block (motion, x, y, k, i + 1, j + 1);

  return ROUND_SHIFT(value, 6);
}

static int
get_dc_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVectorDC *mvdc;

  mvdc = (SchroMotionVectorDC *)
    &motion->motion_vectors[j*params->x_num_blocks + i];

  return mvdc->dc[k] + 128;
}

static int
get_ref1_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx, dy;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 0, x, y, &dx, &dy);
  } else {
    dx = mv->dx[0];
    dy = mv->dy[0];
  }

  value = (motion->ref1_weight + motion->ref2_weight) *
      get_pixel (motion, k, motion->src1, x, y, dx, dy);

  return ROUND_SHIFT(value, motion->ref_weight_precision);
}

static int
get_ref2_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx, dy;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 1, x, y, &dx, &dy);
  } else {
    dx = mv->dx[1];
    dy = mv->dy[1];
  }

  value = (motion->ref1_weight + motion->ref2_weight) *
      get_pixel (motion, k, motion->src2, x, y, dx, dy);

  return ROUND_SHIFT(value, motion->ref_weight_precision);
}

static int
get_biref_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int dx0, dx1, dy0, dy1;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  if (mv->using_global) {
    schro_motion_get_global_vector (motion, 0, x, y, &dx0, &dy0);
    schro_motion_get_global_vector (motion, 1, x, y, &dx1, &dy1);
  } else {
    dx0 = mv->dx[0];
    dy0 = mv->dy[0];
    dx1 = mv->dx[1];
    dy1 = mv->dy[1];
  }

  value = motion->ref1_weight *
    get_pixel (motion, k, motion->src1, x, y, dx0, dy0);
  value += motion->ref2_weight *
    get_pixel (motion, k, motion->src2, x, y, dx1, dy1);

  return ROUND_SHIFT(value, motion->ref_weight_precision);
}

static int
get_pixel (SchroMotion *motion, int k, SchroUpsampledFrame *upframe,
    int x, int y, int dx, int dy)
{
  int px, py;

  if (k > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
  }

  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;

  return schro_upsampled_frame_get_pixel_precN (upframe, k, px, py,
      motion->mv_precision);
}
#endif

static int
get_ramp (int x, int offset)
{
  if (offset == 1) {
    if (x == 0) return 3;
    return 5;
  }
  return 1 + (6 * x + offset - 1)/(2*offset - 1);
}

#ifdef ENABLE_MOTION_REF
int
schro_motion_pixel_predict_block (SchroMotion *motion, int x, int y, int k,
    int i, int j)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int xmin, xmax, ymin, ymax;
  int wx, wy;
  int value;
  int width, height;

  if (i < 0 || j < 0) return 0;
  if (i >= params->x_num_blocks || j >= params->y_num_blocks) return 0;

  width = motion->xbsep * params->x_num_blocks;
  height = motion->ybsep * params->y_num_blocks;

  xmin = i * motion->xbsep - motion->xoffset;
  ymin = j * motion->ybsep - motion->yoffset;
  xmax = (i+1) * motion->xbsep + motion->xoffset;
  ymax = (j+1) * motion->ybsep + motion->yoffset;

  if (x < xmin || y < ymin || x >= xmax || y >= ymax) return 0;

  if (motion->xoffset == 0) {
    wx = 8;
  } else if (x < motion->xoffset || x >= width - motion->xoffset) {
    wx = 8;
  } else if (x - xmin < 2*motion->xoffset) {
    wx = get_ramp (x - xmin, motion->xoffset);
  } else if (xmax - 1 - x < 2*motion->xoffset) {
    wx = get_ramp (xmax - 1 - x, motion->xoffset);
  } else {
    wx = 8;
  }

  if (motion->yoffset == 0) {
    wy = 8;
  } else if (y < motion->yoffset || y >= height - motion->yoffset) {
    wy = 8;
  } else if (y - ymin < 2*motion->yoffset) {
    wy = get_ramp (y - ymin, motion->yoffset);
  } else if (ymax - 1 - y < 2*motion->yoffset) {
    wy = get_ramp (ymax - 1 - y, motion->yoffset);
  } else {
    wy = 8;
  }

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];

  switch (mv->pred_mode) {
    case 0:
      value = get_dc_pixel (motion, i, j, k, x, y);
      break;
    case 1:
      value = get_ref1_pixel (motion, i, j, k, x, y);
      break;
    case 2:
      value = get_ref2_pixel (motion, i, j, k, x, y);
      break;
    case 3:
      value = get_biref_pixel (motion, i, j, k, x, y);
      break;
    default:
      value = 0;
      break;
  }

  return value * wx * wy;
}
#endif

/* motion render (faster) */

void
get_block (SchroMotion *motion, int k, int ref, int i, int j, int dx, int dy)
{
  int px, py;
  int x, y;
  SchroUpsampledFrame *upframe;
  int exp;

  if (k > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
  }
  if (ref) {
    upframe = motion->src2;
  } else {
    upframe = motion->src1;
  }

  x = motion->xbsep * i - motion->xoffset;
  y = motion->ybsep * j - motion->yoffset;
  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;
  exp = 32 << motion->mv_precision;

  px = CLAMP (px, -exp, motion->max_fast_x + exp-1);
  py = CLAMP (py, -exp, motion->max_fast_y + exp-1);

  schro_upsampled_frame_get_block_fast_precN (upframe, k, px, py,
      motion->mv_precision, &motion->block_ref[ref],
      &motion->alloc_block_ref[ref]);
}

static void
get_dc_block (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVectorDC *mvdc;
  int value;
  int ii, jj;

  mvdc = (SchroMotionVectorDC *)
    &motion->motion_vectors[j*params->x_num_blocks + i];

  memcpy (&motion->block, &motion->alloc_block, sizeof(SchroFrameData));
  value = mvdc->dc[k];
  for(jj=0;jj<motion->yblen;jj++) {
    uint8_t *data = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
    /* FIXME splat */
    for(ii=0;ii<motion->xblen;ii++) {
      data[ii] = value + 128;
    }
  }
}

static void
get_ref1_block (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int ii, jj;
  int weight;
  int shift;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->dx[0], mv->dy[0]);

  weight = motion->ref1_weight + motion->ref2_weight;
  shift = motion->ref_weight_precision;

  if (weight == (1<<shift)) {
    memcpy (&motion->block, &motion->block_ref[0],
        sizeof(SchroFrameData));
  } else {
    memcpy (&motion->block, &motion->alloc_block,
        sizeof(SchroFrameData));
    for(jj=0;jj<motion->yblen;jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[0], jj);
      for(ii=0;ii<motion->xblen;ii++) {
        d[ii] = ROUND_SHIFT(s[ii] * weight, shift);
      }
    }
  }
}

static void
get_ref2_block (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int ii, jj;
  int weight;
  int shift;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 1, i, j, mv->dx[1], mv->dy[1]);

  weight = motion->ref1_weight + motion->ref2_weight;
  shift = motion->ref_weight_precision;

  if (weight == (1<<shift)) {
    memcpy (&motion->block, &motion->block_ref[1],
        sizeof(SchroFrameData));
  } else {
    memcpy (&motion->block, &motion->alloc_block,
        sizeof(SchroFrameData));
    for(jj=0;jj<motion->yblen;jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[1], jj);
      for(ii=0;ii<motion->xblen;ii++) {
        d[ii] = ROUND_SHIFT(s[ii] * weight, shift);
      }
    }
  }
}

static void
get_biref_block (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int ii, jj;
  int weight0, weight1;
  int shift;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->dx[0], mv->dy[0]);
  get_block (motion, k, 1, i, j, mv->dx[1], mv->dy[1]);

  weight0 = motion->ref1_weight;
  weight1 = motion->ref2_weight;
  shift = motion->ref_weight_precision;

  memcpy (&motion->block, &motion->alloc_block, sizeof(SchroFrameData));
  if (weight0 == 1 && weight1 == 1 && shift == 1) {
    switch (motion->xblen) {
      case 8:
        oil_avg2_8xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 12:
        oil_avg2_12xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 16:
        oil_avg2_16xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 32:
        oil_avg2_32xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      default:
        for(jj=0;jj<motion->yblen;jj++) {
          uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
          uint8_t *s0 = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[0], jj);
          uint8_t *s1 = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[1], jj);
          for(ii=0;ii<motion->xblen;ii++) {
            d[ii] = (s0[ii] + s1[ii] + 1)>>1;
          }
        }
        break;
    }
  } else {
    int16_t as[4];
    as[0] = weight0;
    as[1] = weight1;
    as[2] = (1<<shift)>>1;
    as[3] = shift;

    switch (motion->xblen) {
      case 8:
        oil_combine2_8xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            as, motion->yblen);
        break;
      case 12:
        oil_combine2_16xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            as, motion->yblen);
        break;
      case 16:
        oil_combine2_16xn_u8(motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            as, motion->yblen);
        break;
      default:
        for(jj=0;jj<motion->yblen;jj++) {
          uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
          uint8_t *s0 = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[0], jj);
          uint8_t *s1 = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[1], jj);
          for(ii=0;ii<motion->xblen;ii++) {
            d[ii] = ROUND_SHIFT(s0[ii] * weight0 + s1[ii] * weight1, shift);
          }
        }
        break;
    }
  }
}

void
schro_motion_block_predict_block (SchroMotion *motion, int x, int y, int k,
    int i, int j)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];

  switch (mv->pred_mode) {
    case 0:
      get_dc_block (motion, i, j, k, x, y);
      break;
    case 1:
      get_ref1_block (motion, i, j, k, x, y);
      break;
    case 2:
      get_ref2_block (motion, i, j, k, x, y);
      break;
    case 3:
      get_biref_block (motion, i, j, k, x, y);
      break;
    default:
      SCHRO_ASSERT(0);
      break;
  }
}

void
schro_motion_block_accumulate (SchroMotion *motion, SchroFrameData *comp,
    int x, int y)
{
  int j;

  switch (motion->xblen) {
    case 6:
      oil_multiply_and_acc_6xn_s16_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y), comp->stride,
          motion->obmc_weight.data, motion->obmc_weight.stride,
          motion->block.data, motion->block.stride,
          motion->yblen);
      break;
    case 8:
      oil_multiply_and_acc_8xn_s16_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y), comp->stride,
          motion->obmc_weight.data, motion->obmc_weight.stride,
          motion->block.data, motion->block.stride,
          motion->yblen);
      break;
    case 12:
      oil_multiply_and_acc_12xn_s16_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y), comp->stride,
          motion->obmc_weight.data, motion->obmc_weight.stride,
          motion->block.data, motion->block.stride,
          motion->yblen);
      break;
    case 16:
      oil_multiply_and_acc_16xn_s16_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y), comp->stride,
          motion->obmc_weight.data, motion->obmc_weight.stride,
          motion->block.data, motion->block.stride,
          motion->yblen);
      break;
    case 24:
      oil_multiply_and_acc_24xn_s16_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y), comp->stride,
          motion->obmc_weight.data, motion->obmc_weight.stride,
          motion->block.data, motion->block.stride,
          motion->yblen);
      break;
    default:
      for(j=0;j<motion->yblen;j++) {
        oil_multiply_and_add_s16_u8 (
            SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j),
            SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j),
            SCHRO_FRAME_DATA_GET_LINE (&motion->obmc_weight, j),
            SCHRO_FRAME_DATA_GET_LINE (&motion->block, j),
            motion->xblen);
      }
      break;
  }
}

void
schro_motion_block_accumulate_slow (SchroMotion *motion, SchroFrameData *comp,
    int x, int y)
{
  int i,j;
  int w_x, w_y;

  for(j=0;j<motion->yblen;j++) {
    int16_t *d = SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j);
    uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block, j);

    if (y + j < 0 || y + j >= comp->height) continue;

    w_y = motion->weight_y[j];
    if (y + j < motion->yoffset) {
      w_y += motion->weight_y[2*motion->yoffset - j - 1];
    }
    if (y + j >=
        motion->params->y_num_blocks * motion->ybsep - motion->yoffset) {
      w_y += motion->weight_y[2*(motion->yblen - motion->yoffset) - j - 1];
    }


    for(i=0;i<motion->xblen;i++) {
      if (x + i < 0 || x + i >= comp->width) continue;

      w_x = motion->weight_x[i];
      if (x + i < motion->xoffset) {
        w_x += motion->weight_x[2*motion->xoffset - i - 1];
      }
      if (x + i >=
          motion->params->x_num_blocks * motion->xbsep - motion->xoffset) {
        w_x += motion->weight_x[2*(motion->xblen - motion->xoffset) - i - 1];
      }

      d[i] += s[i] * w_x * w_y;
    }
  }
}

void
schro_motion_init_obmc_weight (SchroMotion *motion)
{
  int i;
  int j;
  int wx, wy;

  for(i=0;i<motion->xblen;i++){
    if (motion->xoffset == 0) {
      wx = 8;
    } else if (i < 2*motion->xoffset) {
      wx = get_ramp (i, motion->xoffset);
    } else if (motion->xblen - 1 - i < 2*motion->xoffset) {
      wx = get_ramp (motion->xblen - 1 - i, motion->xoffset);
    } else {
      wx = 8;
    }
    motion->weight_x[i] = wx;
  }

  for(j=0;j<motion->yblen;j++){
    if (motion->yoffset == 0) {
      wy = 8;
    } else if (j < 2*motion->yoffset) {
      wy = get_ramp (j, motion->yoffset);
    } else if (motion->yblen - 1 - j < 2*motion->yoffset) {
      wy = get_ramp (motion->yblen - 1 - j, motion->yoffset);
    } else {
      wy = 8;
    }
    motion->weight_y[j] = wy;
  }

  for(j=0;j<motion->yblen;j++){
    int16_t *w = SCHRO_FRAME_DATA_GET_LINE (&motion->obmc_weight, j);

    for(i=0;i<motion->xblen;i++){
      w[i] = motion->weight_x[i] * motion->weight_y[j];
    }
  }

}

void
schro_motion_render (SchroMotion *motion, SchroFrame *dest)
{
  int i, j;
  int x, y;
  int k;
  SchroParams *params = motion->params;
  int max_x_blocks;
  int max_y_blocks;
  int16_t zero = 0;

#ifdef ENABLE_MOTION_REF
  if (_schro_motion_ref) {
    schro_motion_render_ref (motion, dest);
    return;
  }
#endif

  if (params->have_global_motion) {
#ifdef ENABLE_MOTION_REF
    SCHRO_WARNING ("global motion enabled, using reference motion renderer");
    schro_motion_render_ref (motion, dest);
    return;
#else
    SCHRO_ERROR ("global motion enabled, probably will crash");
#endif
  }

  {
    int min_extension;
    int i;

    min_extension = motion->src1->frames[0]->extension;
    for(i=0;i<4;i++){
      if (motion->src1->frames[i]) {
        min_extension = MIN(min_extension, motion->src1->frames[i]->extension);
      }
      if (motion->src2 && motion->src2->frames[i]) {
        min_extension = MIN(min_extension, motion->src2->frames[i]->extension);
      }
    }

    if (MAX(params->xblen_luma, params->yblen_luma) > min_extension) {
#ifdef ENABLE_MOTION_REF
      SCHRO_WARNING ("block size (%dx%d) larger than minimum frame extension %d, using reference motion renderer",
          params->xblen_luma, params->yblen_luma, min_extension);
      schro_motion_render_ref (motion, dest);
      return;
#else
      SCHRO_ERROR ("block size (%dx%d) larger than minimum frame extension %d, probably will crash",
          params->xblen_luma, params->yblen_luma, min_extension);
#endif
    }
  }

  if (params->num_refs == 1) {
    SCHRO_ASSERT(params->picture_weight_2 == 1);
  }

  motion->ref_weight_precision = params->picture_weight_bits;
  motion->ref1_weight = params->picture_weight_1;
  motion->ref2_weight = params->picture_weight_2;

  motion->mv_precision = params->mv_precision;

  for (k=0;k<3;k++){
    SchroFrameData *comp = dest->components + k;

    if (k == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
      motion->width = comp->width;
      motion->height = comp->height;
    } else {
      motion->xbsep = params->xbsep_luma >>
        SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
      motion->ybsep = params->ybsep_luma >>
        SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
      motion->xblen = params->xblen_luma >>
        SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
      motion->yblen = params->yblen_luma >>
        SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
      motion->width = comp->width;
      motion->height = comp->height;
    }
    motion->xoffset = (motion->xblen - motion->xbsep)/2;
    motion->yoffset = (motion->yblen - motion->ybsep)/2;
    motion->max_fast_x = (motion->width - motion->xblen) << motion->mv_precision;
    motion->max_fast_y = (motion->height - motion->yblen) << motion->mv_precision;

    motion->alloc_block.data = schro_malloc (motion->xblen * motion->yblen * sizeof(uint8_t));
    motion->alloc_block.stride = motion->xblen * sizeof(uint8_t);
    motion->alloc_block.width = motion->xblen;
    motion->alloc_block.height = motion->yblen;
    motion->obmc_weight.data = schro_malloc (motion->xblen * motion->yblen * sizeof(int16_t));
    motion->obmc_weight.stride = motion->xblen * sizeof(int16_t);
    motion->obmc_weight.width = motion->xblen;
    motion->obmc_weight.height = motion->yblen;
    motion->alloc_block_ref[0].data = schro_malloc (motion->xblen * motion->yblen * sizeof(uint8_t));
    motion->alloc_block_ref[0].stride = motion->xblen * sizeof(uint8_t);
    motion->alloc_block_ref[0].width = motion->xblen;
    motion->alloc_block_ref[0].height = motion->yblen;
    motion->alloc_block_ref[1].data = schro_malloc (motion->xblen * motion->yblen * sizeof(uint8_t));
    motion->alloc_block_ref[1].stride = motion->xblen * sizeof(uint8_t);
    motion->alloc_block_ref[1].width = motion->xblen;
    motion->alloc_block_ref[1].height = motion->yblen;

    schro_motion_init_obmc_weight (motion);

    for(j=0;j<comp->height;j++){
      oil_splat_s16_ns (SCHRO_FRAME_DATA_GET_LINE(comp, j), &zero,
          comp->width);
    }

    max_x_blocks = MIN(params->x_num_blocks - 1,
        (motion->width - motion->xoffset)/motion->xbsep);
    max_y_blocks = MIN(params->y_num_blocks - 1,
        (motion->height - motion->yoffset)/motion->ybsep);

    j = 0;
    for(i=0;i<params->x_num_blocks;i++){
      x = motion->xbsep * i - motion->xoffset;
      y = motion->ybsep * j - motion->yoffset;

      schro_motion_block_predict_block (motion, x, y, k, i, j);
      schro_motion_block_accumulate_slow (motion, comp, x, y);
    }
    for(j=1;j<max_y_blocks;j++){
      y = motion->ybsep * j - motion->yoffset;

      i = 0;
      {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }

      for(i=1;i<max_x_blocks;i++){
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate (motion, comp, x, y);
      }

      for(;i<params->x_num_blocks;i++){
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
    }
    for(j=max_y_blocks;j<params->y_num_blocks;j++){
      y = motion->ybsep * j - motion->yoffset;
      for(i=0;i<params->x_num_blocks;i++){
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
    }


    for(j=0;j<comp->height;j++){
      int16_t as[2] = { 1, 1 };

      as[1] = 6;
      as[0] = ((1<<as[1])>>1) - (128<<6);

      oil_add_const_rshift_s16 (SCHRO_FRAME_DATA_GET_LINE(comp, j),
          SCHRO_FRAME_DATA_GET_LINE(comp, j), as, motion->width);
    }

    schro_free (motion->alloc_block.data);
    schro_free (motion->obmc_weight.data);
    schro_free (motion->alloc_block_ref[0].data);
    schro_free (motion->alloc_block_ref[1].data);
  }

}



/* original */


void
schro_motion_dc_prediction (SchroMotion *motion, int x, int y, int *pred)
{
  SchroMotionVectorDC *mvdc;
  int i;

  for(i=0;i<3;i++){
    int sum = 0;
    int n = 0;

    if (x>0) {
      mvdc = SCHRO_MOTION_GET_DC_BLOCK(motion,x-1,y);
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    if (y>0) {
      mvdc = SCHRO_MOTION_GET_DC_BLOCK(motion,x,y-1);
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    if (x>0 && y>0) {
      mvdc = SCHRO_MOTION_GET_DC_BLOCK(motion,x-1,y-1);
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    switch(n) {
      case 0:
        pred[i] = 0;
        break;
      case 1:
        pred[i] = (short)sum;
        break;
      case 2:
        pred[i] = (sum+1)>>1;
        break;
      case 3:
        pred[i] = schro_divide(sum + 1,3);
        break;
      default:
        SCHRO_ASSERT(0);
    }
  }
}

int
schro_motion_get_global_prediction (SchroMotion *motion,
    int x, int y)
{
  SchroMotionVector *mv;
  int sum;

  if (x == 0 && y == 0) {
    return 0;
  }
  if (y == 0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,0);
    return mv->using_global;
  }
  if (x == 0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,0,y-1);
    return mv->using_global;
  }

  sum = 0;
  mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y);
  sum += mv->using_global;
  mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-1);
  sum += mv->using_global;
  mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y-1);
  sum += mv->using_global;

  return (sum >= 2);
}

static int
median3(int a, int b, int c)
{
  if (a < b) {
    if (b < c) return b;
    if (c < a) return a;
    return c;
  } else {
    if (a < c) return a;
    if (c < b) return b;
    return c;
  }
}

void
schro_motion_vector_prediction (SchroMotion *motion,
    int x, int y, int *pred_x, int *pred_y, int mode)
{
  SchroMotionVector *mv;
  int vx[3];
  int vy[3];
  int n = 0;

  SCHRO_ASSERT(mode == 1 || mode == 2);
  if (x>0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->dx[mode-1];
      vy[n] = mv->dy[mode-1];
      n++;
    }
  }
  if (y>0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->dx[mode-1];
      vy[n] = mv->dy[mode-1];
      n++;
    }
  }
  if (x>0 && y>0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y-1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->dx[mode-1];
      vy[n] = mv->dy[mode-1];
      n++;
    }
  }
  switch(n) {
    case 0:
      *pred_x = 0;
      *pred_y = 0;
      break;
    case 1:
      *pred_x = vx[0];
      *pred_y = vy[0];
      break;
    case 2:
      *pred_x = (vx[0] + vx[1] + 1)>>1;
      *pred_y = (vy[0] + vy[1] + 1)>>1;
      break;
    case 3:
      *pred_x = median3(vx[0], vx[1], vx[2]);
      *pred_y = median3(vy[0], vy[1], vy[2]);
      break;
    default:
      SCHRO_ASSERT(0);
  }
}

int
schro_motion_split_prediction (SchroMotion *motion, int x, int y)
{
  SchroMotionVector *mv;

  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      mv = SCHRO_MOTION_GET_BLOCK(motion,x-4,0);
      return mv->split;
    }
  } else {
    if (x == 0) {
      mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-4);
      return mv->split;
    } else {
      int sum;

      mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-4);
      sum = mv->split;
      mv = SCHRO_MOTION_GET_BLOCK(motion,x-4,y);
      sum += mv->split;
      mv = SCHRO_MOTION_GET_BLOCK(motion,x-4,y-4);
      sum += mv->split;

      return (sum + 1)/3;
    }
  }
}

int
schro_motion_get_mode_prediction (SchroMotion *motion, int x, int y)
{
  SchroMotionVector *mv;

  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,0);
      return mv->pred_mode;
    }
  } else {
    if (x == 0) {
      mv = SCHRO_MOTION_GET_BLOCK(motion,0,y-1);
      return mv->pred_mode;
    } else {
      int a, b, c;

      mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y);
      a = mv->pred_mode;
      mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-1);
      b = mv->pred_mode;
      mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y-1);
      c = mv->pred_mode;

      return (a&b)|(b&c)|(c&a);
    }
  }
}

int
schro_motion_vector_is_equal (SchroMotionVector *a, SchroMotionVector *b)
{
  if (a == b) return 1;
  return (memcmp (a,b,sizeof(SchroMotionVector))==0);
}

int
schro_motion_verify (SchroMotion *motion)
{
  int x,y;
  SchroMotionVector *mv, *sbmv, *bmv;
  SchroParams *params = motion->params;

  if (motion->src1 == NULL) {
    SCHRO_ERROR("motion->src1 is NULL");
    return 0;
  }

  for(y=0;y<params->y_num_blocks;y++){
    for(x=0;x<params->x_num_blocks;x++){
      mv = &motion->motion_vectors[y*params->x_num_blocks + x];
      sbmv = &motion->motion_vectors[(y&~3)*params->x_num_blocks + (x&~3)];

      if (mv->split != sbmv->split) {
        SCHRO_ERROR("mv(%d,%d) has the wrong split", x, y);
        return 0;
      }

      switch (sbmv->split) {
        case 0:
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
            SCHRO_ERROR("mv(%d,%d) not equal to superblock mv", x, y);
            return 0;
          }
          break;
        case 1:
          bmv = &motion->motion_vectors[(y&~1)*params->x_num_blocks + (x&~1)];
          if (!schro_motion_vector_is_equal (mv, bmv)) {
            SCHRO_ERROR("mv(%d,%d) not equal to 2-block mv", x, y);
            return 0;
          }
          break;
        case 2:
          break;
        default:
          SCHRO_ERROR("mv(%d,%d) had bad split %d", sbmv->split);
          break;
      }

      switch (mv->pred_mode) {
        case 0:
          {
            SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;
            int i;

            for(i=0;i<3;i++){
              /* FIXME 8bit */
              if (mvdc->dc[i] < -128 || mvdc->dc[i] > 127) {
                SCHRO_ERROR("mv(%d,%d) has bad DC value [%d] %d", x, y,
                    i, mvdc->dc[i]);
                return 0;
              }
            }
          }
          break;
        case 1:
          break;
        case 2:
        case 3:
          if (motion->params->num_refs < 2) {
            SCHRO_ERROR("mv(%d,%d) uses non-existent src2", x, y);
            return 0;
          }
          break;
      }

      if (params->have_global_motion == FALSE) {
        if (mv->using_global) {
          SCHRO_ERROR("mv(%d,%d) uses global motion (disabled)", x, y);
          return 0;
        }
      }
    }
  }

  return 1;
}

#ifdef ENABLE_MOTION_REF
void
schro_motion_render_ref (SchroMotion *motion, SchroFrame *dest)
{
  SchroParams *params = motion->params;
  int k;
  int x,y;
  int16_t *line;

  if (params->num_refs == 1) {
    SCHRO_ASSERT(params->picture_weight_2 == 1);
  }

  motion->ref_weight_precision = params->picture_weight_bits;
  motion->ref1_weight = params->picture_weight_1;
  motion->ref2_weight = params->picture_weight_2;

  motion->mv_precision = params->mv_precision;

  for(k=0;k<3;k++){
    SchroFrameData *comp = dest->components + k;

    if (k == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
    } else {
      motion->xbsep = params->xbsep_luma >>
        SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
      motion->ybsep = params->ybsep_luma >>
        SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
      motion->xblen = params->xblen_luma >>
        SCHRO_CHROMA_FORMAT_H_SHIFT(motion->params->video_format->chroma_format);
      motion->yblen = params->yblen_luma >>
        SCHRO_CHROMA_FORMAT_V_SHIFT(motion->params->video_format->chroma_format);
    }
    motion->xoffset = (motion->xblen - motion->xbsep)/2;
    motion->yoffset = (motion->yblen - motion->ybsep)/2;

    for(y=0;y<comp->height;y++){
      line = SCHRO_FRAME_DATA_GET_LINE(comp, y);
      for(x=0;x<comp->width;x++){
        line[x] = CLAMP(schro_motion_pixel_predict (motion, x, y, k), 0, 255);

        /* Note: the 128 offset converts the 0-255 range of the reference
         * pictures into the bipolar range used for Dirac signal processing */
        line[x] -= 128;
      }
    }
  }
}
#endif

