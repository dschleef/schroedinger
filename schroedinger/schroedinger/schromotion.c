
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <string.h>
#include <schroedinger/schrooil.h>


int _schro_motion_ref = FALSE;


SchroMotion *
schro_motion_new (SchroParams *params, SchroUpsampledFrame *ref1,
    SchroUpsampledFrame *ref2)
{
  SchroMotion *motion;

  motion = malloc(sizeof(SchroMotion));
  memset (motion, 0, sizeof(SchroMotion));

  motion->params = params;
  motion->src1 = ref1;
  motion->src2 = ref2;

  motion->motion_vectors = malloc(sizeof(SchroMotionVector)*
      params->x_num_blocks*params->y_num_blocks);
  memset (motion->motion_vectors, 0, sizeof(SchroMotionVector)*
      params->x_num_blocks*params->y_num_blocks);

  return motion;
}

void
schro_motion_free (SchroMotion *motion)
{
  free (motion->motion_vectors);
  free (motion);
}

static int
ilog2 (unsigned int x)
{
  int i;
  for(i=0;x > 1;i++){
    x >>= 1;
  }
  return i;
}

static int
ramp_shift (int ramp)
{
  switch (ramp) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 4:
      return 3;
    case 8:
      return 4;
    default:
      SCHRO_ASSERT(0);
  }
}

static void
obmc_calc (int16_t *data, int x_len, int y_len, int x_ramp,
    int y_ramp)
{
  int i;
  int j;

  if (x_ramp > 0) {
    for(i=0;i<x_len;i++){
      int w;
      if (i < x_ramp) {
        w = 1 + 2*i;
      } else if (i >= x_len - x_ramp) {
        w = 1 + 2*(x_len - 1 - i);
      } else {
        w = x_ramp*2;
      }
      data[i] = w;
    }
  } else {
    for(i=0;i<x_len;i++){
      data[i] = 1;
    }
  }

  if (y_ramp > 0) {
    for(j=0;j<y_len;j++){
      int w;
      if (j < y_ramp) {
        w = 1 + 2*j;
      } else if (j >= y_len - y_ramp) {
        w = 1 + 2*(y_len - 1 - j);
      } else {
        w = y_ramp*2;
      }
      data[x_len * j + 0] = w;
    }
  } else {
    for(j=0;j<y_len;j++){
      data[x_len * j + 0] = 1;
    }
  }

  for(j=1;j<y_len;j++){
    for(i=1;i<x_len;i++){
      data[x_len * j + i] = data[x_len * j + 0] * data[i];
    }
  }
}

void
fixup_region (SchroObmc *obmc, int k)
{
  int i,j;
  int16_t *weights = obmc->regions[k].weights[0];
  int x_len = obmc->x_len;

  /* fix up top */
  if (k<3) {
    for(j=0;j<obmc->y_ramp;j++){
      for(i=0;i<obmc->x_len;i++){
        weights[x_len * j + i] +=
          weights[x_len * (obmc->y_len - obmc->y_ramp + j) + i];
      }
    }
    obmc->regions[k].start_y = obmc->y_ramp/2;
  }
  /* fix up bottom */
  if (k >= 6) {
    for(j=0;j<obmc->y_ramp;j++){
      for(i=0;i<obmc->x_len;i++){
        weights[x_len * (obmc->y_len - obmc->y_ramp + j) + i] +=
          weights[x_len * j + i];
      }
    }
    obmc->regions[k].end_y = obmc->y_len - obmc->y_ramp/2;
  }
  /* fix up left */
  if (k % 3 == 0) {
    for(j=0;j<obmc->y_len;j++){
      for(i=0;i<obmc->x_ramp;i++){
        weights[x_len * j + i] +=
          weights[x_len * j + (obmc->x_len - obmc->x_ramp + i)];
      }
    }
    obmc->regions[k].start_x = obmc->x_ramp/2;
  }
  /* fix up right */
  if (k % 3 == 2) {
    for(j=0;j<obmc->y_len;j++){
      for(i=0;i<obmc->x_ramp;i++){
        weights[x_len * j + (obmc->x_len - obmc->x_ramp + i)] +=
          weights[x_len * j + i];
      }
    }
    obmc->regions[k].end_x = obmc->x_len - obmc->x_ramp/2;
  }
}

static void
scale_region (int16_t *dest, int16_t *src, int n, int weight)
{
  int i;
  for(i=0;i<n;i++) {
    dest[i] = weight * src[i];
  }
}

void
schro_obmc_init (SchroObmc *obmc, int x_len, int y_len, int x_sep, int y_sep,
    int ref1_weight, int ref2_weight, int ref_shift)
{
  int i;
  int k;
  int x_ramp;
  int y_ramp;
  int16_t *region;
  int size;

  SCHRO_DEBUG("obmc init len %d %d sep %d %d", x_len, y_len, x_sep, y_sep);

  memset (obmc, 0, sizeof(*obmc));

  x_ramp = x_len - x_sep;
  y_ramp = y_len - y_sep;

  if (!(x_ramp == 0 || (x_ramp >= 2 && x_ramp == (1<<(ilog2(x_ramp)))))) {
    SCHRO_ERROR ("x_ramp not valid %d", x_ramp);
  }
  if (!(y_ramp == 0 || (y_ramp >= 2 && y_ramp == (1<<(ilog2(y_ramp)))))) {
    SCHRO_ERROR ("y_ramp not valid %d", y_ramp);
  }
  if (2*x_ramp > x_len) {
    SCHRO_ERROR ("x_ramp too large %d", x_ramp);
  }
  if (2*y_ramp > y_len) {
    SCHRO_ERROR ("y_ramp too large %d", y_ramp);
  }

  size = sizeof(int16_t) * x_len * y_len;
  obmc->region_data = malloc(size * 9 * 3);
  region = malloc(size);

  for(i=0;i<9;i++){
    obmc->regions[i].weights[0] = OFFSET(obmc->region_data, size * i);
    obmc->regions[i].weights[1] = OFFSET(obmc->region_data, size * (i+9));
    obmc->regions[i].weights[2] = OFFSET(obmc->region_data, size * (i+18));
    obmc->regions[i].end_x = x_len;
    obmc->regions[i].end_y = y_len;
  }

  obmc->shift = ramp_shift(x_ramp) + ramp_shift(y_ramp) + ref_shift;
  if (obmc->shift > 8) {
    SCHRO_ERROR("obmc shift too large (%d > 8)", obmc->shift);
  }

  obmc->x_ramp = x_ramp;
  obmc->y_ramp = y_ramp;
  obmc->x_len = x_len;
  obmc->y_len = y_len;
  obmc->x_sep = x_sep;
  obmc->y_sep = y_sep;

  obmc_calc (region, x_len, y_len, x_ramp, y_ramp);

  for(k=0;k<9;k++){
    memcpy(obmc->regions[k].weights[0], region, size);

    fixup_region (obmc, k);

    scale_region (obmc->regions[k].weights[1], obmc->regions[k].weights[0],
        x_len * y_len, ref1_weight);
    scale_region (obmc->regions[k].weights[2], obmc->regions[k].weights[0],
        x_len * y_len, ref2_weight);
    scale_region (obmc->regions[k].weights[0], obmc->regions[k].weights[0],
        x_len * y_len, ref1_weight+ref2_weight);
  }

  free(region);
}

void
schro_obmc_cleanup (SchroObmc *obmc)
{
  free(obmc->region_data);
}


/* original */

void
schro_motion_get_global_block (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, SchroGlobalMotion *gm, int refmask)
{
  SchroFrameData *comp;
  SchroFrame *srcframe;
  int offset;
  int i,j;
  int sx, sy;
  int persp;
  int w, h;
  uint8_t *dest;
  int stride;

  w = motion->obmc_luma->x_len;
  h = motion->obmc_luma->y_len;
  if (refmask == 1) {
    srcframe = motion->src1->frames[0];
  } else {
    srcframe = motion->src2->frames[0];
  }

  offset = 0;
  motion->blocks[0] = motion->tmpdata + offset;
  dest = motion->blocks[0];
  stride = w;
  motion->strides[0] = w;
  comp = &srcframe->components[0];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int nx, ny;
      nx = (x + i);
      ny = (y + j);
      persp = (1<<gm->c_exp) - gm->c0 * nx - gm->c1 * ny;
      sx = (persp * (gm->a00 * nx + gm->a01 * ny +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp + 3);
      sy = (persp * (gm->a10 * nx + gm->a11 * ny +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp + 3);
      sx = CLAMP(sx, 0, srcframe->width - 1);
      sy = CLAMP(sy, 0, srcframe->height - 1);
      dest[j*stride + i] = SCHRO_GET(comp->data, sy * comp->stride + sx, uint8_t);
    }
  }

  w >>= motion->params->video_format->chroma_h_shift;
  h >>= motion->params->video_format->chroma_v_shift;

  /* FIXME broken */

  offset += 64*64;
  motion->blocks[1] = motion->tmpdata + offset;
  dest = motion->blocks[1];
  stride = w;
  motion->strides[1] = w;
  comp = &srcframe->components[1];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int si, sj;
      int nx, ny;
      si = i << motion->params->video_format->chroma_h_shift;
      sj = j << motion->params->video_format->chroma_h_shift;
      nx = (x + i);
      ny = (y + j);
      persp = (1<<gm->c_exp) - gm->c0 * nx - gm->c1 * ny;
      sx = (persp * (gm->a00 * nx + gm->a01 * ny +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp + 3);
      sy = (persp * (gm->a10 * nx + gm->a11 * ny +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp + 3);
      sx = CLAMP(sx, 0, srcframe->width - 1);
      sy = CLAMP(sy, 0, srcframe->height - 1);
      sx >>= motion->params->video_format->chroma_h_shift;
      sy >>= motion->params->video_format->chroma_v_shift;
      dest[j*stride + i] = SCHRO_GET(comp->data, sy * comp->stride + sx, uint8_t);
    }
  }

  offset += 64*64;
  motion->blocks[2] = motion->tmpdata + offset;
  dest = motion->blocks[2];
  stride = w;
  motion->strides[2] = w;
  comp = &srcframe->components[2];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int si, sj;
      int nx, ny;
      si = i << motion->params->video_format->chroma_h_shift;
      sj = j << motion->params->video_format->chroma_h_shift;
      nx = (x + i);
      ny = (y + j);
      persp = (1<<gm->c_exp) - gm->c0 * nx - gm->c1 * ny;
      sx = (persp * (gm->a00 * nx + gm->a01 * ny +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp + 3);
      sy = (persp * (gm->a10 * nx + gm->a11 * ny +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp + 3);
      sx = CLAMP(sx, 0, srcframe->width - 1);
      sy = CLAMP(sy, 0, srcframe->height - 1);
      sx >>= motion->params->video_format->chroma_h_shift;
      sy >>= motion->params->video_format->chroma_v_shift;
      dest[j*stride + i] = SCHRO_GET(comp->data, sy * comp->stride + sx, uint8_t);
    }
  }
}

void
schro_motion_get_dc_block (SchroMotion *motion, SchroMotionVector *mv)
{
  int offset;
  SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

  offset = 0;
  memset (motion->tmpdata + offset, mvdc->dc[0], motion->obmc_luma->x_len);
  motion->blocks[0] = motion->tmpdata + offset;
  motion->strides[0] = 0;
  offset += motion->obmc_luma->x_len;

  memset (motion->tmpdata + offset, mvdc->dc[1], motion->obmc_chroma->x_len);
  motion->blocks[1] = motion->tmpdata + offset;
  motion->strides[1] = 0;
  offset += motion->obmc_chroma->x_len;

  memset (motion->tmpdata + offset, mvdc->dc[2], motion->obmc_chroma->x_len);
  motion->blocks[2] = motion->tmpdata + offset;
  motion->strides[2] = 0;
}

void
schro_motion_get_block (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, int refmask)
{
  uint8_t *data;
  int stride;
  int i,j;
  SchroFrame *srcframe;
  SchroFrameData *comp;
  int mx, my;
  int sx, sy;
  int w, h;
  int upsample_index;

  if (refmask & 1) {
    mx = mv->x1<<(3-motion->mv_precision);
    my = mv->y1<<(3-motion->mv_precision);
    sx = x + (mx>>3);
    sy = y + (my>>3);
    upsample_index = (mx&4)>>2 | (my&4)>>1;
    srcframe = motion->src1->frames[upsample_index];
  } else {
    mx = mv->x2<<(3-motion->mv_precision);
    my = mv->y2<<(3-motion->mv_precision);
    sx = x + (mx>>3);
    sy = y + (my>>3);
    upsample_index = (mx&4)>>2 | (my&4)>>1;
    srcframe = motion->src2->frames[upsample_index];
  }
  w = motion->obmc_luma->x_len;
  h = motion->obmc_luma->y_len;

  SCHRO_ASSERT(srcframe);
  if (sx & 3 || sy & 3) {
#if 0
    motion->blocks[0] = motion->tmpdata;
    motion->strides[0] = 64;
    data = motion->blocks[0];
    stride = motion->strides[0];
    for(j=0;j<h;j++){
      for(i=0;i<w;i++){
        int x = 0;
        int src_x = CLAMP(sx + i, 0, comp->width - 1);
        int src_y = CLAMP(sy + j, 0, comp->height - 1);
        int dx, dy;

        dx = (mv->x1 >> 2)&1;
        dy = (mv->y1 >> 2)&1;

        srcframe = motion->src1->frames[upsample_index];
        comp = &srcframe->components[0];
        x += factor[0*16 + sx + sy*4] *
          SCHRO_GET(comp->data, comp->stride * src_y + src_x, uint8_t);

        data[j*stride + i] = (x + 8)>>4;
      }
    }
#endif

    
  }

  /* FIXME move and fix */
  motion->sx_max = srcframe->width - motion->obmc_luma->x_len;
  motion->sy_max = srcframe->height - motion->obmc_luma->y_len;

  if (sx < 0 || sy < 0 || sx > motion->sx_max || sy > motion->sy_max) {
    motion->blocks[0] = motion->tmpdata;
    motion->strides[0] = 64;
    data = motion->blocks[0];
    comp = &srcframe->components[0];
    stride = motion->strides[0];
    for(j=0;j<h;j++){
      for(i=0;i<w;i++){
        int src_x = CLAMP(sx + i, 0, comp->width - 1);
        int src_y = CLAMP(sy + j, 0, comp->height - 1);
        data[j*stride + i] =
          SCHRO_GET(comp->data, comp->stride * src_y + src_x, uint8_t);
      }
    }

    w >>= motion->params->video_format->chroma_h_shift;
    h >>= motion->params->video_format->chroma_v_shift;
    sx >>= motion->params->video_format->chroma_h_shift;
    sy >>= motion->params->video_format->chroma_v_shift;

    motion->blocks[1] = motion->tmpdata + 64*64;
    motion->strides[1] = 64;
    data = motion->blocks[1];
    comp = &srcframe->components[1];
    stride = motion->strides[1];
    for(j=0;j<h;j++){
      for(i=0;i<w;i++){
        int src_x = CLAMP(sx + i, 0, comp->width - 1);
        int src_y = CLAMP(sy + j, 0, comp->height - 1);
        data[j*stride + i] =
          SCHRO_GET(comp->data, comp->stride * src_y + src_x, uint8_t);
      }
    }
    motion->blocks[2] = motion->tmpdata + 64*64*2;
    motion->strides[2] = 64;
    data = motion->blocks[2];
    comp = &srcframe->components[2];
    stride = motion->strides[2];
    for(j=0;j<h;j++){
      for(i=0;i<w;i++){
        int src_x = CLAMP(sx + i, 0, comp->width - 1);
        int src_y = CLAMP(sy + j, 0, comp->height - 1);
        data[j*stride + i] =
          SCHRO_GET(comp->data, comp->stride * src_y + src_x, uint8_t);
      }
    }
    return;
  }

  comp = &srcframe->components[0];
  motion->blocks[0] = OFFSET(comp->data, comp->stride * sy + sx);
  motion->strides[0] = comp->stride;

  sx >>= motion->params->video_format->chroma_h_shift;
  sy >>= motion->params->video_format->chroma_v_shift;
  comp = &srcframe->components[1];
  motion->blocks[1] = OFFSET(comp->data, comp->stride * sy + sx);
  motion->strides[1] = comp->stride;

  comp = &srcframe->components[2];
  motion->blocks[2] = OFFSET(comp->data, comp->stride * sy + sx);
  motion->strides[2] = comp->stride;

}


/* generic */

static int weights[64] = {
  16, 12,  8,  4,
  12,  9,  6,  3,
   8,  6,  4,  2,
   4,  3,  2,  1,

   0,  4,  8, 12,
   0,  3,  6,  9,
   0,  2,  4,  6,
   0,  1,  2,  3,

   0,  0,  0,  0,
   4,  3,  2,  1,
   8,  6,  4,  2,
  12,  9,  6,  3,

   0,  0,  0,  0,
   0,  1,  2,  3,
   0,  2,  4,  6,
   0,  3,  6,  9,
};

int
get_pixel_generic (SchroUpsampledFrame *upframe, int x, int y, int comp_index)
{
  int v = 0;
  SchroFrameData *comp;
  int upsample_index;
  SchroFrame *srcframe;
  int sx, sy;

  upsample_index = (x&4)>>2 | (y&4)>>1;
  srcframe = upframe->frames[upsample_index];
  comp = &srcframe->components[comp_index];
  sx = CLAMP(x>>3, 0, comp->width - 1);
  sy = CLAMP(y>>3, 0, comp->height - 1);
  v += weights[0*16 + (x&3) + (y&3)*4] *
    SCHRO_GET(comp->data, comp->stride * sy + sx, uint8_t);

  upsample_index = ((x+4)&4)>>2 | (y&4)>>1;
  srcframe = upframe->frames[upsample_index];
  comp = &srcframe->components[comp_index];
  sx = CLAMP((x+4)>>3, 0, comp->width - 1);
  sy = CLAMP(y>>3, 0, comp->height - 1);
  v += weights[1*16 + (x&3) + (y&3)*4] *
    SCHRO_GET(comp->data, comp->stride * sy + sx, uint8_t);

  upsample_index = (x&4)>>2 | ((y+4)&4)>>1;
  srcframe = upframe->frames[upsample_index];
  comp = &srcframe->components[comp_index];
  sx = CLAMP(x>>3, 0, comp->width - 1);
  sy = CLAMP((y+4)>>3, 0, comp->height - 1);
  v += weights[2*16 + (x&3) + (y&3)*4] *
    SCHRO_GET(comp->data, comp->stride * sy + sx, uint8_t);

  upsample_index = ((x+4)&4)>>2 | ((y+4)&4)>>1;
  srcframe = upframe->frames[upsample_index];
  comp = &srcframe->components[comp_index];
  sx = CLAMP((x+4)>>3, 0, comp->width - 1);
  sy = CLAMP((y+4)>>3, 0, comp->height - 1);
  v += weights[3*16 + (x&3) + (y&3)*4] *
    SCHRO_GET(comp->data, comp->stride * sy + sx, uint8_t);

  return (v+8)>>4;
}

void
schro_motion_get_global_block_generic (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, SchroGlobalMotion *gm, int refmask)
{
  SchroUpsampledFrame *srcframe;
  int offset;
  int i,j;
  int sx, sy;
  int persp;
  int w, h;
  uint8_t *dest;
  int stride;

  w = motion->obmc_luma->x_len;
  h = motion->obmc_luma->y_len;
  if (refmask == 1) {
    srcframe = motion->src1;
  } else {
    srcframe = motion->src2;
  }

  offset = 0;
  motion->blocks[0] = motion->tmpdata + offset;
  dest = motion->blocks[0];
  stride = w;
  motion->strides[0] = w;
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      persp = (1<<gm->c_exp) - gm->c0 * (x + i) - gm->c1 * (y + j);
      sx = (persp * (gm->a00 * (x + i) + gm->a01 * (y + j) +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp);
      sy = (persp * (gm->a10 * (x + i) + gm->a11 * (y + j) +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp);
      dest[j*stride + i] = get_pixel_generic (srcframe, sx, sy, 0);
    }
  }

  w >>= motion->params->video_format->chroma_h_shift;
  h >>= motion->params->video_format->chroma_v_shift;

  /* FIXME broken */

  offset += 64*64;
  motion->blocks[1] = motion->tmpdata + offset;
  dest = motion->blocks[1];
  stride = w;
  motion->strides[1] = w;
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int si, sj;
      si = i << motion->params->video_format->chroma_h_shift;
      sj = j << motion->params->video_format->chroma_h_shift;
      persp = (1<<gm->c_exp) - gm->c0 * (x + si) - gm->c1 * (y + sj);
      sx = (persp * (gm->a00 * (x + si) + gm->a01 * (y + sj) +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp);
      sy = (persp * (gm->a10 * (x + si) + gm->a11 * (y + sj) +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp);
      sx >>= motion->params->video_format->chroma_h_shift;
      sy >>= motion->params->video_format->chroma_v_shift;
      dest[j*stride + i] = get_pixel_generic (srcframe, sx, sy, 1);
    }
  }

  offset += 64*64;
  motion->blocks[2] = motion->tmpdata + offset;
  dest = motion->blocks[2];
  stride = w;
  motion->strides[2] = w;
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int si, sj;
      si = i << motion->params->video_format->chroma_h_shift;
      sj = j << motion->params->video_format->chroma_h_shift;
      persp = (1<<gm->c_exp) - gm->c0 * (x + si) - gm->c1 * (y + sj);
      sx = (persp * (gm->a00 * (x + si) + gm->a01 * (y + sj) +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp);
      sy = (persp * (gm->a10 * (x + si) + gm->a11 * (y + sj) +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp);
      sx >>= motion->params->video_format->chroma_h_shift;
      sy >>= motion->params->video_format->chroma_v_shift;
      dest[j*stride + i] = get_pixel_generic (srcframe, sx, sy, 2);
    }
  }
}

void
schro_motion_get_block_generic (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, int refmask)
{
  uint8_t *data;
  int stride;
  int i,j;
  SchroUpsampledFrame *srcframe;
  int sx, sy;
  int w, h;

  if (refmask & 1) {
    sx = (x<<3) + mv->x1;
    sy = (y<<3) + mv->y1;
    srcframe = motion->src1;
  } else {
    sx = (x<<3) + mv->x2;
    sy = (y<<3) + mv->y2;
    srcframe = motion->src2;
  }
  w = motion->obmc_luma->x_len;
  h = motion->obmc_luma->y_len;

  SCHRO_ASSERT(srcframe);

  motion->blocks[0] = motion->tmpdata;
  motion->strides[0] = 64;
  data = motion->blocks[0];
  stride = motion->strides[0];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      data[j*stride + i] = get_pixel_generic (srcframe, sx + i*8, sy + j*8, 0);
    }
  }

  sx >>= motion->params->video_format->chroma_h_shift;
  sy >>= motion->params->video_format->chroma_v_shift;
  w >>= motion->params->video_format->chroma_h_shift;
  h >>= motion->params->video_format->chroma_v_shift;

  motion->blocks[1] = motion->tmpdata + 64*64;
  motion->strides[1] = 64;
  data = motion->blocks[1];
  stride = motion->strides[1];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      data[j*stride + i] = get_pixel_generic (srcframe, sx + i*8, sy + j*8, 1);
    }
  }
  motion->blocks[2] = motion->tmpdata + 64*64*2;
  motion->strides[2] = 64;
  data = motion->blocks[2];
  stride = motion->strides[2];
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      data[j*stride + i] = get_pixel_generic (srcframe, sx + i*8, sy + j*8, 2);
    }
  }
}


static void
copy_block (SchroFrame *dest, SchroMotion *motion, int x, int y, int reg,
    int weight_index)
{
  SchroFrameData *comp;
  SchroObmc *obmc;
  SchroObmcRegion *region;
  int k;

  for (k = 0; k < 3; k++) {
    int x0, y0;

    comp = &dest->components[k];
    if (k == 0) {
      obmc = motion->obmc_luma;
    } else {
      obmc = motion->obmc_chroma;
    }
    region = obmc->regions + reg;
    x0 = (x>>comp->h_shift);
    y0 = (y>>comp->v_shift);
    
    if (region->end_x - region->start_x == 12) {
      int16_t *d1;
      int16_t *w1;
      uint8_t *s1;

      d1 = OFFSET(comp->data, comp->stride*(y0 + region->start_y) +
          2*(x0 + region->start_x));
      w1 = region->weights[weight_index] +
        obmc->x_len * region->start_y + region->start_x;
      s1 = OFFSET(motion->blocks[k],
          motion->strides[k]*region->start_y + region->start_x);

      oil_multiply_and_acc_12xn_s16_u8 (d1, comp->stride,
          w1, obmc->x_len * sizeof(int16_t),
          s1, motion->strides[k],
          region->end_y - region->start_y);
    } else {
      int j;
      for(j=region->start_y;j<region->end_y;j++){
        oil_multiply_and_add_s16_u8 (
            OFFSET(comp->data, comp->stride*(y0+j) + 2*(x0 + region->start_x)),
            OFFSET(comp->data, comp->stride*(y0+j) + 2*(x0 + region->start_x)),
            OFFSET(region->weights[weight_index],
              sizeof(int16_t) * (obmc->x_len*j + region->start_x)),
            OFFSET(motion->blocks[k], motion->strides[k]*j + region->start_x),
            region->end_x - region->start_x);
      }
    }
  }
}
    
static void
clear_rows (SchroFrame *frame, int y, int n)
{
  SchroFrameData *comp;
  uint8_t zero = 0;
  int ymin, ymax;
  int k;
  int j;

  for(k=0;k<3;k++){
    comp = &frame->components[k];
    ymax = MIN ((y + n)>>comp->v_shift, comp->height);
    ymin = MAX (y>>comp->v_shift, 0);
    for(j=ymin;j<ymax;j++){
      oil_splat_u8_ns (OFFSET(comp->data, j * comp->stride), &zero,
            comp->width * sizeof(int16_t));
    }
  }
}

static void
shift_rows (SchroFrame *frame, int y, int n, int shift_luma, int shift_chroma)
{
  SchroFrameData *comp;
  int ymin, ymax;
  uint16_t *data;
  int16_t s[2];
  int k;
  int j;

  for(k=0;k<3;k++){
    comp = &frame->components[k];
    if (k == 0) {
      s[1] = shift_luma;
    } else {
      s[1] = shift_chroma;
    }
    s[0] = ((1<<s[1])>>1) - (128<<s[1]);

    ymax = MIN ((y + n)>>comp->v_shift, comp->height);
    ymin = MAX (y>>comp->v_shift, 0);
    for(j=ymin;j<ymax;j++){
      data = OFFSET(comp->data, j * comp->stride);
      oil_addc_rshift_u16(data, data, s, comp->width);
    }
  }
}

void
schro_motion_render (SchroMotion *motion, SchroFrame *dest)
{
  int i, j;
  int x, y;
  SchroObmc *obmc_luma;
  SchroObmc *obmc_chroma;
  SchroMotionVector *motion_vectors = motion->motion_vectors;
  SchroParams *params = motion->params;

  if (_schro_motion_ref) {
    return schro_motion_render_ref (motion, dest);
  }

  motion->mv_precision = params->mv_precision;

  obmc_luma = malloc(sizeof(*obmc_luma));
  schro_obmc_init (obmc_luma,
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma,
      params->picture_weight_1, params->picture_weight_2,
      params->picture_weight_bits);
  obmc_chroma = malloc(sizeof(*obmc_chroma));
  schro_obmc_init (obmc_chroma,
      params->xblen_luma>>motion->params->video_format->chroma_h_shift,
      params->yblen_luma>>motion->params->video_format->chroma_v_shift,
      params->xbsep_luma>>motion->params->video_format->chroma_h_shift,
      params->ybsep_luma>>motion->params->video_format->chroma_v_shift,
      params->picture_weight_1, params->picture_weight_2,
      params->picture_weight_bits);
  motion->obmc_luma = obmc_luma;
  motion->obmc_chroma = obmc_chroma;
  motion->tmpdata = malloc (64*64*3);

  clear_rows (dest, 0, obmc_luma->y_ramp/2);

  for(j=0;j<params->y_num_blocks;j++){
    int region_y;

    y = j*obmc_luma->y_sep - obmc_luma->y_ramp/2;
    if (j == 0) {
      region_y = 0;
    } else if (j == params->y_num_blocks - 1) {
      region_y = 6;
    } else {
      region_y = 3;
    }

    clear_rows (dest, y + obmc_luma->y_ramp, obmc_luma->y_sep);

    for(i=0;i<params->x_num_blocks;i++){
      int region;
      SchroMotionVector *mv = &motion_vectors[j*params->x_num_blocks + i];

      x = i*obmc_luma->x_sep - obmc_luma->x_ramp/2;
      if (i == 0) {
        region = region_y + 0;
      } else if (i == params->x_num_blocks - 1) {
        region = region_y + 2;
      } else {
        region = region_y + 1;
      }

      switch (mv->pred_mode) {
        case 0:
          schro_motion_get_dc_block (motion, mv);
          copy_block (dest, motion, x, y, region, 0);
          break;
        case 1:
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[0];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 1);
          } else {
            schro_motion_get_block (motion, mv, x, y, 1);
          }
          copy_block (dest, motion, x, y, region, 0);
          break;
        case 2:
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[1];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 2);
          } else {
            schro_motion_get_block (motion, mv, x, y, 2);
          }
          copy_block (dest, motion, x, y, region, 0);
          break;
        case 3:
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[0];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 1);
          } else {
            schro_motion_get_block (motion, mv, x, y, 1);
          }
          copy_block (dest, motion, x, y, region, 1);
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[1];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 2);
          } else {
            schro_motion_get_block (motion, mv, x, y, 2);
          }
          copy_block (dest, motion, x, y, region, 2);
          break;
        default:
          break;
      }
    }

    shift_rows (dest, y - obmc_luma->y_ramp/2, obmc_luma->y_sep,
        obmc_luma->shift, obmc_chroma->shift);
  }

  y = params->y_num_blocks*obmc_luma->y_sep - obmc_luma->y_ramp/2;
  shift_rows (dest, y - obmc_luma->y_ramp/2, obmc_luma->y_ramp,
      obmc_luma->shift, obmc_chroma->shift);

  schro_obmc_cleanup (obmc_luma);
  free(obmc_luma);
  schro_obmc_cleanup (obmc_chroma);
  free(obmc_chroma);
  free(motion->tmpdata);
}

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
        pred[i] = sum;
        break;
      case 2:
        pred[i] = (sum+1)/2;
        break;
      case 3:
        pred[i] = (sum+1)/3;
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
      if (mode == 1) {
        vx[n] = mv->x1;
        vy[n] = mv->y1;
      } else {
        vx[n] = mv->x2;
        vy[n] = mv->y2;
      }
      n++;
    }
  }
  if (y>0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x,y-1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      if (mode == 1) {
        vx[n] = mv->x1;
        vy[n] = mv->y1;
      } else {
        vx[n] = mv->x2;
        vy[n] = mv->y2;
      }
      n++;
    }
  }
  if (x>0 && y>0) {
    mv = SCHRO_MOTION_GET_BLOCK(motion,x-1,y-1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      if (mode == 1) {
        vx[n] = mv->x1;
        vy[n] = mv->y1;
      } else {
        vx[n] = mv->x2;
        vy[n] = mv->y2;
      }
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

  for(y=0;y<params->y_num_blocks;y++){
    for(x=0;x<params->x_num_blocks;x++){
      mv = &motion->motion_vectors[y*params->x_num_blocks + x];
      sbmv = &motion->motion_vectors[(y&~3)*params->x_num_blocks + (x&~3)];

      switch (sbmv->split) {
        case 0:
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
            SCHRO_ERROR("mv(%d,%d) not equal to superblock mv", x, y);
            return 0;
          }
          break;
        case 1:
          bmv = &motion->motion_vectors[(y&~1)*params->x_num_blocks + (x&~1)];
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
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

      if (mv->pred_mode == 0) {
        /* hard to screw this one up */
      } else {
        if ((mv->pred_mode & 2) && motion->src2->frames[0] == NULL) {
          SCHRO_ERROR("mv(%d,%d) uses non-existent src2", x, y);
          return 0;
        }
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

void
schro_upsampled_frame_upsample (SchroUpsampledFrame *df)
{
  if (df->frames[1]) return;

  df->frames[1] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  df->frames[2] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  df->frames[3] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  schro_frame_upsample_horiz (df->frames[1], df->frames[0]);
  schro_frame_upsample_vert (df->frames[2], df->frames[0]);
  schro_frame_upsample_horiz (df->frames[3], df->frames[2]);
}

SchroUpsampledFrame *
schro_upsampled_frame_new (SchroFrame *frame)
{
  SchroUpsampledFrame *df;

  df = malloc(sizeof(SchroUpsampledFrame));
  memset (df, 0, sizeof(*df));

  df->frames[0] = frame;

  return df;
}

void
schro_upsampled_frame_free (SchroUpsampledFrame *df)
{
  int i;
  for(i=0;i<4;i++){
    if (df->frames[i]) {
      schro_frame_unref (df->frames[i]);
    }
  }
  free(df);
}


//#define EXP_16NOV07

static int get_dc_pixel (SchroMotion *motion, int i, int j, int k, int x, int y);
static int get_ref1_pixel (SchroMotion *motion, int i, int j, int k, int x, int y);
static int get_ref2_pixel (SchroMotion *motion, int i, int j, int k, int x, int y);
static int get_biref_pixel (SchroMotion *motion, int i, int j, int k, int x, int y);
static int get_pixel (SchroMotion *motion, int k, SchroUpsampledFrame *upframe,
    int x, int y, int dx, int dy);
static void get_global_mv (SchroMotion *motion, int ref, int x, int y, int *dx,
    int *dy);
static int up2by2 (SchroUpsampledFrame *upframe, int k, int x, int y);
static int pixel_predict (SchroMotion *motion, int x, int y, int k);
static int pixel_predict_block (SchroMotion *motion, int x, int y, int k, int i, int j);
static int get_shift (int offset);

void
schro_motion_render_ref (SchroMotion *motion, SchroFrame *dest)
{
  SchroParams *params = motion->params;
  int k;
  int x,y;
  int16_t *line;

  SCHRO_ASSERT(motion->src1->frames[0]);
  SCHRO_ASSERT(motion->src1->frames[1]);
  SCHRO_ASSERT(motion->src1->frames[2]);
  SCHRO_ASSERT(motion->src1->frames[3]);

  motion->dc_weight = 1<<params->picture_weight_bits;
  motion->biref1_weight = params->picture_weight_1;
  motion->biref2_weight = params->picture_weight_2;
  motion->ref_weight = params->picture_weight_1 + params->picture_weight_2;

  motion->mv_precision = params->mv_precision;

  for(k=0;k<3;k++){
    SchroFrameData *comp = dest->components + k;

    if (k == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
    } else {
      motion->xbsep = params->xbsep_luma >> params->video_format->chroma_h_shift;
      motion->ybsep = params->ybsep_luma >> params->video_format->chroma_v_shift;
      motion->xblen = params->xblen_luma >> params->video_format->chroma_h_shift;
      motion->yblen = params->yblen_luma >> params->video_format->chroma_v_shift;
    }
    motion->xoffset = (motion->xblen - motion->xbsep)/2;
    motion->yoffset = (motion->yblen - motion->ybsep)/2;
#ifdef EXP_16NOV07
    motion->shift = params->picture_weight_bits + 6;
    get_shift(0);
#else
    motion->shift = params->picture_weight_bits +
      get_shift (motion->xoffset) + get_shift(motion->yoffset);
#endif

    for(y=0;y<comp->height;y++){
      line = OFFSET(comp->data, y*comp->stride);
      for(x=0;x<comp->width;x++){
        line[x] = CLAMP(pixel_predict (motion, x, y, k), 0, 255);

        /* FIXME */
        line[x] -= 128;
      }
    }
  }
}

static int
get_shift (int offset)
{
  switch (offset) {
    case 0:
      return 0;
    case 1:
      return 2;
    case 2:
      return 3;
    case 4:
      return 4;
    case 8:
      return 5;
  }
  SCHRO_ERROR("%d", offset);
  SCHRO_ASSERT(0);
  return 0;
}

static int pixel_predict (SchroMotion *motion, int x, int y, int k)
{
  int i,j;
  int value;

  i = (x + motion->xbsep/2) / motion->xbsep - 1;
  j = (y + motion->ybsep/2) / motion->ybsep - 1;

  value = pixel_predict_block (motion, x, y, k, i, j);
  value += pixel_predict_block (motion, x, y, k, i + 1, j);
  value += pixel_predict_block (motion, x, y, k, i, j + 1);
  value += pixel_predict_block (motion, x, y, k, i + 1, j + 1);

  return ROUND_SHIFT(value, motion->shift);
}

static int
pixel_predict_block (SchroMotion *motion, int x, int y, int k, int i, int j)
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

#ifdef EXP_16NOV07
  if (motion->xoffset == 0) {
    wx = 8;
  } else if (x < motion->xoffset || x >= width - motion->xoffset) {
    wx = 8;
  } else if (x - xmin < 2*motion->xoffset) {
    wx = 1 + (6*(x-xmin) + motion->xoffset - 1)/(2*motion->xoffset - 1);
  } else if (xmax - 1 - x < 2*motion->xoffset) {
    wx = 7 - (6*(x-xmax+2*motion->xoffset) + motion->xoffset - 1)/(2*motion->xoffset - 1);
  } else {
    wx = 8;
  }

  if (motion->yoffset == 0) {
    wy = 8;
  } else if (y < motion->yoffset || y >= width - motion->yoffset) {
    wy = 8;
  } else if (y - ymin < 2*motion->yoffset) {
    wy = 1 + (6*(y-ymin) + motion->yoffset - 1)/(2*motion->yoffset - 1);
  } else if (ymax - 1 - y < 2*motion->yoffset) {
    wy = 7 - (6*(y-ymax+2*motion->yoffset) + motion->yoffset - 1)/(2*motion->yoffset - 1);
  } else {
    wy = 8;
  }
#else
  if (motion->xoffset == 0) {
    wx = 1;
  } else if (x < motion->xoffset || x >= width - motion->xoffset) {
    wx = motion->xoffset * 4;
  } else if (x - xmin < 2*motion->xoffset) {
    wx = (x - xmin) * 2 + 1;
  } else if (xmax - 1 - x < 2*motion->xoffset) {
    wx = (xmax - 1 - x) * 2 + 1;
  } else {
    wx = motion->xoffset * 4;
  }

  if (motion->yoffset == 0) {
    wy = 1;
  } else if (y < motion->yoffset || y >= height - motion->yoffset) {
    wy = motion->yoffset * 4;
  } else if (y - ymin < 2*motion->yoffset) {
    wy = (y - ymin) * 2 + 1;
  } else if (ymax - 1 - y < 2*motion->yoffset) {
    wy = (ymax - 1 - y) * 2 + 1;
  } else {
    wy = motion->yoffset * 4;
  }
#endif

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

static int
get_dc_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVectorDC *mvdc;

  mvdc = (SchroMotionVectorDC *)
    &motion->motion_vectors[j*params->x_num_blocks + i];

  return mvdc->dc[k] * motion->dc_weight;
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
    get_global_mv (motion, 0, x, y, &dx, &dy);
  } else {
    dx = mv->x1;
    dy = mv->y1;
  }

  value = get_pixel (motion, k, motion->src1, x, y, dx, dy);
  return value * motion->ref_weight;
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
    get_global_mv (motion, 1, x, y, &dx, &dy);
  } else {
    dx = mv->x2;
    dy = mv->y2;
  }
  value = get_pixel (motion, k, motion->src2, x, y, dx, dy);

  return value * motion->ref_weight;
}

static int
get_biref_pixel (SchroMotion *motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value1;
  int value2;
  int dx1, dx2, dy1, dy2;

  mv = &motion->motion_vectors[j*params->x_num_blocks + i];
  if (mv->using_global) {
    get_global_mv (motion, 0, x, y, &dx1, &dy1);
    get_global_mv (motion, 1, x, y, &dx2, &dy2);
  } else {
    dx1 = mv->x1;
    dy1 = mv->y1;
    dx2 = mv->x2;
    dy2 = mv->y2;
  }
  value1 = get_pixel (motion, k, motion->src1, x, y, dx1, dy1);
  value2 = get_pixel (motion, k, motion->src2, x, y, dx2, dy2);

  return value1 * motion->biref1_weight + value2 * motion->biref2_weight;
}

static void
get_global_mv (SchroMotion *motion, int ref, int x, int y, int *dx,
    int *dy)
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
get_pixel (SchroMotion *motion, int k, SchroUpsampledFrame *upframe,
    int x, int y, int dx, int dy)
{
  int px, py;
  int hx, hy;
  int rx, ry;
  int w00, w01, w10, w11;
  int shift;
  int value;

  if (motion->mv_precision > 0) {
    px = (x << motion->mv_precision) + dx;
    py = (y << motion->mv_precision) + dy;
  } else {
    px = (x + dx) << 1;
    py = (y + dy) << 1;
  }

  if (motion->mv_precision < 2) {
    return up2by2(upframe, k, px, py);
  }

  shift = motion->mv_precision - 1;
  hx = px >> shift;
  hy = py >> shift;

  rx = px - (hx << shift);
  ry = py - (hy << shift);

  w00 = ((1<<shift) - ry) * ((1<<shift) - rx);
  w01 = ((1<<shift) - ry) * rx;
  w10 = ry * ((1<<shift) - rx);
  w11 = ry * rx;

  value = w00 * up2by2 (upframe, k, hx, hy);
  value += w01 * up2by2 (upframe, k, hx + 1, hy);
  value += w10 * up2by2 (upframe, k, hx, hy + 1);
  value += w11 * up2by2 (upframe, k, hx + 1, hy + 1);

  return ROUND_SHIFT(value, shift-1);
}

static int
up2by2 (SchroUpsampledFrame *upframe, int k, int x, int y)
{
  SchroFrameData *comp;
  uint8_t *line;
  int i;
  
  comp = upframe->frames[0]->components + k;
  x = CLAMP(x, 0, comp->width * 2 - 1);
  y = CLAMP(y, 0, comp->height * 2 - 1);

  i = ((y&1)<<1) | (x&1);
  x >>= 1;
  y >>= 1;

  comp = upframe->frames[i]->components + k;
  line = OFFSET(comp->data, y*comp->stride);

  return line[x];
}



