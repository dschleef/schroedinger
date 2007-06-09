
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
//#include <stdlib.h>
#include <string.h>
//#include <stdio.h>
#include <schroedinger/schrooil.h>


void schro_decoder_predict (SchroDecoder *decoder);


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

void
schro_obmc_init (SchroObmc *obmc, int x_len, int y_len, int x_sep, int y_sep)
{
  int i;
  int j;
  int k;
  int x_ramp;
  int y_ramp;

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

  obmc->stride = sizeof(int16_t) * x_len;
  obmc->region_data = malloc(obmc->stride * y_len * 9);
  obmc->tmpdata = malloc(x_len * y_len);

  for(i=0;i<9;i++){
    obmc->regions[i].weights = OFFSET(obmc->region_data,
        obmc->stride * y_len * i);
    obmc->regions[i].end_x = x_len;
    obmc->regions[i].end_y = y_len;
  }

  obmc->shift = ramp_shift(x_ramp) + ramp_shift(y_ramp);
  if (obmc->shift > 8) {
    SCHRO_ERROR("obmc shift too large (%d > 8)", obmc->shift);
  }

  obmc->x_ramp = x_ramp;
  obmc->y_ramp = y_ramp;
  obmc->x_len = x_len;
  obmc->y_len = y_len;
  obmc->x_sep = x_sep;
  obmc->y_sep = y_sep;

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
      obmc->regions[0].weights[i] = w;
    }
  } else {
    for(i=0;i<x_len;i++){
      obmc->regions[0].weights[i] = 1;
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
      SCHRO_GET(obmc->regions[0].weights, obmc->stride * j, int16_t) = w;
    }
  } else {
    for(j=0;j<y_len;j++){
      SCHRO_GET(obmc->regions[0].weights, obmc->stride * j, int16_t) = 1;
    }
  }

  for(j=1;j<y_len;j++){
    for(i=1;i<x_len;i++){
      SCHRO_GET(obmc->regions[0].weights, obmc->stride*j + 2*i, int16_t) =
        SCHRO_GET(obmc->regions[0].weights, obmc->stride*j, int16_t) *
        obmc->regions[0].weights[i];
    }
  }
  for(i=1;i<9;i++){
    memcpy(obmc->regions[i].weights, obmc->regions[0].weights,
        obmc->stride*y_len);
  }

  /* fix up top */
  for(k=0;k<3;k++){
    for(j=0;j<y_ramp;j++){
      for(i=0;i<x_len;i++){
        SCHRO_GET(obmc->regions[k].weights, obmc->stride*j + 2*i, int16_t) +=
          SCHRO_GET(obmc->regions[k].weights,
              obmc->stride*(y_len - y_ramp + j) + 2*i, int16_t);
      }
    }
    obmc->regions[k].start_y = y_ramp/2;
  }
  /* fix up bottom */
  for(k=6;k<9;k++){
    for(j=0;j<y_ramp;j++){
      for(i=0;i<x_len;i++){
        SCHRO_GET(obmc->regions[k].weights,
            obmc->stride*(y_len - y_ramp + j) + 2*i, int16_t) += 
          SCHRO_GET(obmc->regions[k].weights, obmc->stride*j + 2*i, int16_t);
      }
    }
    obmc->regions[k].end_y = y_len - y_ramp/2;
  }
  /* fix up left */
  for(k=0;k<9;k+=3){
    for(j=0;j<y_len;j++){
      for(i=0;i<x_ramp;i++){
        SCHRO_GET(obmc->regions[k].weights, obmc->stride*j + 2*i, int16_t) +=
          SCHRO_GET(obmc->regions[k].weights, obmc->stride*j + 2*(x_len - x_ramp + i),
              int16_t);
      }
    }
    obmc->regions[k].start_x = x_ramp/2;
  }
  /* fix up right */
  for(k=2;k<9;k+=3){
    for(j=0;j<y_len;j++){
      for(i=0;i<x_ramp;i++){
        SCHRO_GET(obmc->regions[k].weights,
            obmc->stride*j + 2*(x_len - x_ramp + i), int16_t) += 
          SCHRO_GET(obmc->regions[k].weights, obmc->stride*j + 2*i, int16_t);
      }
    }
    obmc->regions[k].end_x = x_len - x_ramp/2;
  }

  /* fix up pointers */
  for(k=0;k<9;k++){
    obmc->regions[k].weights = OFFSET(obmc->regions[k].weights,
        obmc->stride * obmc->regions[k].start_y +
        sizeof(int16_t) * obmc->regions[k].start_x);
  }
}

void
schro_obmc_cleanup (SchroObmc *obmc)
{
  free(obmc->region_data);
  free(obmc->tmpdata);
}


/* original */

void
schro_motion_get_global_block (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, SchroGlobalMotion *gm, int refmask)
{
  SchroFrameComponent *comp;
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
  SchroFrameComponent *comp;
  int sx, sy;
  int w, h;
  int upsample_index;

  if (refmask & 1) {
    sx = x + (mv->x1>>3);
    sy = y + (mv->y1>>3);
    upsample_index = (mv->x1&4)>>2 | (mv->y1&4)>>1;
    srcframe = motion->src1->frames[upsample_index];
  } else {
    sx = x + (mv->x2>>3);
    sy = y + (mv->y2>>3);
    upsample_index = (mv->x2&4)>>2 | (mv->y2&4)>>1;
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
  SchroFrameComponent *comp;
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
copy_block (SchroFrame *dest, SchroMotion *motion, int x, int y, int reg)
{
  SchroFrameComponent *comp;
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
    x0 = (x>>comp->h_shift) + region->start_x;
    y0 = (y>>comp->v_shift) + region->start_y;
    
    if (region->end_x - region->start_x == 12) {
      int16_t *d1 = OFFSET(comp->data, comp->stride*y0 + 2*x0);
      int16_t *s1 = region->weights;

      oil_multiply_and_acc_12xn_s16_u8 (d1, comp->stride, s1, obmc->stride,
          motion->blocks[k] +
            motion->strides[k]*region->start_y + region->start_x,
          motion->strides[k],
          region->end_y - region->start_y);
    } else {
      int j;
      for(j=0;j<region->end_y - region->start_y;j++){
        oil_multiply_and_add_s16_u8 (
            OFFSET(comp->data, comp->stride*(y0+j) + 2*x0),
            OFFSET(comp->data, comp->stride*(y0+j) + 2*x0),
            OFFSET(region->weights, obmc->stride*j),
            OFFSET(motion->blocks[k], motion->strides[k]*(j+region->start_y) +
              region->start_x),
            region->end_x - region->start_x);
      }
    }
  }
}
    
static void
clear_rows (SchroFrame *frame, int y, int n)
{
  SchroFrameComponent *comp;
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
  SchroFrameComponent *comp;
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
    s[0] = (1<<s[1])>>1;

    ymax = MIN ((y + n)>>comp->v_shift, comp->height);
    ymin = MAX (y>>comp->v_shift, 0);
    for(j=ymin;j<ymax;j++){
      data = OFFSET(comp->data, j * comp->stride);
      oil_add_const_rshift_u16(data, data, s, comp->width);
    }
  }
}

void
schro_frame_copy_with_motion (SchroFrame *dest, SchroMotion *motion)
{
  int i, j;
  int x, y;
  SchroObmc *obmc_luma;
  SchroObmc *obmc_chroma;
  SchroMotionVector *motion_vectors = motion->motion_vectors;
  SchroParams *params = motion->params;

  obmc_luma = malloc(sizeof(*obmc_luma));
  schro_obmc_init (obmc_luma,
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);
  obmc_chroma = malloc(sizeof(*obmc_chroma));
  schro_obmc_init (obmc_chroma,
      params->xblen_luma>>motion->params->video_format->chroma_h_shift,
      params->yblen_luma>>motion->params->video_format->chroma_v_shift,
      params->xbsep_luma>>motion->params->video_format->chroma_h_shift,
      params->ybsep_luma>>motion->params->video_format->chroma_v_shift);
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

      if (mv->pred_mode == 0) {
        schro_motion_get_dc_block (motion, mv);
      } else {
        if (mv->pred_mode & 1) {
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[0];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 1);
          } else {
            schro_motion_get_block (motion, mv, x, y, 1);
          }
        }
        if (mv->pred_mode & 2) {
          if (mv->using_global) {
            SchroGlobalMotion *gm = &motion->params->global_motion[1];
            schro_motion_get_global_block_generic (motion, mv, x, y, gm, 2);
          } else {
            schro_motion_get_block (motion, mv, x, y, 2);
          }
        }
      }

      copy_block (dest, motion, x, y, region);
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
schro_motion_dc_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred)
{
  SchroMotionVectorDC *mvdc;
  int i;

  for(i=0;i<3;i++){
    int sum = 0;
    int n = 0;

    if (x>0) {
      mvdc = (SchroMotionVectorDC *)&motion_vectors[y*params->x_num_blocks + (x-1)];
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    if (y>0) {
      mvdc = (SchroMotionVectorDC *)&motion_vectors[(y-1)*params->x_num_blocks + x];
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    if (x>0 && y>0) {
      mvdc = (SchroMotionVectorDC *)&motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
      if (mvdc->pred_mode == 0) {
        sum += mvdc->dc[i];
        n++;
      }
    }
    switch(n) {
      case 0:
        pred[i] = 128;
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

void
schro_motion_field_get_global_prediction (SchroMotionField *mf,
    int x, int y, int *pred)
{
  if (x == 0 && y == 0) {
    *pred = 0;
    return;
  }
  if (y == 0) {
    *pred = mf->motion_vectors[x-1].using_global;
    return;
  }
  if (x == 0) {
    *pred = mf->motion_vectors[(y-1)*mf->x_num_blocks].using_global;
    return;
  }

  *pred = (mf->motion_vectors[(y-1)*mf->x_num_blocks + (x-1)].using_global +
      mf->motion_vectors[(y-1)*mf->x_num_blocks + x].using_global +
      mf->motion_vectors[y*mf->x_num_blocks + (x-1)].using_global) >= 2;
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
schro_motion_vector_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred_x, int *pred_y, int mode)
{
  SchroMotionVector *mv = &motion_vectors[y*params->x_num_blocks + x];
  int vx[3];
  int vy[3];
  int n = 0;

  SCHRO_ASSERT(mode == 1 || mode == 2);
  if (x>0) {
    mv = &motion_vectors[y*params->x_num_blocks + (x-1)];
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
    mv = &motion_vectors[(y-1)*params->x_num_blocks + x];
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
    mv = &motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
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
      {
        int shift = 3 - params->mv_precision;
        *pred_x = ((((vx[0] + vx[1])>>shift) + 1)/2) <<shift;
        *pred_y = ((((vy[0] + vy[1])>>shift) + 1)/2) <<shift;
      }
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
schro_motion_split_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y)
{
  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      return motion_vectors[x-4].split;
    }
  } else {
    if (x == 0) {
      return motion_vectors[(y-4)*params->x_num_blocks].split;
    } else {
      int value;
      value = (motion_vectors[(y-4)*params->x_num_blocks + (x-4)].split +
          motion_vectors[(y-4)*params->x_num_blocks + x].split +
          motion_vectors[y*params->x_num_blocks + (x-4)].split + 1) / 3;
      return value;
    }
  }
}

int
schro_motion_get_mode_prediction (SchroMotionField *mf, int x, int y)
{
  SchroMotionVector *mv = &mf->motion_vectors[y*mf->x_num_blocks + x];

  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      return mv[-1].pred_mode;
    }
  } else {
    if (x == 0) {
      return mv[-mf->x_num_blocks].pred_mode;
    } else {
      int ref0, ref1;
      ref0 = ((mv[-mf->x_num_blocks-1].pred_mode & 1) +
          (mv[-mf->x_num_blocks].pred_mode & 1) +
          (mv[-1].pred_mode & 1)) >= 2;
      ref1 = ((mv[-mf->x_num_blocks-1].pred_mode & 2) +
          (mv[-mf->x_num_blocks].pred_mode & 2) +
          (mv[-1].pred_mode & 2)) >= 4;
      return (ref1<<1) | (ref0);
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
  unsigned int precision_mask;
  SchroMotionVector *mv, *sbmv, *bmv;
  SchroParams *params = motion->params;

  precision_mask = 0x7 >> params->mv_precision;

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
        if (!mv->using_global && (mv->x1 & precision_mask || mv->y1 & precision_mask)) {
          SCHRO_ERROR("mv1 (%d,%d) has subpixel components not allowed by precision",
              x,y);
          return 0;
        }
        if (!mv->using_global && (mv->x2 & precision_mask || mv->y2 & precision_mask)) {
          SCHRO_ERROR("mv2 (%d,%d) has subpixel components not allowed by precision",
              x,y);
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
  schro_frame_upsample_horiz (df->frames[3], df->frames[1]);
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

#if 0
typedef struct _SchroBlock SchroBlock;
struct _SchroBlock {
  uint8_t *blocks[3];
  int strides[3];
  uint8_t *tmpdata;
};

void
get_block_simple (SchroBlock *dest, SchroUpsampledFrame *upframe, int x, int y)
{
  SchroFrame *srcframe;
  SchroFrameComponent *comp;
  int upsample_index;
  int sx, sy;

  upsample_index = (x&4)>>2 | (y&4)>>1;

  sx = x >> 3;
  sy = y >> 3;

  srcframe = upframe->frames[upsample_index];

  comp = &srcframe->components[0];
  dest->blocks[0] = OFFSET(comp->data, comp->stride * sy + sx);
  dest->strides[0] = comp->stride;

  sx >>= SCHRO_FRAME_FORMAT_H_SHIFT(srcframe->format);
  sy >>= SCHRO_FRAME_FORMAT_V_SHIFT(srcframe->format);

  comp = &srcframe->components[1];
  dest->blocks[1] = OFFSET(comp->data, comp->stride * sy + sx);
  dest->strides[1] = comp->stride;

  comp = &srcframe->components[2];
  dest->blocks[2] = OFFSET(comp->data, comp->stride * sy + sx);
  dest->strides[2] = comp->stride;
}
#endif

