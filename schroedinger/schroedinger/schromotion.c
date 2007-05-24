
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
//#include <stdlib.h>
#include <string.h>
//#include <stdio.h>


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
  if (2*y_ramp > x_len) {
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

#if 0
void
block_get (uint8_t **dest, int &stride, SchroFrameComponent *src, int x, int y,
    int width, int height)
{
  int i,j;
  int sx, sy;

  if (x >= 0 && y >=0 &&
      x + width < src->width &&
      y + height < src->height) {
    *dest = src->data + y*src->stride + x;
    *stride = src->stride;
    return;
  }

  for(j=0;j<height;j++){
    sy = CLAMP(j + y, 0, height);
    for(i=0;i<width;i++){
      sx = CLAMP(i + x, 0, width);
      (*dest)[j*(*stride) + i] = src->data[sy*src->stride + sx];
    }
  }
}
#endif

#if 0
void
block_get (uint8_t **dest, int *stride, SchroFrameComponent *src,
    int x, int y, int width, int height)
{
  int i,j;
  int fx,fy;

  fx = x&0x7;
  fy = y&0x7;
  x >>= 3;
  y >>= 3;

  if (sx == 0 && sy == 0) {
    int sx,sy;

    if (x >= 0 && y >=0 &&
        x + width < src->width &&
        y + height < src->height) {
      *dest = src->data + y*src->stride + x;
      *stride = src->stride;
      return;
    }

    for(j=0;j<height;j++){
      sy = CLAMP(y + j, 0, height - 1);
      for(i=0;i<width;i++){
        sx = CLAMP(x + i, 0, width - 1);
        (*dest)[j*(*stride) + i] = src->data[sy*src->stride + sx];
      }
    }
  } else {
    SCHRO_ASSERT(0);
  }
}
#endif

void
global_block_get (uint8_t *dest, int stride, SchroFrameComponent *src,
    int x, int y, int width, int height, SchroGlobalMotion *gm)
{
  int i,j;
  int sx, sy;
  int persp;
  uint8_t *sdata = src->data;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      persp = (1<<gm->c_exp) - gm->c0 * (x + i) - gm->c1 * (y + j);
      sx = (persp * (gm->a00 * (x + i) + gm->a01 * (y + j) +
          (1<<gm->a_exp) * gm->b0)) >> (gm->c_exp + gm->a_exp);
      sy = (persp * (gm->a10 * (x + i) + gm->a11 * (y + j) +
          (1<<gm->a_exp) * gm->b1)) >> (gm->c_exp + gm->a_exp);
      sx = CLAMP(sx, 0, src->width - 1);
      sy = CLAMP(sy, 0, src->height - 1);
      dest[j*stride + i] = sdata[sy * src->stride + sx];
    }
  }
}

void
get_dc_block (SchroMotion *motion, SchroMotionVector *mv)
{
  int offset;

  offset = 0;
  memset (motion->tmpdata + offset, mv->u.dc[0], motion->obmc_luma->x_len);
  motion->blocks[0] = motion->tmpdata + offset;
  motion->strides[0] = 0;
  offset += motion->obmc_luma->x_len;

  memset (motion->tmpdata + offset, mv->u.dc[1], motion->obmc_chroma->x_len);
  motion->blocks[1] = motion->tmpdata + offset;
  motion->strides[1] = 0;
  offset += motion->obmc_chroma->x_len;

  memset (motion->tmpdata + offset, mv->u.dc[2], motion->obmc_chroma->x_len);
  motion->blocks[2] = motion->tmpdata + offset;
  motion->strides[2] = 0;
}

#if 0
void
splat_block_general (SchroFrame *dest, SchroMotion *motion,
    int x, int y, SchroMotionVector *mv)
{
  int i,j;
  int k;
  SchroObmcRegion *region;
  uint8_t tmp[12];

  x -= obmc->x_ramp/2;
  y -= obmc->y_ramp/2;

  k = 0;
  if (x>0) k++;
  if (x + obmc->x_len >= dest->width) k++;
  if (y>0) k+=3;
  if (y + obmc->y_len >= dest->height) k+=3;

  region = obmc->regions + k;
  x += region->start_x;
  y += region->start_y;

  for(i=0;i<12;i++) tmp[i] = value;

  if (region->end_x - region->start_x == 12) {
    int16_t *d1 = OFFSET(dest->data, dest->stride*y + 2*x);
    int16_t *s1 = region->weights;

    oil_multiply_and_acc_12xn_s16_u8 (d1, dest->stride, s1, obmc->stride,
        tmp, 0, region->end_y - region->start_y);
  } else {
    for(j=0;j<region->end_y - region->start_y;j++){
      oil_multiply_and_add_s16_u8 (
          OFFSET(dest->data, dest->stride*(y+j) + 2*x),
          OFFSET(dest->data, dest->stride*(y+j) + 2*x),
          OFFSET(region->weights, obmc->stride*j),
          tmp,
          region->end_x - region->start_x);
    }
  }

}
#endif

void
get_global_block (SchroMotion *motion, SchroMotionVector *mv,
    int x, int y, SchroGlobalMotion *gm, int which)
{
  int offset;

  /* FIXME */

  offset = 0;
  memset (motion->tmpdata + offset, 128, motion->obmc_luma->x_len);
  motion->blocks[0] = motion->tmpdata + offset;
  motion->strides[0] = 0;
  offset += motion->obmc_luma->x_len;

  memset (motion->tmpdata + offset, 128, motion->obmc_chroma->x_len);
  motion->blocks[1] = motion->tmpdata + offset;
  motion->strides[1] = 0;
  offset += motion->obmc_chroma->x_len;

  memset (motion->tmpdata + offset, 128, motion->obmc_chroma->x_len);
  motion->blocks[2] = motion->tmpdata + offset;
  motion->strides[2] = 0;
}

void
get_block_simple (SchroMotion *motion, int x, int y, int which)
{
  SchroFrame *srcframe;
  SchroFrameComponent *comp;
  int w, h;
  int upsample_index;

  upsample_index = (x&4)>>2 | (y&4)>>1;
  w = 12;
  h = 12;

  if (which == 2) {
    srcframe = motion->src2[upsample_index];
  } else {
    srcframe = motion->src1[upsample_index];
  }

  comp = &srcframe->components[0];
  motion->blocks[0] = OFFSET(comp->data, comp->stride * y + x);
  motion->strides[0] = comp->stride;

  x >>= motion->params->video_format->chroma_h_shift;
  y >>= motion->params->video_format->chroma_v_shift;
  comp = &srcframe->components[1];
  motion->blocks[1] = OFFSET(comp->data, comp->stride * y + x);
  motion->strides[1] = comp->stride;

  comp = &srcframe->components[2];
  motion->blocks[2] = OFFSET(comp->data, comp->stride * y + x);
  motion->strides[2] = comp->stride;

}
void
get_block (SchroMotion *motion, SchroMotionVector *mv, int x, int y, int which)
{
  uint8_t *data;
  int stride;
  int i,j;
  SchroFrame *srcframe;
  SchroFrameComponent *comp;
  int sx, sy;
  int w, h;
  int upsample_index;

  sx = x + (mv->u.xy.x>>3);
  sy = y + (mv->u.xy.y>>3);
  upsample_index = (mv->u.xy.x&4)>>2 | (mv->u.xy.y&4)>>1;
  w = 12;
  h = 12;

  /* FIXME */
  SCHRO_ASSERT(upsample_index == 0);

  if (which == 2) {
    srcframe = motion->src2[upsample_index];
  } else {
    srcframe = motion->src1[upsample_index];
  }
  SCHRO_ASSERT(srcframe);
#if 0
  if (sx & 3 || sy & 3) {
    /* FIXME */
  } else {
    if (sx < 0 || sy < 0 || sx > motion->sx_max || sy > motion->sy_max) {
      data = motion->obmc_luma->tmpdata;
      stride = motion->obmc_luma->x_len;
      for(j=0;j<region->end_y - region->start_y;j++){
        for(i=0;i<region->end_x - region->start_x;i++){
          int src_x = CLAMP(sx + i, 0, srcframe->width - 1);
          int src_y = CLAMP(sy + j, 0, srcframe->height - 1);
          data[j*stride + i] =
            SCHRO_GET(src->data, src->stride * src_y + src_x, uint8_t);
        }
      }
    } else {
      data = OFFSET(src->data, src->stride * sy + sx);
      stride = src->stride;
    }
  }
#endif

  /* FIXME move and fix */
  motion->sx_max = srcframe->width - 12;
  motion->sy_max = srcframe->height - 12;

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
    motion->blocks[1] = motion->tmpdata + 64*64;
    motion->strides[1] = 64;
    data = motion->blocks[1];
    comp = &srcframe->components[1];
    stride = motion->strides[0];
    for(j=0;j<h/2;j++){
      for(i=0;i<w/2;i++){
        int src_x = CLAMP(sx/2 + i, 0, comp->width - 1);
        int src_y = CLAMP(sy/2 + j, 0, comp->height - 1);
        data[j*stride + i] =
          SCHRO_GET(comp->data, comp->stride * src_y + src_x, uint8_t);
      }
    }
    motion->blocks[2] = motion->tmpdata + 64*64*2;
    motion->strides[2] = 64;
    data = motion->blocks[2];
    comp = &srcframe->components[2];
    stride = motion->strides[0];
    for(j=0;j<h/2;j++){
      for(i=0;i<w/2;i++){
        int src_x = CLAMP(sx/2 + i, 0, comp->width - 1);
        int src_y = CLAMP(sy/2 + j, 0, comp->height - 1);
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

void
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
    
#if 0
void
copy_block_general (SchroFrameComponent *dest, int x, int y,
    SchroFrameComponent *src, int sx, int sy, SchroObmc *obmc)
{
  int i,j;
  int k;
  SchroObmcRegion *region;
  uint8_t *data;
  int stride;

  SCHRO_ASSERT(x>=0);
  SCHRO_ASSERT(y>=0);
  SCHRO_ASSERT(x + obmc->x_sep<=dest->width);
  SCHRO_ASSERT(y + obmc->y_sep<=dest->height);

  x -= obmc->x_ramp/2;
  y -= obmc->y_ramp/2;
  sx -= obmc->x_ramp/2;
  sy -= obmc->y_ramp/2;

  k = 0;
  if (x>0) k++;
  if (x + obmc->x_len >= dest->width) k++;
  if (y>0) k+=3;
  if (y + obmc->y_len >= dest->height) k+=3;

  region = obmc->regions + k;

  x += region->start_x;
  y += region->start_y;
  sx += region->start_x;
  sy += region->start_y;

  if (sx < 0 || sy < 0 || 
      sx + (region->end_x - region->start_x) >= src->width ||
      sy + (region->end_y - region->start_y) >= src->height) {
    data = obmc->tmpdata;
    stride = obmc->x_len;
    for(j=0;j<region->end_y - region->start_y;j++){
      for(i=0;i<region->end_x - region->start_x;i++){
        int src_x = CLAMP(sx + i, 0, src->width - 1);
        int src_y = CLAMP(sy + j, 0, src->height - 1);
        data[j*stride + i] =
          SCHRO_GET(src->data, src->stride * src_y + src_x, uint8_t);
      }
    }
  } else {
    data = OFFSET(src->data, src->stride * sy + sx);
    stride = src->stride;
  }

  if (region->end_x - region->start_x == 12) {
    int16_t *d1 = OFFSET(dest->data, dest->stride*y + 2*x);
    int16_t *s1 = region->weights;

    oil_multiply_and_acc_12xn_s16_u8 (d1, dest->stride, s1, obmc->stride,
        data, stride, region->end_y - region->start_y);
  } else {
    for(j=0;j<region->end_y - region->start_y;j++){
      oil_multiply_and_add_s16_u8 (
          OFFSET(dest->data, dest->stride*(y+j) + 2*x),
          OFFSET(dest->data, dest->stride*(y+j) + 2*x),
          OFFSET(region->weights, obmc->stride*j),
          data + stride * j,
          region->end_x - region->start_x);
    }
  }


}
#endif

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
  int16_t *data;
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
      oil_add_const_rshift_s16(data, data, s, comp->width);
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
        get_dc_block (motion, mv);
      } else {
        if (mv->pred_mode & 1) {
          if (mv->using_global) {
            SchroGlobalMotion *gm = NULL;
            get_global_block (motion, mv, x, y, gm, 1);
          } else {
            get_block (motion, mv, x, y, 1);
          }
        }
        if (mv->pred_mode & 2) {
          if (mv->using_global) {
            SchroGlobalMotion *gm = NULL;
            get_global_block (motion, mv, x, y, gm, 2);
          } else {
            get_block (motion, mv, x, y, 2);
          }
        }
      }

      copy_block (dest, motion, x, y, region);
    }

    shift_rows (dest, y - obmc_luma->y_ramp/2, obmc_luma->y_sep,
        obmc_luma->shift, obmc_chroma->shift);
  }

  y = params->y_num_blocks*obmc_luma->y_sep;
  shift_rows (dest, y - obmc_luma->y_ramp/2, obmc_luma->y_ramp/2,
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
  SchroMotionVector *mv;
  int i;

  for(i=0;i<3;i++){
    int sum = 0;
    int n = 0;

    if (x>0) {
      mv = &motion_vectors[y*params->x_num_blocks + (x-1)];
      if (mv->pred_mode == 0) {
        sum += mv->u.dc[i];
        n++;
      }
    }
    if (y>0) {
      mv = &motion_vectors[(y-1)*params->x_num_blocks + x];
      if (mv->pred_mode == 0) {
        sum += mv->u.dc[i];
        n++;
      }
    }
    if (x>0 && y>0) {
      mv = &motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
      if (mv->pred_mode == 0) {
        sum += mv->u.dc[i];
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

  if (x>0) {
    mv = &motion_vectors[y*params->x_num_blocks + (x-1)];
    if (mv->using_global == FALSE && mv->pred_mode == mode) {
      vx[n] = mv->u.xy.x;
      vy[n] = mv->u.xy.y;
      n++;
    }
  }
  if (y>0) {
    mv = &motion_vectors[(y-1)*params->x_num_blocks + x];
    if (mv->using_global == FALSE && mv->pred_mode == mode) {
      vx[n] = mv->u.xy.x;
      vy[n] = mv->u.xy.y;
      n++;
    }
  }
  if (x>0 && y>0) {
    mv = &motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
    if (mv->using_global == FALSE && mv->pred_mode == mode) {
      vx[n] = mv->u.xy.x;
      vy[n] = mv->u.xy.y;
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

      if (mv->pred_mode) {
        /* hard to screw this one up */
      } else {
        if (mv->pred_mode & 2 && motion->src2[0] == NULL) {
          SCHRO_ERROR("mv(%d,%d) uses non-existent src2", x, y);
          return 0;
        }
#if 0
        if (mv->u.xy.x & 0x7 || mv->u.xy.y & 0x7) {
          SCHRO_ERROR("mv(%d,%d) has subpixel components (not implemented)",
              x, y);
          return 0;
        }
#endif
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

