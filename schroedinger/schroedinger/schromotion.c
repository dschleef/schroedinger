
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void schro_decoder_predict (SchroDecoder *decoder);



#if 0
static void
copy_block_4x4 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<4;j++){
    *(uint32_t *)(dest + dstr*j) = *(uint32_t *)(src + sstr*j);
  }
}

static void
copy_block_8x8 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<8;j++){
    *(uint64_t *)(dest + dstr*j) = *(uint64_t *)(src + sstr*j);
  }
}
#endif

#if 0
static void
copy_block_6x6_4x4 (int16_t *dest, int dstr, uint8_t *src, int sstr)
{
  int i, j;
  const int weights[6] = { 1, 3, 4, 4, 3, 1 };

  for(j=0;j<6;j++){
    for(i=0;i<6;i++){
      dest[i] += weights[i]*weights[j]*src[i];
    }
    dest = (int16_t *)((uint8_t *)dest + dstr);
    src += sstr;
  }
}
#endif

#if 0
static void
copy_block_12x12_8x8 (int16_t *dest, int dstr, uint8_t *src, int sstr)
{
  int i, j;
  const int weights[12] = { 1, 3, 5, 7, 8, 8, 8, 8, 7, 5, 3, 1 };

  for(j=0;j<12;j++){
    for(i=0;i<12;i++){
      dest[i] += weights[i]*weights[j]*src[i];
    }
    dest = (int16_t *)((uint8_t *)dest + dstr);
    src += sstr;
  }
}
#endif

#if 0
static void
splat_block_6x6_4x4 (int16_t *dest, int dstr, int value)
{
  int i, j;
  const int weights[6] = { 1, 3, 4, 4, 3, 1 };

  for(j=0;j<6;j++){
    for(i=0;i<6;i++){
      dest[i] += weights[i]*weights[j]*value;
    }
    dest = (int16_t *)((uint8_t *)dest + dstr);
  }
}
#endif

#if 0
static void
splat_block_12x12_8x8 (int16_t *dest, int dstr, int value)
{
  int i, j;
  const int weights[12] = { 1, 3, 5, 7, 8, 8, 8, 8, 7, 5, 3, 1 };

  for(j=0;j<12;j++){
    for(i=0;i<12;i++){
      dest[i] += weights[i]*weights[j]*value;
    }
    dest = (int16_t *)((uint8_t *)dest + dstr);
  }
}
#endif

#if 0
static void
copy_block (uint8_t *dest, int dstr, uint8_t *src, int sstr, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = src[sstr*j+i];
    }
  }
}
#endif

#if 0
static void
splat_block (uint8_t *dest, int dstr, int value, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = value;
    }
  }
}
#endif

#define SCHRO_GET(ptr, offset, type) (*(type *)((uint8_t *)(ptr) + (offset)) )

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

  SCHRO_ASSERT(x_ramp != 0);
  SCHRO_ASSERT(y_ramp != 0);
  SCHRO_ASSERT((x_ramp&1) == 0);
  SCHRO_ASSERT((y_ramp&1) == 0);
  SCHRO_ASSERT(2*x_ramp <= x_len);
  SCHRO_ASSERT(2*y_ramp <= y_len);

  for(i=0;i<9;i++){
    obmc->regions[i].weights = malloc(sizeof(int16_t) * x_len * y_len);
    obmc->regions[i].end_x = x_len;
    obmc->regions[i].end_y = y_len;
  }
  obmc->stride = sizeof(int16_t) * x_len;
  obmc->max_weight = 4 * x_ramp * y_ramp;
  obmc->x_ramp = x_ramp;
  obmc->y_ramp = y_ramp;
  obmc->x_len = x_len;
  obmc->y_len = y_len;
  obmc->x_sep = x_sep;
  obmc->y_sep = y_sep;

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

}

void
schro_obmc_cleanup (SchroObmc *obmc)
{
  int i;
  for(i=0;i<9;i++){
    if (obmc->regions[i].weights) {
      free(obmc->regions[i].weights);
    }
  }
}

void
splat_block_general (SchroFrameComponent *dest, int x, int y, int value,
    SchroObmc *obmc)
{
  int i,j;
  int k;
  int weight;
  SchroObmcRegion *region;

  x -= obmc->x_ramp/2;
  y -= obmc->y_ramp/2;

  k = 0;
  if (x>0) k++;
  if (x + obmc->x_len >= dest->width) k++;
  if (y>0) k+=3;
  if (y + obmc->y_len >= dest->height) k+=3;

  region = obmc->regions + k;

  for(j=region->start_y;j<region->end_y;j++){
    for(i=region->start_x;i<region->end_x;i++){
      weight = SCHRO_GET(region->weights, obmc->stride*j + 2*i, int16_t);
      SCHRO_GET(dest->data, dest->stride*(y+j) + 2*(x+i), int16_t) +=
        weight * value;
    }
  }
}

void
notoil_multiply_and_add_s16 (int16_t *dest, int16_t *src1, uint8_t *src2,
    int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] += src1[i] * src2[i];
  }
}

#define OFFSET(ptr,offset) ((void *)((uint8_t *)ptr + offset))
#define CLAMP(x,a,b) ((x)<(a) ? (a) : ((x)>(b) ? (b) : (x)))

void
copy_block_general (SchroFrameComponent *dest, int x, int y,
    SchroFrameComponent *src, int sx, int sy, SchroObmc *obmc)
{
  int i,j;
  int k;
  int weight;
  int value;
  SchroObmcRegion *region;

  SCHRO_ASSERT(x>=0);
  SCHRO_ASSERT(y>=0);
  SCHRO_ASSERT(x + obmc->x_sep<=dest->width);
  SCHRO_ASSERT(y + obmc->y_sep<=dest->height);
//SCHRO_ERROR("xy %d %d sxy %d %d", x, y, sx, sy);

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

  if (sx + region->start_x <0 || sy + region->start_y <0 || 
      sx + region->end_x >= src->width ||
      sy + region->end_y >= src->height) {
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        int src_x = CLAMP(sx + i, 0, src->width);
        int src_y = CLAMP(sy + j, 0, src->height);
        weight = SCHRO_GET(region->weights, obmc->stride*j + 2*i, int16_t);
        value = SCHRO_GET(src->data, src->stride * src_y + src_x, uint8_t);
        SCHRO_GET(dest->data, dest->stride*(y+j) + 2*(x+i), int16_t) +=
          weight * value;
      }
    }
  } else {
    for(j=region->start_y;j<region->end_y;j++){
      notoil_multiply_and_add_s16 (
          OFFSET(dest->data, dest->stride*(y+j) + 2*(x + region->start_x)),
          OFFSET(region->weights, obmc->stride*j + 2*region->start_x),
          OFFSET(src->data, src->stride * (sy + j) + sx + region->start_x),
          region->end_x - region->start_x);
#if 0
      for(i=region->start_x;i<region->end_x;i++){
        weight = SCHRO_GET(region->weights, obmc->stride*j + 2*i, int16_t);
        value = SCHRO_GET(src->data, src->stride * (sy + j) + sx+i, uint8_t);
        SCHRO_GET(dest->data, dest->stride*(y+j) + 2*(x+i), int16_t) +=
          weight * value;
      }
#endif
    }
  }
}

static void
oil_divpow2_s16(int16_t *data, int *shift, int n)
{
  int i;
  int16_t round;

  if (*shift > 1) {
    round = 1<<(*shift-1);
  } else {
    round = 0;
  }

  for(i=0;i<n;i++){
    data[i] = (data[i] + round) >> (*shift);
  }
}

static int
ilog2(unsigned int x)
{
  int shift=0;
  
  x>>=1;
  while(x) {
    x>>=1;
    shift++;
  }

  return shift;
}

#define OFFSET(ptr,offset) ((void *)((uint8_t *)ptr + offset))

void
schro_frame_copy_with_motion (SchroFrame *dest, SchroFrame *src1,
    SchroFrame *src2, SchroMotionVector *motion_vectors, SchroParams *params)
{
  SchroFrame *frame = dest;
  SchroFrame *reference_frame = src1;
  int i, j;
  int dx, dy;
  int x, y;
  SchroObmc obmc_luma;
  SchroObmc obmc_chroma;
  uint8_t zero = 0;

  schro_obmc_init (&obmc_luma, 12, 12, 8, 8);
  schro_obmc_init (&obmc_chroma, 6, 6, 4, 4);

  oil_splat_u8_ns (dest->components[0].data, &zero, dest->components[0].length);
  oil_splat_u8_ns (dest->components[1].data, &zero, dest->components[1].length);
  oil_splat_u8_ns (dest->components[2].data, &zero, dest->components[2].length);
  for(j=0;j<params->y_num_blocks;j++){
    y = j*params->ybsep_luma;

    for(i=0;i<params->x_num_blocks;i++){
      SchroMotionVector *mv = &motion_vectors[j*params->x_num_blocks + i];

      x = i*params->xbsep_luma;

      if (mv->pred_mode == 0) {
        splat_block_general (&frame->components[0], x, y, mv->dc[0],
            &obmc_luma);
        splat_block_general (&frame->components[1], x/2, y/2, mv->dc[1],
            &obmc_chroma);
        splat_block_general (&frame->components[2], x/2, y/2, mv->dc[2],
            &obmc_chroma);
      } else {
        dx = mv->x;
        dy = mv->y;

        /* FIXME This is only roughly correct */
        SCHRO_ASSERT(x + dx >= 0);
        //SCHRO_ASSERT(x + dx < params->mc_luma_width - params->xbsep_luma);
        SCHRO_ASSERT(x + dx < params->mc_luma_width);
        SCHRO_ASSERT(y + dy >= 0);
        //SCHRO_ASSERT(y + dy < params->mc_luma_height - params->ybsep_luma);
        SCHRO_ASSERT(y + dy < params->mc_luma_height);

        copy_block_general (&frame->components[0], x, y,
            &reference_frame->components[0], x+dx, y+dy, &obmc_luma);
        copy_block_general (&frame->components[1], x/2, y/2,
            &reference_frame->components[1], (x+dx)/2, (y+dy)/2, &obmc_chroma);
        copy_block_general (&frame->components[2], x/2, y/2,
            &reference_frame->components[2], (x+dx)/2, (y+dy)/2, &obmc_chroma);
      }
    }
  }
  {
    int16_t *data;
    int shift;
    data = frame->components[0].data;
    shift = ilog2(obmc_luma.max_weight);
    for(j=0;j<frame->components[0].height;j++){
      oil_divpow2_s16(data, &shift, frame->components[0].width);
      data = OFFSET(data, frame->components[0].stride);
    }
    data = frame->components[1].data;
    shift = ilog2(obmc_chroma.max_weight);
    for(j=0;j<frame->components[1].height;j++){
      oil_divpow2_s16(data, &shift, frame->components[1].width);
      data = OFFSET(data, frame->components[1].stride);
    }
    data = frame->components[2].data;
    for(j=0;j<frame->components[2].height;j++){
      oil_divpow2_s16(data, &shift, frame->components[2].width);
      data = OFFSET(data, frame->components[2].stride);
    }
  }

  schro_obmc_cleanup (&obmc_luma);
  schro_obmc_cleanup (&obmc_chroma);
}

void
schro_motion_dc_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred)
{
  SchroMotionVector *mv = &motion_vectors[y*(4*params->x_num_mb) + x];
  int i;

  for(i=0;i<3;i++){
    int sum = 0;
    int n = 0;

    if (x>0) {
      mv = &motion_vectors[y*(4*params->x_num_mb) + (x-1)];
      if (mv->pred_mode == 0) {
        sum += mv->dc[i];
        n++;
      }
    }
    if (y>0) {
      mv = &motion_vectors[(y-1)*(4*params->x_num_mb) + x];
      if (mv->pred_mode == 0) {
        sum += mv->dc[i];
        n++;
      }
    }
    if (x>0 && y>0) {
      mv = &motion_vectors[(y-1)*(4*params->x_num_mb) + (x-1)];
      if (mv->pred_mode == 0) {
        sum += mv->dc[i];
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
schro_motion_vector_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred_x, int *pred_y)
{
  SchroMotionVector *mv = &motion_vectors[y*(4*params->x_num_mb) + x];
  int sum_x = 0;
  int sum_y = 0;
  int n = 0;

  if (x>0) {
    mv = &motion_vectors[y*(4*params->x_num_mb) + (x-1)];
    if (mv->pred_mode == 1) {
      sum_x += mv->x;
      sum_y += mv->y;
      n++;
    }
  }
  if (y>0) {
    mv = &motion_vectors[(y-1)*(4*params->x_num_mb) + x];
    if (mv->pred_mode == 1) {
      sum_x += mv->x;
      sum_y += mv->y;
      n++;
    }
  }
  if (x>0 && y>0) {
    mv = &motion_vectors[(y-1)*(4*params->x_num_mb) + (x-1)];
    if (mv->pred_mode == 1) {
      sum_x += mv->x;
      sum_y += mv->y;
      n++;
    }
  }
  switch(n) {
    case 0:
      *pred_x = 0;
      *pred_y = 0;
      break;
    case 1:
      *pred_x = sum_x;
      *pred_y = sum_y;
      break;
    case 2:
      *pred_x = (sum_x + 1)/2;
      *pred_y = (sum_y + 1)/2;
      break;
    case 3:
      *pred_x = (sum_x + 1)/3;
      *pred_y = (sum_y + 1)/3;
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
      return motion_vectors[(y-4)*(4*params->x_num_mb)].split;
    } else {
      int value;
      value = (motion_vectors[(y-4)*(4*params->x_num_mb) + (x-4)].split +
          motion_vectors[(y-4)*(4*params->x_num_mb) + x].split +
          motion_vectors[y*(4*params->x_num_mb) + (x-4)].split + 1) / 3;
      return value;
    }
  }
}

