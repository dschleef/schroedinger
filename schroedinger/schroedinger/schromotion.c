
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void schro_decoder_predict (SchroDecoder *decoder);



void
schro_obmc_init (SchroObmc *obmc, int x_len, int y_len, int x_sep, int y_sep)
{
  int i;
  int j;
  int k;
  int x_ramp;
  int y_ramp;
  int weight;

  memset (obmc, 0, sizeof(*obmc));

  x_ramp = x_len - x_sep;
  y_ramp = y_len - y_sep;

  SCHRO_ASSERT(x_ramp != 0);
  SCHRO_ASSERT(y_ramp != 0);
  SCHRO_ASSERT((x_ramp&1) == 0);
  SCHRO_ASSERT((y_ramp&1) == 0);
  SCHRO_ASSERT(2*x_ramp <= x_len);
  SCHRO_ASSERT(2*y_ramp <= y_len);

  obmc->stride = sizeof(int16_t) * x_len;
  obmc->region_data = malloc(obmc->stride * y_len * 9);
  obmc->tmpdata = malloc(x_len * y_len);

  for(i=0;i<9;i++){
    obmc->regions[i].weights = OFFSET(obmc->region_data,
        obmc->stride * y_len * i);
    obmc->regions[i].end_x = x_len;
    obmc->regions[i].end_y = y_len;
  }
  obmc->shift = -1;
  for(weight = 4 * x_ramp * y_ramp; weight; weight>>=1) {
    obmc->shift++;
  }
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

  if (k==4 && obmc->x_sep == 12) {
    uint8_t tmp[12];
    int16_t *d1 = OFFSET(dest->data, dest->stride*y + 2*x);
    int16_t *s1 = region->weights;

    for(i=0;i<12;i++) tmp[i] = value;
    oil_multiply_and_acc_12xn_s16_u8 (d1, dest->stride, s1, obmc->stride,
        tmp, 0, 12);
  } else {
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        weight = SCHRO_GET(region->weights, obmc->stride*j + 2*i, int16_t);
        SCHRO_GET(dest->data, dest->stride*(y+j) + 2*(x+i), int16_t) +=
          weight * value;
      }
    }
  }
}

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

static void
clear_rows (SchroFrameComponent *comp, int y, int n)
{
  uint8_t zero = 0;
  int ymax;

  ymax = MIN (y + n, comp->height);
  for(;y<ymax;y++){
    oil_splat_u8_ns (OFFSET(comp->data, y * comp->stride), &zero,
          comp->width * sizeof(int16_t));
  }
}

static void
shift_rows (SchroFrameComponent *comp, int y, int n, int shift)
{
  int ymax;
  int16_t *data;
  int16_t s[2];

  s[1] = shift;
  s[0] = (1<<s[1])>>1;

  ymax = MIN (y + n, comp->height);
  if (y < 0) y = 0;
  for(;y<ymax;y++){
    data = OFFSET(comp->data, y * comp->stride);
    oil_add_const_rshift_s16(data, data, s, comp->width);
  }
}

void
schro_frame_copy_with_motion (SchroFrame *dest, SchroFrame *src1,
    SchroFrame *src2, SchroMotionVector *motion_vectors, SchroParams *params)
{
  int i, j;
  int k;
  int dx, dy;
  int x, y;
  SchroObmc obmc_luma;
  SchroObmc obmc_chroma;

  schro_obmc_init (&obmc_luma, 12, 12, 8, 8);
  schro_obmc_init (&obmc_chroma, 6, 6, 4, 4);

  for(k=0;k<3;k++){
    SchroFrameComponent *component = dest->components + k;
    SchroObmc *obmc;

    if (k == 0) {
      obmc = &obmc_luma;
    } else {
      obmc = &obmc_chroma;
    }

    clear_rows (component, 0, obmc->y_ramp/2);

  for(j=0;j<params->y_num_blocks;j++){
    y = j*obmc->y_sep;

    clear_rows (component, y + obmc->y_ramp/2, obmc->y_sep);

    for(i=0;i<params->x_num_blocks;i++){
      SchroMotionVector *mv = &motion_vectors[j*params->x_num_blocks + i];

      x = i*obmc->x_sep;

      if (mv->pred_mode == 0) {
        splat_block_general (component, x, y, mv->dc[k], obmc);
      } else {
        SchroFrame *ref;

        if (mv->pred_mode == 1) {
          ref = src1;
        } else {
          ref = src2;
        }

        dx = mv->x;
        dy = mv->y;

#if 0
        /* FIXME This is only roughly correct */
        SCHRO_ASSERT(x + dx >= 0);
        //SCHRO_ASSERT(x + dx < params->mc_luma_width - params->xbsep_luma);
        SCHRO_ASSERT(x + dx < params->mc_luma_width);
        SCHRO_ASSERT(y + dy >= 0);
        //SCHRO_ASSERT(y + dy < params->mc_luma_height - params->ybsep_luma);
        SCHRO_ASSERT(y + dy < params->mc_luma_height);
#endif

        if (k == 0) {
          copy_block_general (component, x, y, &ref->components[k],
              x+dx, y+dy, obmc);
        } else {
          copy_block_general (component, x, y, &ref->components[k],
              (x*2+dx)>>1, (y*2+dy)>>1, obmc);
        }
      }
    }

    shift_rows (component, y - obmc->y_ramp/2, obmc->y_sep, obmc->shift);
  }

  y = params->y_num_blocks*obmc->y_sep;
  shift_rows (component, y - obmc->y_ramp/2, obmc->y_ramp/2, obmc->shift);
}

  schro_obmc_cleanup (&obmc_luma);
  schro_obmc_cleanup (&obmc_chroma);
}

void
schro_motion_dc_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred)
{
  SchroMotionVector *mv = &motion_vectors[y*params->x_num_blocks + x];
  int i;

  for(i=0;i<3;i++){
    int sum = 0;
    int n = 0;

    if (x>0) {
      mv = &motion_vectors[y*params->x_num_blocks + (x-1)];
      if (mv->pred_mode == 0) {
        sum += mv->dc[i];
        n++;
      }
    }
    if (y>0) {
      mv = &motion_vectors[(y-1)*params->x_num_blocks + x];
      if (mv->pred_mode == 0) {
        sum += mv->dc[i];
        n++;
      }
    }
    if (x>0 && y>0) {
      mv = &motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
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
    SchroParams *params, int x, int y, int *pred_x, int *pred_y, int mode)
{
  SchroMotionVector *mv = &motion_vectors[y*params->x_num_blocks + x];
  int sum_x = 0;
  int sum_y = 0;
  int n = 0;

  if (x>0) {
    mv = &motion_vectors[y*params->x_num_blocks + (x-1)];
    if (mv->pred_mode == mode) {
      sum_x += mv->x;
      sum_y += mv->y;
      n++;
    }
  }
  if (y>0) {
    mv = &motion_vectors[(y-1)*params->x_num_blocks + x];
    if (mv->pred_mode == mode) {
      sum_x += mv->x;
      sum_y += mv->y;
      n++;
    }
  }
  if (x>0 && y>0) {
    mv = &motion_vectors[(y-1)*params->x_num_blocks + (x-1)];
    if (mv->pred_mode == mode) {
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

