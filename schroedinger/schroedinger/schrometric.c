
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
//#include <stdlib.h>
//#include <string.h>
//#include <stdio.h>


int
schro_metric_absdiff_u8 (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;

  if (height == 8 && width == 8) {
    uint32_t m;
    oil_sad8x8_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
  } else {
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
      }
    }
  }

  return metric;
}

#if 0
static void
schro_metric_haar_transform (int16_t *diff)
{
  int i,j;
  int a,b;

#if 0
  for(j=0;j<8;j+=2){
    for(i=0;i<8;i+=2){
      diff[8*j+i+1] -= (diff[8*j+i] + diff[8*j+i+1])/2;
      diff[8*j+i] += diff[8*j+i+1];
      diff[8*(j+1)+i+1] -= (diff[8*(j+1)+i] + diff[8*(j+1)+i+1])/2;
      diff[8*(j+1)+i] += diff[8*(j+1)+i+1];
      diff[8*(j+1)+i] -= (diff[8*j+i] + diff[8*(j+1)+i])/2;
      diff[8*j+i] += diff[8*(j+1)+i];
      diff[8*(j+1)+i+1] -= (diff[8*j+i+1] + diff[8*(j+1)+i+1])/2;
      diff[8*j+i+1] += diff[8*(j+1)+i+1];
    }
  }
  for(j=0;j<8;j+=4){
    for(i=0;i<8;i+=4){
      diff[8*j+i+2] -= (diff[8*j+i] + diff[8*j+i+2])/2;
      diff[8*j+i] += diff[8*j+i+2];
      diff[8*(j+2)+i+2] -= (diff[8*(j+2)+i] + diff[8*(j+2)+i+2])/2;
      diff[8*(j+2)+i] += diff[8*(j+2)+i+2];
      diff[8*(j+2)+i] -= (diff[8*j+i] + diff[8*(j+2)+i])/2;
      diff[8*j+i] += diff[8*(j+2)+i];
      diff[8*(j+2)+i+2] -= (diff[8*j+i+2] + diff[8*(j+2)+i+2])/2;
      diff[8*j+i+2] += diff[8*(j+2)+i+2];
    }
  }
  i = 0;
  j = 0;
  diff[8*j+i+4] -= (diff[8*j+i] + diff[8*j+i+4])/2;
  diff[8*j+i] += diff[8*j+i+4];
  diff[8*(j+4)+i+4] -= (diff[8*(j+4)+i] + diff[8*(j+4)+i+4])/2;
  diff[8*(j+4)+i] += diff[8*(j+4)+i+4];
  diff[8*(j+4)+i] -= (diff[8*j+i] + diff[8*(j+4)+i])/2;
  diff[8*j+i] += diff[8*(j+4)+i];
  diff[8*(j+4)+i+4] -= (diff[8*j+i+4] + diff[8*(j+4)+i+4])/2;
  diff[8*j+i+4] += diff[8*(j+4)+i+4];
#endif
#if 1
  for(j=0;j<8;j++){
    for(i=0;i<8;i+=2){
      a = (diff[8*j+i] + diff[8*j+i+1])/2;
      b = diff[8*j+i+1] - a;
      diff[8*j+i] = a;
      diff[8*j+i+1] = b;
    }
  }
  for(j=0;j<8;j++){
    for(i=0;i<8;i+=2){
      a = (diff[j+8*i] + diff[j+8*(i+1)])/2;
      b = diff[j+8*(i+1)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+1)] = b;
    }
  }
  for(j=0;j<8;j+=2){
    for(i=0;i<8;i+=4){
      a = (diff[8*j+i] + diff[8*j+i+2])/2;
      b = diff[8*j+i+2] - a;
      diff[8*j+i] = a;
      diff[8*j+i+2] = b;
    }
  }
  for(j=0;j<8;j+=2){
    for(i=0;i<8;i+=4){
      a = (diff[j+8*i] + diff[j+8*(i+2)])/2;
      b = diff[j+8*(i+2)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+2)] = b;
    }
  }
  for(j=0;j<8;j+=4){
    for(i=0;i<8;i+=8){
      a = (diff[8*j+i] + diff[8*j+i+4])/2;
      b = diff[8*j+i+4] - a;
      diff[8*j+i] = a;
      diff[8*j+i+4] = b;
    }
  }
  for(j=0;j<8;j+=4){
    for(i=0;i<8;i+=8){
      a = (diff[j+8*i] + diff[j+8*(i+4)])/2;
      b = diff[j+8*(i+4)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+4)] = b;
    }
  }
#endif
}
#endif

int
schro_metric_haar (uint8_t *src1, int stride1, uint8_t *src2, int stride2,
    int width, int height)
{
  int16_t diff[64];
  int i;
  int j;

  SCHRO_ASSERT(width==8);
  SCHRO_ASSERT(height==8);

  {
    unsigned int m;
    oil_sad8x8_u8(&m, src1, stride1, src2, stride2);
    return m;
  }

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      diff[j*8+i] = src1[j*stride1 + i] - src2[j*stride2 + i];
    }
  }

  //schro_metric_haar_transform (diff);

  return schro_metric_abssum_s16 (diff, sizeof(int16_t)*8, 8, 8);
}

int
schro_metric_haar_const (uint8_t *data, int stride, int dc_value,
    int width, int height)
{
  int16_t diff[64];
  int i;
  int j;

  SCHRO_ASSERT(width==8);
  SCHRO_ASSERT(height==8);

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      diff[j*8+i] = data[j*stride + i] - dc_value;
    }
  }

  //schro_metric_haar_transform (diff);

  return schro_metric_abssum_s16 (diff, sizeof(int16_t)*8, 8, 8);
}

int
schro_metric_abssum_s16 (int16_t *data, int stride, int width,
    int height)
{
  int sum = 0;
  int i,j;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += abs(data[j*(stride/sizeof(int16_t)) + i]);
    }
  }

  return sum;
}

int
schro_metric_sum_u8 (uint8_t *data, int stride, int width, int height)
{
  int sum = 0;
  int i,j;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += data[j*stride + i];
    }
  }

  return sum;
}

