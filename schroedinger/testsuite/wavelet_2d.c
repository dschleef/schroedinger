
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int16_t *frame_data;
int16_t *tmp;

int
basic_test(int filter_index, int width, int height, int transform_depth)
{
  int level;

  frame_data = malloc(width*height*2);
  tmp = malloc(1024*2);
  memset (frame_data, 0, width*height*2);

  for(level=0;level<transform_depth;level++) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = width << level;

    schro_wavelet_transform_2d (filter_index,
        frame_data, stride, w, h, tmp);
  }

  free(frame_data);
  free(tmp);

  return 0;
}

int
constant_test(int filter_index, int width, int height)
{
  int i;
  int j;

  frame_data = malloc(width*height*2);
  tmp = malloc(1024*2);

  for(i=0;i<height*width;i++){
    frame_data[i] = 10;
  }

  schro_wavelet_transform_2d (filter_index,
        frame_data, width*2, width, height, tmp);

#if 1
  for(i=0;i<height;i++){
    for(j=0;j<width;j++){
      printf("%d ", frame_data[i*width + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  schro_wavelet_inverse_transform_2d (filter_index,
        frame_data, width*2, width, height, tmp);

#if 1
  for(i=0;i<height;i++){
    for(j=0;j<width;j++){
      printf("%d ", frame_data[i*width + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  free(frame_data);
  free(tmp);

  return 0;
}

int
vramp_test(int filter_index, int width, int height)
{
  int i;
  int j;

  frame_data = malloc(width*height*2);
  tmp = malloc(1024*2);

  for (j=0;j<width;j++) {
    for(i=0;i<width;i++){
      //frame_data[j*width + i] = j;
      frame_data[j*width + i] = i;
    }
  }

  schro_wavelet_transform_2d (filter_index,
        frame_data, width*2, width, height, tmp);
#if 1
  for(i=0;i<height;i++){
    for(j=0;j<width;j++){
      printf("%d ", frame_data[i*width + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif
  schro_wavelet_transform_2d (filter_index,
        frame_data, width*4, width/2, height/2, tmp);

#if 1
  for(i=0;i<height;i++){
    for(j=0;j<width;j++){
      printf("%d ", frame_data[i*width + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  schro_wavelet_inverse_transform_2d (filter_index,
        frame_data, width*4, width/2, height/2, tmp);
  schro_wavelet_inverse_transform_2d (filter_index,
        frame_data, width*2, width, height, tmp);

#if 1
  for(i=0;i<height;i++){
    for(j=0;j<width;j++){
      printf("%d ", frame_data[i*width + j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  free(frame_data);
  free(tmp);

  return 0;
}


int
main (int argc, char *argv[])
{
  schro_init();

#if 0
  basic_test(SCHRO_WAVELET_5_3, 640,512,6);

  constant_test(SCHRO_WAVELET_5_3, 32,20);
  constant_test(SCHRO_WAVELET_5_3, 10,2);

  vramp_test(SCHRO_WAVELET_5_3, 16, 16);
#endif

  basic_test(SCHRO_WAVELET_HAAR, 640,512,6);

  constant_test(SCHRO_WAVELET_HAAR, 32,20);
  constant_test(SCHRO_WAVELET_HAAR, 10,2);

  vramp_test(SCHRO_WAVELET_HAAR, 16, 16);

#if 0
  constant_test(SCHRO_WAVELET_DAUB97, 10, 10);
#endif

  return 0;
}


