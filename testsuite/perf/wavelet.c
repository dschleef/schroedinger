
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrooil.h>
#include <schroedinger/schrodomain.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>


int16_t tmp[2048+100];


double
gettime (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec*1e-9;
}

int oil_profile_get_min (OilProfile *prof)
{
  int i;
  int min;
  min = prof->hist_time[0];
  for(i=0;i<10;i++){
    if (prof->hist_count[i] > 0) {
      if (prof->hist_time[i] < min) {
        min = prof->hist_time[i];
      }
    }
  }
  return min;
}

void
wavelet_speed (SchroFrame *frame, int filter, int width, int height)
{
  int i;
  int j;
  double start, stop;
  SchroParams params;

  params.transform_depth = 4;
  params.iwt_luma_width = ROUND_UP_POW2(width, params.transform_depth);
  params.iwt_luma_height = ROUND_UP_POW2(height, params.transform_depth);
  params.iwt_chroma_width = params.iwt_luma_width/2;
  params.iwt_chroma_height = params.iwt_luma_height;
  params.wavelet_filter_index = filter;

  for(j=0;j<10;j++){
    start = gettime();
    for(i=0;i<10;i++){
      schro_frame_iwt_transform (frame, &params);
    }
    stop = gettime();

    printf("time %g, %g fps\n", (stop - start)/10.0, 10.0/(stop-start));
  }
}

#ifdef HAVE_CUDA
SchroMemoryDomain *schro_memory_domain_new_cuda (void);
void schro_gpuframe_iwt_transform (SchroFrame *frame, SchroParams *params);

void
wavelet_speed_cuda (SchroFrame *frame, int filter, int width, int height)
{
  int i;
  int j;
  double start, stop;
  SchroParams params;

  params.transform_depth = 4;
  params.iwt_luma_width = ROUND_UP_POW2(width, params.transform_depth);
  params.iwt_luma_height = ROUND_UP_POW2(height, params.transform_depth);
  params.iwt_chroma_width = params.iwt_luma_width/2;
  params.iwt_chroma_height = params.iwt_luma_height;
  params.wavelet_filter_index = filter;

  for(j=0;j<10;j++){
    start = gettime();
    for(i=0;i<10;i++){
      schro_gpuframe_iwt_transform (frame, &params);
    }
    stop = gettime();

    printf("time %g, %g fps\n", (stop - start)/10.0, 10.0/(stop-start));
  }
}
#endif


int
main (int argc, char *argv[])
{
  int i;
  SchroFrame *frame;
  int width;
  int height;

  width = 1920; height = 1088;
  //width = 1280; height = 720;

  oil_init();
  schro_init();

  printf ("size %dx%d\n", width, height);

  frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_422,
      width, height);

  for(i=0;i<7;i++){
    printf("wavelet %d\n", i);
    wavelet_speed (frame, i, width, height);
  }

#ifdef HAVE_CUDA
  {
    SchroMemoryDomain *cuda_domain = schro_memory_domain_new_cuda ();

    frame = schro_frame_new_and_alloc (cuda_domain, SCHRO_FRAME_FORMAT_S16_422,
        width, height);

    for(i=0;i<7;i++){
      printf("wavelet %d\n", i);
      wavelet_speed_cuda (frame, i, width, height);
    }
  }
#endif

  return 0;
}

