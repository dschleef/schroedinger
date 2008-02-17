
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrooil.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int16_t tmp[2048+100];
int16_t *data;

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
wavelet_speed (int filter, int width, int height)
{
  OilProfile prof1;
  OilProfile prof2;
  double ave_horiz, ave_vert;
  int i;
  SchroFrame *frame1;
  SchroFrame *frame2;

  frame1 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_444, width, height);
  frame2 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_444, width, height);

  oil_profile_init (&prof1);
  oil_profile_init (&prof2);

  schro_frame_upsample_horiz (frame2, frame1);
  schro_frame_upsample_vert (frame2, frame1);
  for(i=0;i<10;i++){
    oil_profile_start (&prof1);
    schro_frame_upsample_horiz (frame2, frame1);
    oil_profile_stop (&prof1);

    oil_profile_start (&prof2);
    schro_frame_upsample_vert (frame2, frame1);
    oil_profile_stop (&prof2);
  }

  ave_horiz = oil_profile_get_min (&prof1);
  ave_vert = oil_profile_get_min (&prof2);
  printf("%d %d %g %g %g %g\n", width, height, ave_horiz, ave_vert,
      ave_horiz/(width*height), ave_vert/(width*height));

  schro_frame_unref (frame1);
  schro_frame_unref (frame2);
}


int
main (int argc, char *argv[])
{
  int i;

  oil_init();

  data = malloc(2048*512*2);

#if 0
  for(i=0;i<8;i++){
    printf("wavelet %d\n", i);
    wavelet_speed (i, 256, 256);
  }
#endif
  for(i=16;i<=2048;i+=16){
    wavelet_speed (1, i, 256);
  }

  free(data);

  return 0;
}

