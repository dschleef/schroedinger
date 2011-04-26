
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <orc-test/orcprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int16_t tmp[2048+100];
int16_t *data;

int orc_profile_get_min (OrcProfile *prof)
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
  OrcProfile prof1;
  OrcProfile prof2;
  double ave_fwd, ave_rev;
  int i;
  SchroFrameData fd;

  fd.format = SCHRO_FRAME_FORMAT_S16_444;
  fd.data = data;
  fd.stride = width*2;
  fd.width = width;
  fd.height = height;

  orc_profile_init (&prof1);
  orc_profile_init (&prof2);

  for(i=0;i<10;i++){
    orc_profile_start (&prof1);
    schro_wavelet_transform_2d (&fd, filter, tmp);
    orc_profile_stop (&prof1);

    orc_profile_start (&prof2);
    schro_wavelet_inverse_transform_2d (&fd, &fd, filter, tmp);
    orc_profile_stop (&prof2);
  }

  //orc_profile_get_ave_std (&prof1, &ave_fwd, &std);
  //printf("fwd %g (%g)\n", ave, std);

  //orc_profile_get_ave_std (&prof2, &ave_rev, &std);
  //printf("rev %g (%g)\n", ave, std);

  ave_fwd = orc_profile_get_min (&prof1);
  ave_rev = orc_profile_get_min (&prof2);
  printf("%d %d %g %g %g %g\n", width, height, ave_fwd, ave_rev,
      ave_fwd/(width*height), ave_rev/(width*height));
}


int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  data = malloc(2048*512*2);

#if 0
  for(i=0;i<8;i++){
    printf("wavelet %d\n", i);
    wavelet_speed (i, 256, 256);
  }
#endif
  for(i=16;i<=2048;i+=16){
    wavelet_speed (3, i, 256);
  }

  free(data);

  return 0;
}

