
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <orc/orc.h>
#include <orc-test/orcprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int16_t tmp[2048+100];

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
upsample_speed (int filter, int width, int height)
{
  OrcProfile prof1;
  double ave;
  int i;
  SchroFrame *frame1;
  SchroFrame *frame2;
  SchroUpsampledFrame *upframe;
  SchroMemoryDomain *mem;

  mem = schro_memory_domain_new_local ();
  frame1 = schro_frame_new_and_alloc (mem, SCHRO_FRAME_FORMAT_U8_444, width, height);
  frame2 = schro_frame_new_and_alloc (mem, SCHRO_FRAME_FORMAT_U8_444, width, height);

  orc_profile_init (&prof1);

  upframe = schro_upsampled_frame_new (schro_frame_ref(frame1));
  schro_upsampled_frame_upsample (upframe);
  schro_upsampled_frame_free (upframe);
  for(i=0;i<10;i++) {
    upframe = schro_upsampled_frame_new (schro_frame_ref(frame1));
    orc_profile_start (&prof1);
    schro_upsampled_frame_upsample (upframe);
    orc_profile_stop (&prof1);
    schro_upsampled_frame_free (upframe);
  }

  ave = orc_profile_get_min (&prof1);
  printf("%d %d %g %g\n", width, height, ave, ave/(width*height));

  schro_frame_unref (frame1);
  schro_frame_unref (frame2);

  schro_memory_domain_free (mem);
}


int
main (int argc, char *argv[])
{
  int i;

  orc_init();

  for(i=16;i<=2048;i+=16){
    upsample_speed (1, i, (i*9)/16);
  }

  return 0;
}

