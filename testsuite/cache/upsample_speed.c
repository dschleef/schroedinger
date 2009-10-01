
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
  SchroFrame *frame1;
  SchroUpsampledFrame *upframe;
  SchroMemoryDomain *mem;
  int i;

  mem = schro_memory_domain_new_local ();
  frame1 = schro_frame_new_and_alloc (mem, SCHRO_FRAME_FORMAT_U8_444, width, height);

  for(i=0;i<100;i++){
    upframe = schro_upsampled_frame_new (schro_frame_ref(frame1));
    schro_upsampled_frame_upsample (upframe);
    schro_upsampled_frame_free (upframe);
  }

  schro_frame_unref (frame1);

  schro_memory_domain_free (mem);
}


int
main (int argc, char *argv[])
{
  orc_init();

  upsample_speed (1, 1920, 1080);

  return 0;
}

