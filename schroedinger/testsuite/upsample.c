
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schropredict.h>
#include <schroedinger/schrodebug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <liboil/liboilrandom.h>


int
main (int argc, char *argv[])
{
  SchroFrame *frame;
  SchroFrame *frame2;

  schro_init();

  frame = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8, 320,240, 160, 120);
  frame2 = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8, 320,240, 160, 120);

  schro_frame_h_upsample (frame2, frame);
  schro_frame_v_upsample (frame2, frame);

  return 0;
}

