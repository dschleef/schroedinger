
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>

void
create_pattern (SchroFrame *frame)
{
  int i,j;
  uint8_t *data;

  for(j=0;j<frame->height;j++){
    data = OFFSET(frame->components[0].data, j*frame->components[0].stride);
    for(i=0;i<frame->width;i++){
      data[i] = (((i>>4)&1) ^ ((j>>4)&1)) * 255;
    }
  }
  memset (frame->components[1].data, 0, frame->components[1].length);
  memset (frame->components[2].data, 0, frame->components[2].length);
  schro_frame_filter_lowpass2 (frame, 5.0);
}

void
distort (SchroFrame *frame)
{
  int i,j;
  uint8_t *data;

  for(j=0;j<frame->height;j++){
    data = OFFSET(frame->components[0].data, j*frame->components[0].stride);
    for(i=0;i<frame->width;i++){
      data[i] &= 0xfc;
    }
  }
}


void
test (void)
{
  SchroFrame *frame1;
  SchroFrame *frame2;

  frame1 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, 256, 256);
  frame2 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, 256, 256);

  create_pattern (frame1);
  create_pattern (frame2);

  distort (frame2);

  schro_frame_ssim (frame1, frame2);

  schro_frame_unref (frame1);
  schro_frame_unref (frame2);
}

int
main (int argc, char *argv[])
{
  schro_init();

  test();

  return 0;
}


