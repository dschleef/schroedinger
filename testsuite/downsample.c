
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <orc-test/orcrandom.h>

#include "common.h"

void ref_frame_downsample (SchroFrame *dest, SchroFrame *src);

void frame_create_test_pattern(SchroFrame *frame, int type);

int failed = FALSE;

void
test (int width, int height)
{
  SchroFrame *frame;
  SchroFrame *frame_ref;
  SchroFrame *frame_test;
  char name[TEST_PATTERN_NAME_SIZE];
  int i;

  printf("size %dx%d\n", width, height);

  frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, width, height);
  frame_ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420,
      ROUND_UP_SHIFT(width, 1), ROUND_UP_SHIFT(height, 1));
  frame_test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420,
      ROUND_UP_SHIFT(width, 1), ROUND_UP_SHIFT(height, 1));

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (frame->components + 0, name, i);
    test_pattern_generate (frame->components + 1, name, i);
    test_pattern_generate (frame->components + 2, name, i);

    ref_frame_downsample (frame_ref, frame);
    schro_frame_downsample (frame_test, frame);

    if (frame_compare (frame_ref, frame_test)) {
      printf("  pattern %s: OK\n", name);
    } else {
      printf("  pattern %s: broken\n", name);
      frame_data_dump_full (frame_test->components + 0,
          frame_ref->components + 0, frame->components + 0);
      failed = TRUE;
    }
  }

  schro_frame_unref (frame_ref);
  schro_frame_unref (frame_test);
  schro_frame_unref (frame);
}

int
main (int argc, char *argv[])
{
  int width;
  int height;

  schro_init();

  for(width=10;width<40;width++){
    for(height=10;height<40;height++){
      test (width, height);
    }
  }

  if (failed) {
    printf("FAILED\n");
  } else {
    printf("SUCCESS\n");
  }

  return failed;
}


int
component_get (SchroFrameData *src, int i, int j)
{
  uint8_t *data;

  i = CLAMP(i,0,src->width-1);
  j = CLAMP(j,0,src->height-1);
  data = OFFSET(src->data, j*src->stride);

  return data[i];
}

void
ref_frame_component_downsample (SchroFrameData *dest,
    SchroFrameData *src)
{
  static const int taps[4] = { 6, 26, 26, 6 };
  int i,j;
  int k,l;
  uint8_t *ddata;

  for(j=0;j<dest->height;j++){
    ddata = OFFSET(dest->data, dest->stride * j);
    for(i=0;i<dest->width;i++){
      int x = 0;
      for(l=0;l<4;l++){
        int y = 0;
        for(k=0;k<4;k++){
          y += component_get (src, (i*2-1) + l, (j*2-1) + k) * taps[k];
        }
        x += CLAMP((y + 32) >> 6,0,255) * taps[l];
      }
      ddata[i] = CLAMP((x + 32) >> 6,0,255);
    }
  }
}

void
ref_frame_downsample (SchroFrame *dest, SchroFrame *src)
{
  ref_frame_component_downsample (dest->components+0, src->components+0);
  ref_frame_component_downsample (dest->components+1, src->components+1);
  ref_frame_component_downsample (dest->components+2, src->components+2);
}

void
frame_create_test_pattern(SchroFrame *frame, int type)
{
  int j,k;
  uint8_t *data;

  for(k=0;k<3;k++){
    for(j=0;j<frame->components[k].height;j++){
      data = OFFSET(frame->components[k].data,j*frame->components[k].stride);
      orc_random_bits (&context, data, frame->components[k].width);
    }
  }
}


