
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

#include "common.h"

void upsample_line (uint8_t *dest, int dstr, uint8_t *src, int sstr, int n);
void ref_h_upsample (SchroFrame *dest, SchroFrame *src);
void ref_v_upsample (SchroFrame *dest, SchroFrame *src);

void test (int width, int height);

int failed = 0;

int dump_all = FALSE;

int
main (int argc, char *argv[])
{
  int width, height;

  schro_init();

  for(width=1;width<=20;width++){
    for(height=1;height<=20;height++){
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

void test (int width, int height)
{
  SchroFrame *frame;
  SchroFrame *frame_ref;
  SchroFrame *frame_1;
  SchroFrame *frame_test;
  SchroUpsampledFrame *upframe;
  char name[TEST_PATTERN_NAME_SIZE];
  int i;
  int ok;

  frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, width, height);
  frame_ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, width, height);
  frame_1 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420, width, height);

  printf("HORIZONTAL %dx%d\n", width, height);
  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (frame->components + 0, name, i);
    test_pattern_generate (frame->components + 1, name, i);
    test_pattern_generate (frame->components + 2, name, i);

    ref_h_upsample (frame_ref, frame);
    upframe = schro_upsampled_frame_new (schro_frame_ref(frame));
    schro_upsampled_frame_upsample (upframe);

    frame_test = upframe->frames[1];
    ok = frame_compare (frame_ref, frame_test);
    printf("  pattern %s: %s\n", name, ok ? "OK" : "broken");
    if (dump_all || !ok) {
      frame_data_dump_full (frame_test->components + 0,
          frame_ref->components + 0, frame->components + 0);
      failed = TRUE;
    }

    schro_upsampled_frame_free (upframe);
  }

  printf("VERTICAL %dx%d\n", width, height);
  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (frame->components + 0, name, i);
    test_pattern_generate (frame->components + 1, name, i);
    test_pattern_generate (frame->components + 2, name, i);

    ref_v_upsample (frame_ref, frame);
    upframe = schro_upsampled_frame_new (schro_frame_ref(frame));
    schro_upsampled_frame_upsample (upframe);

    frame_test = upframe->frames[2];
    ok = frame_compare (frame_ref, frame_test);
    printf("  pattern %s: %s\n", name, ok ? "OK" : "broken");
    if (dump_all || !ok) {
      frame_data_dump_full (frame_test->components + 0,
          frame_ref->components + 0, frame->components + 0);
      failed = TRUE;
    }

    schro_upsampled_frame_free (upframe);
  }

  printf("HV %dx%d\n", width, height);
  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (frame->components + 0, name, i);
    test_pattern_generate (frame->components + 1, name, i);
    test_pattern_generate (frame->components + 2, name, i);

    ref_v_upsample (frame_1, frame);
    ref_h_upsample (frame_ref, frame_1);
    upframe = schro_upsampled_frame_new (schro_frame_ref(frame));
    schro_upsampled_frame_upsample (upframe);

    frame_test = upframe->frames[3];
    ok = frame_compare (frame_ref, frame_test);
    printf("  pattern %s: %s\n", name, ok ? "OK" : "broken");
    if (dump_all || !ok) {
      frame_data_dump_full (frame_test->components + 0,
          frame_ref->components + 0, frame->components + 0);
      failed = TRUE;
    }

    schro_upsampled_frame_free (upframe);
  }

  schro_frame_unref (frame);
  schro_frame_unref (frame_ref);
}


void
upsample_line (uint8_t *dest, int dstr, uint8_t *src, int sstr, int n)
{
  int i;
  int j;
  int x;
  int weights[8] = { -1, 3, -7, 21, 21, -7, 3, -1 };

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<8;j++){
      x += weights[j] * SCHRO_GET(src, sstr * CLAMP(i+j-3,0,n-1), uint8_t);
    }
    x += 16;
    x >>= 5;
    SCHRO_GET(dest, dstr * i, uint8_t) = CLAMP(x, 0, 255);
  }
}

void
ref_h_upsample (SchroFrame *dest, SchroFrame *src)
{
  int j,k;
  uint8_t *d;
  uint8_t *s;

  for(k=0;k<3;k++){
    for(j=0;j<dest->components[k].height;j++){
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      upsample_line (d, 1, s, 1, dest->components[k].width);
    }
  }
}

void
ref_v_upsample (SchroFrame *dest, SchroFrame *src)
{
  int i,k;
  SchroFrameData *scomp;
  SchroFrameData *dcomp;

  for(k=0;k<3;k++){
    dcomp = dest->components + k;
    scomp = src->components + k;
    for(i=0;i<dest->components[k].width;i++){
      upsample_line (OFFSET(dcomp->data, i), dcomp->stride,
          OFFSET(scomp->data, i), scomp->stride,
          dcomp->height);
    }
  }
}

