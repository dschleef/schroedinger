
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>
#include <schroedinger/schrocuda.h>
#include <schroedinger/schrogpuframe.h>
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

SchroMemoryDomain *cpu_domain;
SchroMemoryDomain *cuda_domain;

int
main (int argc, char *argv[])
{
  //int width, height;

  schro_init();

  cpu_domain = schro_memory_domain_new_local ();
  cuda_domain = schro_memory_domain_new_cuda ();

#if 0
  for(width=1;width<=20;width++){
    for(height=1;height<=20;height++){
      test (width, height);
    }
  }
#endif
  test (640, 480);

  if (failed) {
    printf("FAILED\n");
  } else {
    printf("SUCCESS\n");
  }

  return failed;
}

void test (int width, int height)
{
  SchroFrame *frame_u8;
  SchroFrame *frame;
  SchroFrame *frame_ref;
  SchroFrame *frame_test;
  SchroFrame *frame_cuda;
  SchroParams params;
  char name[TEST_PATTERN_NAME_SIZE];
  int i;
  int ok;

  params.iwt_luma_width = width;
  params.iwt_luma_height = height;
  params.iwt_chroma_width = width/2;
  params.iwt_chroma_height = height/2;
  params.transform_depth = 1;

  frame_u8 = schro_frame_new_and_alloc (cpu_domain,
      SCHRO_FRAME_FORMAT_U8_420, width, height);
  frame = schro_frame_new_and_alloc (cpu_domain,
      SCHRO_FRAME_FORMAT_S16_420, width, height);
  frame_cuda = schro_frame_new_and_alloc (cuda_domain,
      SCHRO_FRAME_FORMAT_S16_420, width, height);
  frame_ref = schro_frame_new_and_alloc (cpu_domain,
      SCHRO_FRAME_FORMAT_S16_420, width, height);
  frame_test = schro_frame_new_and_alloc (cpu_domain,
      SCHRO_FRAME_FORMAT_S16_420, width, height);

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (frame_u8->components + 0, name, i);
    test_pattern_generate (frame_u8->components + 1, name, i);
    test_pattern_generate (frame_u8->components + 2, name, i);
    schro_frame_convert (frame, frame_u8);

    schro_frame_convert (frame_ref, frame_u8);
    schro_frame_inverse_iwt_transform (frame_ref, &params);

    schro_frame_to_gpu (frame_cuda, frame);
    schro_gpuframe_inverse_iwt_transform (frame_cuda, &params);
    schro_gpuframe_to_cpu (frame_test, frame_cuda);

    ok = frame_compare (frame_ref, frame_test);
    printf("  pattern %s: %s\n", name, ok ? "OK" : "broken");
    if (dump_all || !ok) {
      frame_data_dump_full (frame_test->components + 0,
          frame_ref->components + 0, frame->components + 0);
      failed = TRUE;
    }
  }

  schro_frame_unref (frame);
  schro_frame_unref (frame_ref);
  schro_frame_unref (frame_test);
  schro_frame_unref (frame_cuda);
}


