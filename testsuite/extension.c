
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
  schro_init();

  test (100, 100);

  if (failed) {
    printf("FAILED\n");
  } else {
    printf("SUCCESS\n");
  }

  return failed;
}

void
schro_upsampled_frame_get_subdata_prec1 (SchroUpsampledFrame *upframe,
    int component, int x, int y, SchroFrameData *fd);

void test (int width, int height)
{
  SchroFrame *frame;
  SchroUpsampledFrame *upframe;
  char name[TEST_PATTERN_NAME_SIZE];
  SchroFrameData fd_ref;
  SchroFrameData fd;
  int x,y;
  int ok;

  fd_ref.data = schro_malloc (16 * 16);
  fd_ref.stride = 16;
  fd_ref.width = 16;
  fd_ref.height = 16;;

  fd.width = 16;
  fd.height = 16;;

  frame = schro_frame_new_and_alloc_extended (NULL, SCHRO_FRAME_FORMAT_U8_420,
      width, height, 32);

  test_pattern_generate (frame->components + 0, name, 0);
  test_pattern_generate (frame->components + 1, name, 0);
  test_pattern_generate (frame->components + 2, name, 0);

  schro_frame_mc_edgeextend (frame);

  upframe = schro_upsampled_frame_new (schro_frame_ref(frame));
  schro_upsampled_frame_upsample (upframe);

#if 0
  /* just the corners */
  for(y=-10;y<-8;y++){
    for(x=-10;x<-8;x++){
      printf ("%d,%d\n", x, y);

      schro_upsampled_frame_get_block_precN (upframe, 0, x, y, 1, &fd_ref);
      schro_upsampled_frame_get_subdata_prec1 (upframe, 0, x, y, &fd);
      frame_data_dump_full (&fd, &fd_ref, &fd_ref);
    }
  }

  for(y=200-16;y<200-16+2;y++){
    for(x=184;x<186;x++){
      printf ("%d,%d\n", x, y);

      schro_upsampled_frame_get_block_precN (upframe, 0, x, y, 1, &fd_ref);
      schro_upsampled_frame_get_subdata_prec1 (upframe, 0, x, y, &fd);
      frame_data_dump_full (&fd, &fd_ref, &fd_ref);
    }
  }

  for(y=-10;y<-8;y++){
    for(x=184;x<186;x++){
      printf ("%d,%d\n", x, y);

      schro_upsampled_frame_get_block_precN (upframe, 0, x, y, 1, &fd_ref);
      schro_upsampled_frame_get_subdata_prec1 (upframe, 0, x, y, &fd);
      frame_data_dump_full (&fd, &fd_ref, &fd_ref);
    }
  }

  for(y=184;y<186;y++){
    for(x=-10;x<-8;x++){
      printf ("%d,%d\n", x, y);

      schro_upsampled_frame_get_block_precN (upframe, 0, x, y, 1, &fd_ref);
      schro_upsampled_frame_get_subdata_prec1 (upframe, 0, x, y, &fd);
      frame_data_dump_full (&fd, &fd_ref, &fd_ref);
    }
  }
#endif

  for(y=-32*2;y<100+32*2-16*2;y++) {
    for(x=-32*2;x<100+32*2-16*2;x++) {
      schro_upsampled_frame_get_block_fast_precN (upframe, 0, x, y, 1, &fd_ref, &fd_ref);
      schro_upsampled_frame_get_subdata_prec1 (upframe, 0, x, y, &fd);
      ok = frame_data_compare (&fd, &fd_ref);
      if (dump_all || !ok) {
        printf ("%d,%d\n", x, y);
        frame_data_dump_full (&fd, &fd_ref, &fd_ref);
      }
      if (!ok) {
        failed = TRUE;
      }
    }
  }

  schro_frame_unref (frame);
}

