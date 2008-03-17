
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

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboilrandom.h>

void
schro_frame_clear (SchroFrame *frame)
{
  memset(frame->components[0].data, 0, frame->components[0].length);
  memset(frame->components[1].data, 0, frame->components[1].length);
  memset(frame->components[2].data, 0, frame->components[2].length);
}

void
schro_frame_create_pattern (SchroFrame *frame, int type)
{
  int i,j,k;
  SchroFrameData *comp;

  switch (type) {
    case 0:
      oil_random_u8 (frame->components[0].data, frame->components[0].length);
      oil_random_u8 (frame->components[1].data, frame->components[1].length);
      oil_random_u8 (frame->components[2].data, frame->components[2].length);
      break;
    case 1:
      for(k=0;k<3;k++){
        comp = &frame->components[k];
        for(j=0;j<comp->height;j++){
          for(i=0;i<comp->width;i++){
            SCHRO_GET(comp->data, j*comp->stride + i, uint8_t) = i*4 + j*2;
          }
        }
      }

      break;
  }
}

int
schro_frame_compare (SchroFrame *a, SchroFrame *b)
{
  SchroFrameData *comp_a;
  SchroFrameData *comp_b;
  int k;
  int i,j;

  for(k=0;k<3;k++){
    comp_a = &a->components[k];
    comp_b = &b->components[k];
    for(j=0;j<comp_a->height;j++){
      for(i=0;i<comp_a->width;i++){
        if (SCHRO_GET(comp_a->data, j*comp_a->stride + i, uint8_t) !=
            SCHRO_GET(comp_b->data, j*comp_b->stride + i, uint8_t)) {
          SCHRO_ERROR("difference comp=%d x=%d y=%d", k, i, j);
          return FALSE;
        }
      }
    }
  }

  return TRUE;
}

void
schro_frame_dump (SchroFrame *frame)
{
  SchroFrameData *comp;
  int i,j;

  comp = &frame->components[0];
  for(j=0;j<20;j++){
    for(i=0;i<20;i++) {
      printf("%-3d ", SCHRO_GET(comp->data, j*comp->stride + i, uint8_t));
    }
    printf("\n");
  }
}

int
main (int argc, char *argv[])
{
  SchroFrame *dest;
  SchroFrame *dest_u8;
  SchroFrame *ref;
  SchroUpsampledFrame *uref;
  SchroParams params;
  SchroVideoFormat video_format;
  SchroMotionVector *motion_vectors;
  int i,j;

  schro_init();

  video_format.width = 720;
  video_format.height = 480;
  video_format.chroma_format = SCHRO_CHROMA_420;
  schro_video_format_validate (&video_format);

  params.video_format = &video_format;
  params.xbsep_luma = 8;
  params.ybsep_luma = 8;
  params.xblen_luma = 12;
  params.yblen_luma = 12;

  schro_params_calculate_mc_sizes(&params);

  dest = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_420,
      video_format.width, video_format.height);
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420,
      video_format.width, video_format.height);
  dest_u8 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420,
      video_format.width, video_format.height);

  schro_frame_clear(dest);
  schro_frame_create_pattern(ref,1);

  motion_vectors = malloc(sizeof(SchroMotionVector) *
      params.x_num_blocks * params.y_num_blocks);
  memset (motion_vectors, 0, sizeof(SchroMotionVector) *
      params.x_num_blocks * params.y_num_blocks);
  
  printf("sizeof(SchroMotionVector) = %lu\n",(unsigned long) sizeof(SchroMotionVector));
  printf("num blocks %d x %d\n", params.x_num_blocks, params.y_num_blocks);
  for(j=0;j<params.y_num_blocks;j++){
    int jj;
    jj = j * params.x_num_blocks;
    for(i=0;i<params.x_num_blocks;i++){
#if 0
      if (i == 0 || i == 2 || i == params.x_num_blocks*2) {
        motion_vectors[jj+i].u.dc[0] = 100;
        motion_vectors[jj+i].u.dc[1] = 100;
        motion_vectors[jj+i].u.dc[2] = 0;
        motion_vectors[jj+i].pred_mode = 0;
      } else {
        motion_vectors[jj+i].u.dc[0] = 0;
        motion_vectors[jj+i].u.dc[1] = 0;
        motion_vectors[jj+i].u.dc[2] = 0;
        motion_vectors[jj+i].pred_mode = 0;
      }
#endif
      motion_vectors[jj+i].dx[0] = 0;
      motion_vectors[jj+i].dy[0] = -8*i;
      motion_vectors[jj+i].pred_mode = 1;
      motion_vectors[jj+i].split = 2;
    }
  }

  uref = schro_upsampled_frame_new (ref);

  {
    SchroMotion motion;

    motion.src1 = uref;
    motion.src2 = NULL;
    motion.motion_vectors = motion_vectors;
    motion.params = &params;
    schro_motion_render (&motion, dest);
  }

  schro_frame_convert (dest_u8, dest);
  schro_frame_dump (dest_u8);
  //schro_frame_compare (ref, dest_u8);

  schro_upsampled_frame_free (uref);
  schro_frame_unref (dest);
  schro_frame_unref (dest_u8);
  free (motion_vectors);

  return 0;
}

