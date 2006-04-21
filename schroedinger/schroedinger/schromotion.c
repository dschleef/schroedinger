
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


void schro_decoder_predict (SchroDecoder *decoder);



static void
copy_block_4x4 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<4;j++){
    *(uint32_t *)(dest + dstr*j) = *(uint32_t *)(src + sstr*j);
  }
}

static void
copy_block_8x8 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<8;j++){
    *(uint64_t *)(dest + dstr*j) = *(uint64_t *)(src + sstr*j);
  }
}

#if 0
static void
copy_block (uint8_t *dest, int dstr, uint8_t *src, int sstr, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = src[sstr*j+i];
    }
  }
}
#endif

static void
splat_block (uint8_t *dest, int dstr, int value, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = value;
    }
  }
}

void
schro_frame_copy_with_motion (SchroFrame *dest, SchroFrame *src1,
    SchroFrame *src2, SchroMotionVector *motion_vectors, SchroParams *params)
{
  SchroFrame *frame = dest;
  SchroFrame *reference_frame = src1;
  int i, j;
  int dx, dy;
  int x, y;
  uint8_t *data;
  int stride;
  uint8_t *ref_data;
  int ref_stride;

  for(j=0;j<params->y_num_blocks;j++){
    for(i=0;i<params->x_num_blocks;i++){
      SchroMotionVector *mv = &motion_vectors[j*params->x_num_blocks + i];

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      if (mv->pred_mode == 0) {
        data = frame->components[0].data;
        stride = frame->components[0].stride;
        splat_block (data + y * stride + x, stride, mv->dc[0], 8, 8);

        data = frame->components[1].data;
        stride = frame->components[1].stride;
        splat_block (data + y/2 * stride + x/2, stride, mv->dc[1], 4, 4);

        data = frame->components[2].data;
        stride = frame->components[2].stride;
        splat_block (data + y/2 * stride + x/2, stride, mv->dc[2], 4, 4);
      } else {
        dx = mv->x;
        dy = mv->y;

        /* FIXME This is only roughly correct */
        SCHRO_ASSERT(x + dx >= 0);
        //SCHRO_ASSERT(x + dx < params->mc_luma_width - params->xbsep_luma);
        SCHRO_ASSERT(x + dx < params->mc_luma_width);
        SCHRO_ASSERT(y + dy >= 0);
        //SCHRO_ASSERT(y + dy < params->mc_luma_height - params->ybsep_luma);
        SCHRO_ASSERT(y + dy < params->mc_luma_height);

        data = frame->components[0].data;
        stride = frame->components[0].stride;
        ref_data = reference_frame->components[0].data;
        ref_stride = reference_frame->components[0].stride;
        copy_block_8x8 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

        x /= 2;
        dx /= 2;
        y /= 2;
        dy /= 2;

        data = frame->components[1].data;
        stride = frame->components[1].stride;
        ref_data = reference_frame->components[1].data;
        ref_stride = reference_frame->components[1].stride;
        copy_block_4x4 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

        data = frame->components[2].data;
        stride = frame->components[2].stride;
        ref_data = reference_frame->components[2].data;
        ref_stride = reference_frame->components[2].stride;
        copy_block_4x4 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);
      }
    }
  }
}

