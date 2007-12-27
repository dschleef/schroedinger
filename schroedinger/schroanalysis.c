
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrooil.h>

void
schro_encoder_frame_downsample (SchroEncoderFrame *frame)
{
  int i;

  SCHRO_DEBUG("downsampling frame %d", frame->frame_number);

  for(i=0;i<5;i++){
    frame->downsampled_frames[i] =
      schro_frame_new_and_alloc (frame->filtered_frame->format,
          ROUND_UP_SHIFT(frame->filtered_frame->width, i+1),
          ROUND_UP_SHIFT(frame->filtered_frame->height, i+1));
  }

  schro_frame_downsample (frame->downsampled_frames[0],
      frame->filtered_frame);
  schro_frame_downsample (frame->downsampled_frames[1],
      frame->downsampled_frames[0]);
  schro_frame_downsample (frame->downsampled_frames[2],
      frame->downsampled_frames[1]);
  schro_frame_downsample (frame->downsampled_frames[3],
      frame->downsampled_frames[2]);
  schro_frame_downsample (frame->downsampled_frames[4],
      frame->downsampled_frames[3]);
}

static double
schro_frame_component_squared_error (SchroFrameData *a,
    SchroFrameData *b)
{
  int j;
  double sum;
  
  SCHRO_ASSERT(a->width == b->width);
  SCHRO_ASSERT(a->height == b->height);

  sum = 0;
  for(j=0;j<a->height;j++){
    int32_t linesum;

    oil_sum_square_diff_u8 (&linesum, OFFSET(a->data, a->stride * j),
        OFFSET(b->data, b->stride * j), a->width);
    sum += linesum;
  }
  return sum;
}

double
schro_frame_mean_squared_error (SchroFrame *a, SchroFrame *b)
{
  double sum, n;

  sum = schro_frame_component_squared_error (&a->components[0],
      &b->components[0]);
  n = a->components[0].width * a->components[0].height;

  sum += schro_frame_component_squared_error (&a->components[1],
      &b->components[1]);
  n += a->components[1].width * a->components[1].height;

  sum += schro_frame_component_squared_error (&a->components[2],
      &b->components[2]);
  n += a->components[2].width * a->components[2].height;

  return sum/n;
}


