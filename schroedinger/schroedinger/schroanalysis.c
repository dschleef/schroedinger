
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>


void
schro_encoder_frame_analyse (SchroEncoder *encoder, SchroEncoderFrame *frame)
{
  int i;

  SCHRO_DEBUG("downsampling frame");

  for(i=0;i<5;i++){
    frame->downsampled_frames[i] =
      schro_frame_new_and_alloc (frame->original_frame->format,
          ROUND_UP_SHIFT(frame->original_frame->width, i+1),
          ROUND_UP_SHIFT(frame->original_frame->height, i+1));
  }

  schro_frame_downsample (frame->downsampled_frames[0],
      frame->original_frame);
  schro_frame_downsample (frame->downsampled_frames[1],
      frame->downsampled_frames[0]);
  schro_frame_downsample (frame->downsampled_frames[2],
      frame->downsampled_frames[1]);
  schro_frame_downsample (frame->downsampled_frames[3],
      frame->downsampled_frames[2]);
  schro_frame_downsample (frame->downsampled_frames[4],
      frame->downsampled_frames[3]);

}

