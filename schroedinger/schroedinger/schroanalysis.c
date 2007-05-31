
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
      frame->original_frame, 1);
  schro_frame_downsample (frame->downsampled_frames[1],
      frame->downsampled_frames[0], 1);
  schro_frame_downsample (frame->downsampled_frames[2],
      frame->downsampled_frames[1], 1);
  schro_frame_downsample (frame->downsampled_frames[3],
      frame->downsampled_frames[2], 1);
  schro_frame_downsample (frame->downsampled_frames[4],
      frame->downsampled_frames[3], 1);

}

void
schro_encoder_reference_analyse (SchroEncoderFrame *frame)
{
  SCHRO_DEBUG("upsampling frame");

  frame->upsampled_h =
    schro_frame_new_and_alloc (frame->reconstructed_frame->format,
        frame->reconstructed_frame->width,
        frame->reconstructed_frame->height);
  frame->upsampled_v =
    schro_frame_new_and_alloc (frame->reconstructed_frame->format,
        frame->reconstructed_frame->width,
        frame->reconstructed_frame->height);
  frame->upsampled_hv =
    schro_frame_new_and_alloc (frame->reconstructed_frame->format,
        frame->reconstructed_frame->width,
        frame->reconstructed_frame->height);

  schro_frame_upsample_horiz (frame->upsampled_h, frame->reconstructed_frame);
  schro_frame_upsample_vert (frame->upsampled_h, frame->reconstructed_frame);
  schro_frame_upsample_vert (frame->upsampled_hv, frame->upsampled_h);

}

