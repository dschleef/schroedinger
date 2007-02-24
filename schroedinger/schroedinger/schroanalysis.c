
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrointernal.h>


void
schro_encoder_frame_analyse (SchroEncoder *encoder, SchroEncoderFrame *frame)
{
  int i;

  for(i=0;i<5;i++){
    frame->downsampled_frames[i] =
      schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8,
          ROUND_UP_SHIFT(encoder->video_format.width, i+1),
          ROUND_UP_SHIFT(encoder->video_format.height, i+1),
          ROUND_UP_SHIFT(encoder->video_format.chroma_width, i+1),
          ROUND_UP_SHIFT(encoder->video_format.chroma_height, i+1));
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

