
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <liboil/liboilrandom.h>

static void
frame_free (SchroFrame *frame, void *priv)
{
  free (priv);
}

void
test (int w, int h)
{
  int size;
  int go;
  int n_frames;
  uint8_t *picture;
  SchroEncoder *encoder;
  SchroBuffer *buffer;
  SchroFrame *frame;
  SchroVideoFormat *format;

  encoder = schro_encoder_new();
  format = schro_encoder_get_video_format(encoder);
  format->width = w;
  format->height = h;
  schro_encoder_set_video_format (encoder, format);
  free (format);

  size = ROUND_UP_4 (w) * ROUND_UP_2 (h);
  size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);
  size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);

  n_frames = 0;
  go = 1;
  while (go) {
    int x;

    switch (schro_encoder_iterate (encoder)) {
      case SCHRO_STATE_NEED_FRAME:
        if (n_frames < 10) {
          //SCHRO_ERROR("frame %d", n_frames);

          picture = malloc(size);
          oil_random_u8(picture, size);

          frame = schro_frame_new_from_data_I420 (picture, w, h);

          schro_frame_set_free_callback (frame, frame_free, picture);

          schro_encoder_push_frame (encoder, frame);

          n_frames++;
        }
        if (n_frames == 10) {
          schro_encoder_end_of_stream (encoder);
          n_frames++;
        }
        break;
      case SCHRO_STATE_HAVE_BUFFER:
        buffer = schro_encoder_pull (encoder, &x);
        schro_buffer_unref (buffer);
        break;
      case SCHRO_STATE_AGAIN:
        break;
      case SCHRO_STATE_END_OF_STREAM:
        go = 0;
        break;
      default:
        break;
    }
  }

  schro_encoder_free (encoder);
}

int
main (int argc, char *argv[])
{
  schro_init();

  test(1920,1080);
  //test(64,64);

  return 0;
}


