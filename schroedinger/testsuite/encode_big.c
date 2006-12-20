
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <liboil/liboilrandom.h>

#define ROUND_UP_2(x) (((x) + 1) & ~1)
#define ROUND_UP_4(x) (((x) + 3) & ~3)
#define ROUND_UP_8(x) (((x) + 7) & ~7)

static void
frame_free (SchroFrame *frame, void *priv)
{
  free (priv);
}

void
test (int w, int h)
{
  int i;
  int size;
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

  size = ROUND_UP_4 (w) * ROUND_UP_2 (h);
  size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);
  size += (ROUND_UP_8 (w)/2) * (ROUND_UP_2 (h)/2);

  for(i=0;i<10;i++){
    picture = malloc(size);
    oil_random_u8(picture, size);

    frame = schro_frame_new_I420 (picture, w, h);

    schro_frame_set_free_callback (frame, frame_free, picture);

    schro_encoder_push_frame (encoder, frame);

    buffer = schro_encoder_encode (encoder);
    if (buffer) {
      schro_buffer_unref (buffer);
    }
  }
  schro_encoder_end_of_stream (encoder);
  while ((buffer = schro_encoder_encode (encoder))) {
    schro_buffer_unref (buffer);
  }

  schro_encoder_free (encoder);
}

int
main (int argc, char *argv[])
{
  schro_init();

  test(1920,1080);

  return 0;
}


