
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>



CaridDecoder *
carid_decoder_new (void)
{
  CaridDecoder *decoder;

  decoder = malloc(sizeof(CaridDecoder));
  memset (decoder, 0, sizeof(CaridDecoder));

  decoder->tmpbuf = malloc(1024 * 2);
  decoder->tmpbuf2 = malloc(1024 * 2);

  return decoder;
}

void
carid_decoder_free (CaridDecoder *decoder)
{
  if (decoder->buffer) {
    carid_buffer_unref (decoder->buffer);
  }

  free (decoder->tmpbuf);
  free (decoder->tmpbuf2);
  free (decoder);
}

void
carid_decoder_set_output_buffer (CaridDecoder *decoder, CaridBuffer *buffer)
{
  if (decoder->buffer) {
    carid_buffer_unref (decoder->buffer);
  }
  decoder->buffer = buffer;
}


void
carid_decoder_set_wavelet_type (CaridDecoder *decoder, int wavelet_type)
{
  decoder->wavelet_type = wavelet_type;
}

void
carid_decoder_set_size (CaridDecoder *decoder, int width, int height)
{
  if (decoder->width == width && decoder->height == height) return;

  decoder->width = width;
  decoder->height = height;
}

static int
round_up_pow2 (int x, int pow)
{
  x += (1<<pow) - 1;
  x &= ~((1<<pow) - 1);
  return x;
}

void
carid_decoder_decode (CaridDecoder *decoder, CaridBuffer *buffer)
{
  int w;
  int h;
  int16_t *tmp = decoder->tmpbuf;
  int level;
  int i;
  int16_t *data;
  uint8_t *dec_data;
  
  data = (int16_t *)buffer->data;
  dec_data = (uint8_t *)decoder->buffer->data;

  w = round_up_pow2(decoder->width, 6);
  h = round_up_pow2(decoder->height, 6);

  for(level=0;level<6;level++) {
    int sb_w;
    int sb_h;

    sb_w = w >> (5 - level);
    sb_h = h >> (5 - level);

    for(i=0;i<sb_w;i++) {
      carid_interleave_str (tmp, data + i, w*2, sb_h);
      carid_lift_synth_str (decoder->wavelet_type, data + i, w*2, tmp, sb_h);
    }
    for(i=0;i<sb_h;i++) {
      carid_interleave (tmp, data + i*w, sb_w);
      carid_lift_synth (decoder->wavelet_type, data + i*w, tmp, sb_w);
    }

  }

  for(i=0;i<decoder->height;i++){
    oil_convert_u8_s16 (dec_data + i*decoder->width, data + i*w,
        decoder->width);
  }

  carid_buffer_unref (buffer);
}

