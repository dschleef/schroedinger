
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>



CaridEncoder *
carid_encoder_new (void)
{
  CaridEncoder *encoder;

  encoder = malloc(sizeof(CaridEncoder));
  memset (encoder, 0, sizeof(CaridEncoder));

  encoder->tmpbuf = malloc(1024 * 2);
  encoder->tmpbuf2 = malloc(1024 * 2);

  return encoder;
}

void
carid_encoder_free (CaridEncoder *encoder)
{
  if (encoder->buffer) {
    carid_buffer_unref (encoder->buffer);
  }

  free (encoder->tmpbuf);
  free (encoder->tmpbuf2);
  free (encoder);
}


void
carid_encoder_set_wavelet_type (CaridEncoder *encoder, int wavelet_type)
{
  encoder->wavelet_type = wavelet_type;
}

void
carid_encoder_set_size (CaridEncoder *encoder, int width, int height)
{
  if (encoder->width == width && encoder->height == height) return;

  encoder->width = width;
  encoder->height = height;

  if (encoder->buffer) {
    carid_buffer_unref (encoder->buffer);
    encoder->buffer = NULL;
  }
}

static int
round_up_pow2 (int x, int pow)
{
  x += (1<<pow) - 1;
  x &= ~((1<<pow) - 1);
  return x;
}

static void
notoil_splat_u16 (uint16_t *dest, uint16_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[0];
  }
}

CaridBuffer *
carid_encoder_encode (CaridEncoder *encoder, CaridBuffer *buffer)
{
  int w;
  int h;
  int16_t *tmp = encoder->tmpbuf;
  int level;
  int i;
  uint8_t *data;
  int16_t *enc_data;
  
  w = round_up_pow2(encoder->width, 6);
  h = round_up_pow2(encoder->height, 6);
  if (encoder->buffer == NULL) {
    encoder->buffer = carid_buffer_new_and_alloc (w*h*sizeof(int16_t));
  }

  data = (uint8_t *)buffer->data;
  enc_data = (int16_t *)encoder->buffer->data;

  for(i = 0; i<encoder->height; i++) {
    oil_convert_s16_u8 (enc_data + i*w, data + i*encoder->width, encoder->width);
    notoil_splat_u16 (enc_data + i*w + encoder->width,
        enc_data + i*w + encoder->width - 1,
        w - encoder->width);
  }
  for (i = encoder->height; i < h; i++) {
    oil_memcpy (enc_data + i*w, enc_data + (encoder->height - 1)*w,
        w*2);
  }

  for(level=0;level<6;level++) {
    int sb_w;
    int sb_h;

    sb_w = w >> level;
    sb_h = h >> level;

    for(i=0;i<sb_h;i++) {
      carid_lift_split (encoder->wavelet_type, tmp, enc_data + i*w, sb_w);
      carid_deinterleave (enc_data + i*w, tmp, sb_w);
    }
    for(i=0;i<sb_w;i++) {
      carid_lift_split_str (encoder->wavelet_type, tmp, enc_data + i, w*2, sb_h);
      carid_deinterleave_str (enc_data + i, w*2, tmp, sb_h);
    }

  }

  carid_buffer_ref (encoder->buffer);

  return encoder->buffer;
}

