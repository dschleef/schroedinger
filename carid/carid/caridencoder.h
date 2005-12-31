
#ifndef __CARID_ENCODER_H__
#define __CARID_ENCODER_H__

#include <carid/caridbuffer.h>


typedef struct _CaridEncoder CaridEncoder;

struct _CaridEncoder {
  CaridBuffer *buffer;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int width;
  int height;

  int wavelet_type;
};

CaridEncoder * carid_encoder_new (void);
void carid_encoder_free (CaridEncoder *encoder);
void carid_encoder_set_size (CaridEncoder *encoder, int width, int height);
void carid_encoder_set_wavelet_type (CaridEncoder *encoder, int wavelet_type);
CaridBuffer * carid_encoder_encode (CaridEncoder *encoder, CaridBuffer *buffer);

#endif

