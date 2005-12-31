
#ifndef __CARID_DECODER_H__
#define __CARID_DECODER_H__

#include <carid/caridbuffer.h>


typedef struct _CaridDecoder CaridDecoder;

struct _CaridDecoder {
  CaridBuffer *buffer;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int height;
  int width;

  int wavelet_type;
};

CaridDecoder * carid_decoder_new (void);
void carid_decoder_free (CaridDecoder *decoder);
void carid_decoder_set_output_buffer (CaridDecoder *decoder, CaridBuffer *buffer);
void carid_decoder_set_wavelet_type (CaridDecoder *decoder, int wavelet_type);
void carid_decoder_set_size (CaridDecoder *decoder, int width, int height);
void carid_decoder_decode (CaridDecoder *decoder, CaridBuffer *buffer);

#endif

