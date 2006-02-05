
#ifndef __CARID_DECODER_H__
#define __CARID_DECODER_H__

#include <carid/caridbuffer.h>
#include <carid/caridparams.h>


typedef struct _CaridDecoder CaridDecoder;

struct _CaridDecoder {
  CaridBuffer *frame_buffer[3];
  CaridBuffer *output_buffer;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int code;
  int rap_frame_number;
  int next_parse_offset;
  int prev_parse_offset;
  int frame_number_offset;

  CaridBits *bits;

  CaridParams params;

  CaridSubband subbands[1+6*3];
};

CaridDecoder * carid_decoder_new (void);
void carid_decoder_free (CaridDecoder *decoder);
void carid_decoder_set_output_buffer (CaridDecoder *decoder, CaridBuffer *buffer);
int carid_decoder_is_parse_header (CaridBuffer *buffer);
int carid_decoder_is_rap (CaridBuffer *buffer);
void carid_decoder_decode (CaridDecoder *decoder, CaridBuffer *buffer);
void carid_decoder_decode_parse_header (CaridDecoder *decoder);
void carid_decoder_decode_rap (CaridDecoder *decoder);
void carid_decoder_decode_frame_header (CaridDecoder *decoder);
void carid_decoder_decode_transform_parameters (CaridDecoder *decoder);
void carid_decoder_decode_transform_data (CaridDecoder *decoder, int component);
void carid_decoder_decode_subband (CaridDecoder *decoder, int component, int index);
void carid_decoder_iwt_transform (CaridDecoder *decoder, int component);
void carid_decoder_copy_from_frame_buffer (CaridDecoder *decoder, CaridBuffer *buffer);

#endif

