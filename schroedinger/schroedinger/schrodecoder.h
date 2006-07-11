
#ifndef __SCHRO_DECODER_H__
#define __SCHRO_DECODER_H__

#include <schroedinger/schrobuffer.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/schroframe.h>


typedef struct _SchroDecoder SchroDecoder;

struct _SchroDecoder {
  SchroFrame *frame;
  SchroFrame *mc_tmp_frame;
  int n_reference_frames;
  SchroFrame *reference_frames[10];
  SchroFrame *output_frames[10];
  int n_output_frames;

  int next_frame_number;

  int picture_number;
  int n_refs;
  int reference1;
  int reference2;
  SchroFrame *ref0;
  SchroFrame *ref1;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int code;
  int rap_frame_number;
  int next_parse_offset;
  int prev_parse_offset;

  SchroBits *bits;

  SchroParams params;

  SchroSubband subbands[1+6*3];

  SchroMotionVector *motion_vectors;

  int n_retire;
  int retire_list[10];

  int frame_queue_length;
  SchroFrame *frame_queue[10];
};

SchroDecoder * schro_decoder_new (void);
void schro_decoder_free (SchroDecoder *decoder);
void schro_decoder_add_output_frame (SchroDecoder *decoder, SchroFrame *frame);
int schro_decoder_is_parse_header (SchroBuffer *buffer);
int schro_decoder_is_rap (SchroBuffer *buffer);
SchroFrame *schro_decoder_decode (SchroDecoder *decoder, SchroBuffer *buffer);
void schro_decoder_decode_parse_header (SchroDecoder *decoder);
void schro_decoder_decode_rap (SchroDecoder *decoder);
void schro_decoder_decode_frame_header (SchroDecoder *decoder);
void schro_decoder_decode_frame_prediction (SchroDecoder *decoder);
void schro_decoder_decode_prediction_data (SchroDecoder *decoder);
void schro_decoder_decode_transform_parameters (SchroDecoder *decoder);
void schro_decoder_decode_transform_data (SchroDecoder *decoder, int component);
void schro_decoder_decode_subband (SchroDecoder *decoder, int component, int index);
void schro_decoder_iwt_transform (SchroDecoder *decoder, int component);
void schro_decoder_copy_from_frame_buffer (SchroDecoder *decoder, SchroBuffer *buffer);

#endif

