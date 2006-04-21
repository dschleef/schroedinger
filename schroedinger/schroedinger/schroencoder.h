
#ifndef __SCHRO_ENCODER_H__
#define __SCHRO_ENCODER_H__

#include <schroedinger/schrobuffer.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/schroframe.h>


typedef struct _SchroEncoder SchroEncoder;
typedef struct _SchroEncoderParams SchroEncoderParams;

struct _SchroEncoderParams {
  int quant_index_dc;
  int quant_index[6];
};

struct _SchroEncoder {
  SchroFrame *frame;
  SchroFrame *encode_frame;

  SchroBits *bits;

  SchroFrame *frame_queue[10];
  int frame_queue_length;

  int frame_queue_index;

  SchroFrame *reference_frames[10];
  int n_reference_frames;

  int need_rap;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  //int width;
  //int height;

  int wavelet_type;

  SchroParams params;
  SchroEncoderParams encoder_params;

  SchroBuffer *subband_buffer;

  SchroSubband subbands[1+6*3];

  int frame_number;

  SchroMotionVector *motion_vectors;

  SchroPicture picture_list[10];
  int n_pictures;
  int picture_index;

  /* picture that is currently being encoded */
  SchroPicture *picture;

  /* current reference frames */
  SchroFrame *ref_frame0;
  SchroFrame *ref_frame1;
};

SchroEncoder * schro_encoder_new (void);
void schro_encoder_free (SchroEncoder *encoder);
void schro_encoder_set_size (SchroEncoder *encoder, int width, int height);
void schro_encoder_set_wavelet_type (SchroEncoder *encoder, int wavelet_type);
void schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame);
SchroBuffer * schro_encoder_encode (SchroEncoder *encoder);
void schro_encoder_copy_to_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);
void schro_encoder_encode_rap (SchroEncoder *encoder);
void schro_encoder_encode_intra (SchroEncoder *encoder);
void schro_encoder_encode_inter (SchroEncoder *encoder);
void schro_encoder_encode_frame_header (SchroEncoder *encoder, int parse_code);
void schro_encoder_encode_frame_prediction (SchroEncoder *encoder);
void schro_encoder_encode_transform_parameters (SchroEncoder *encoder);
void schro_encoder_encode_transform_data (SchroEncoder *encoder, int component);
void schro_encoder_encode_subband (SchroEncoder *encoder, int component, int index);
void schro_encoder_inverse_iwt_transform (SchroEncoder *encoder, int component);
void schro_encoder_copy_from_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);
void schro_encoder_motion_predict (SchroEncoder *encoder);

#endif

