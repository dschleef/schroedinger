
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
  SchroFrame *tmp_frame0;
  SchroFrame *tmp_frame1;

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

  int wavelet_type;

  int version_major;
  int version_minor;
  int profile;
  int level;

  int video_format_index;

  SchroVideoFormat video_format;
  SchroParams params;
  SchroEncoderParams encoder_params;

  SchroBuffer *subband_buffer;

  SchroSubband subbands[1+6*3];

  int frame_number;
  int end_of_stream;

  SchroMotionVector *motion_vectors;
  SchroMotionVector *motion_vectors_dc;
#if 0
  SchroMotionVector *motion_vectors_none;
  SchroMotionVector *motion_vectors_scan;
#endif

  SchroPicture picture_list[10];
  int n_pictures;
  int picture_index;

  /* picture that is currently being encoded */
  SchroPicture *picture;

  /* current reference frames */
  SchroFrame *ref_frame0;
  SchroFrame *ref_frame1;

  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;

  int base_quant;

  double metric_to_cost;
  int stats_metric;
  int stats_dc_blocks;
  int stats_none_blocks;
  int stats_scan_blocks;
};

SchroEncoder * schro_encoder_new (void);
void schro_encoder_free (SchroEncoder *encoder);
SchroVideoFormat * schro_encoder_get_video_format (SchroEncoder *encoder);
void schro_encoder_set_video_format (SchroEncoder *encoder,
    SchroVideoFormat *video_format);
void schro_encoder_end_of_stream (SchroEncoder *encoder);
void schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame);
SchroBuffer * schro_encoder_encode (SchroEncoder *encoder);

void schro_encoder_copy_to_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);
void schro_encoder_encode_access_unit_header (SchroEncoder *encoder);
void schro_encoder_encode_intra (SchroEncoder *encoder);
void schro_encoder_encode_inter (SchroEncoder *encoder);
void schro_encoder_encode_parse_info (SchroEncoder *encoder, int parse_code);
void schro_encoder_encode_frame_prediction (SchroEncoder *encoder);
void schro_encoder_encode_transform_parameters (SchroEncoder *encoder);
void schro_encoder_encode_transform_data (SchroEncoder *encoder, int component);
void schro_encoder_encode_subband (SchroEncoder *encoder, int component, int index);
void schro_encoder_inverse_iwt_transform (SchroEncoder *encoder, int component);
void schro_encoder_copy_from_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);

#endif

