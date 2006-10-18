
#ifndef __SCHRO_ENCODER_H__
#define __SCHRO_ENCODER_H__

#include <schroedinger/schrobuffer.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/schroframe.h>


typedef struct _SchroEncoder SchroEncoder;
typedef struct _SchroEncoderParams SchroEncoderParams;
typedef struct _SchroEncoderTask SchroEncoderTask;
typedef struct _SchroEncoderReference SchroEncoderReference;

/* forward reference */
typedef struct _SchroPredictionVector SchroPredictionVector;
typedef struct _SchroPredictionList SchroPredictionList;

struct _SchroEncoderParams {
  int ignore;
};

struct _SchroEncoderReference {
  int valid;
  int frame_number;
  SchroFrame *frames[5];
};

struct _SchroEncoder {
  //SchroFrame *tmp_frame0;
  //SchroFrame *tmp_frame1;

  //SchroFrame *encode_frame;

  //SchroBits *bits;
  
  SchroEncoderTask *task;

  SchroFrame *frame_queue[SCHRO_MAX_REFERENCE_FRAMES];
  int frame_queue_length;

  int frame_queue_index;

  SchroEncoderReference reference_frames[SCHRO_MAX_REFERENCE_FRAMES];
  int n_reference_frames;

  int need_rap;

  //int16_t *tmpbuf;
  //int16_t *tmpbuf2;

  int version_major;
  int version_minor;
  int profile;
  int level;

  int video_format_index;

  SchroVideoFormat video_format;
  //SchroParams params;
  SchroEncoderParams encoder_params;

  //SchroBuffer *subband_buffer;

  //SchroSubband subbands[1+SCHRO_MAX_TRANSFORM_DEPTH*3];

  int frame_number;
  int end_of_stream;
  int prev_offset;

  int last_au_frame;
  int au_distance;
  int next_slot;
  int next_frame;

  int output_slot;
  struct {
    int slot;
    int presentation_frame;
    SchroBuffer *buffer;
  } output_queue[10];

  //SchroMotionVector *motion_vectors;
  //SchroPredictionList *predict_lists;

  SchroPicture picture_list[10];
  int n_pictures;
  int picture_index;

  /* picture that is currently being encoded */
  //SchroPicture *picture;

  /* current reference frames */
  //SchroEncoderReference *ref_frame0;
  //SchroEncoderReference *ref_frame1;

  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;

  //double metric_to_cost;
  //int stats_metric;
  //int stats_dc_blocks;
  //int stats_none_blocks;
  //int stats_scan_blocks;

  //int16_t *quant_data;
  
  int engine_init;
};

struct _SchroEncoderTask {
  int state;

  SchroEncoder *encoder;
  SchroParams params;
  
  SchroBuffer *outbuffer;
  SchroBits *bits;
  SchroFrame *encode_frame;

  SchroFrame *tmp_frame0;
  SchroFrame *tmp_frame1;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  SchroBuffer *subband_buffer;
  SchroSubband subbands[1+SCHRO_MAX_TRANSFORM_DEPTH*3];

  SchroEncoderReference *ref_frame0;
  SchroEncoderReference *ref_frame1;

  int16_t *quant_data;

  SchroMotionVector *motion_vectors;
  SchroPredictionList *predict_lists;

  double metric_to_cost;
  int stats_metric;
  int stats_dc_blocks;
  int stats_none_blocks;
  int stats_scan_blocks;

  SchroEncoderReference *dest_ref;

  int slot;
  int is_ref;
  int frame_number;
  int reference_frame_number[2];

  int n_retire;
  int retire[SCHRO_MAX_REFERENCE_FRAMES];

  int presentation_frame;
};

struct _SchroEncoderSettings {
  int transform_depth;
  int wavelet_filter_index;

#if 0
  /* stuff we don't handle yet */
  int profile;
  int level;

  int xbsep_luma;
  int ybsep_luma;
  int xblen_luma;
  int yblen_luma;
#endif

};

SchroEncoder * schro_encoder_new (void);
void schro_encoder_free (SchroEncoder *encoder);
SchroVideoFormat * schro_encoder_get_video_format (SchroEncoder *encoder);
void schro_encoder_set_video_format (SchroEncoder *encoder,
    SchroVideoFormat *video_format);
void schro_encoder_end_of_stream (SchroEncoder *encoder);
void schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame);
int schro_encoder_iterate (SchroEncoder *encoder);
SchroBuffer * schro_encoder_encode (SchroEncoder *encoder);

void schro_encoder_copy_to_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);
void schro_encoder_encode_access_unit_header (SchroEncoder *encoder, SchroBits *bits);
void schro_encoder_encode_parse_info (SchroBits *bits, int parse_code);

SchroBuffer * schro_encoder_pull (SchroEncoder *encoder,
    int *n_decodable_frames);

#endif

