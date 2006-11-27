
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

typedef enum {
  SCHRO_PREF_ENGINE,
  SCHRO_PREF_REF_DISTANCE,
  SCHRO_PREF_TRANSFORM_DEPTH,
  SCHRO_PREF_INTRA_WAVELET,
  SCHRO_PREF_INTER_WAVELET,
  SCHRO_PREF_QUANT_BASE,
  SCHRO_PREF_QUANT_OFFSET_NONREF,
  SCHRO_PREF_QUANT_OFFSET_SUBBAND,
  SCHRO_PREF_QUANT_DC,
  SCHRO_PREF_QUANT_DC_OFFSET_NONREF,
  SCHRO_PREF_LAST
} SchroPrefEnum;

struct _SchroEncoderParams {
  int ignore;
};

struct _SchroEncoderReference {
  int valid;
  int frame_number;
  SchroFrame *frames[5];
};

struct _SchroEncoder {
  SchroEncoderTask *task;

  SchroFrame *frame_queue[SCHRO_FRAME_QUEUE_LENGTH];
  int frame_queue_length;

  int frame_queue_index;

  SchroEncoderReference reference_frames[SCHRO_MAX_REFERENCE_FRAMES];
  int n_reference_frames;

  int need_rap;

  int version_major;
  int version_minor;
  int profile;
  int level;

  int video_format_index;

  SchroVideoFormat video_format;
  SchroEncoderParams encoder_params;

  //int frame_number;
  int end_of_stream;
  int prev_offset;

  int au_frame;
  int au_distance;
  int next_slot;
  int next_frame;

  int output_slot;
  struct {
    int slot;
    int presentation_frame;
    SchroBuffer *buffer;
  } output_queue[SCHRO_FRAME_QUEUE_LENGTH];

  SchroPicture picture_list[SCHRO_FRAME_QUEUE_LENGTH];
  int n_pictures;
  int picture_index;

#if 0
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
#endif

  int engine_init;
  int engine;

  int prefs[SCHRO_PREF_LAST];

  /* engine specific stuff */

  int last_ref;
  int ref_distance;
  int next_ref;
  int mid1_ref;
  int mid2_ref;
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

  SchroMotionField *motion_field;

  //SchroPredictionList *predict_lists;
  SchroMotionField *motion_fields[32];

  SchroEncoderReference *dest_ref;

  int slot;
  int is_ref;
  int frame_number;
  int reference_frame_number[2];

  int n_retire;
  int retire[SCHRO_MAX_REFERENCE_FRAMES];

  int presentation_frame;

  /* engine specific stuff */

  /* intra_only */

  /* backref */

  /* tworef */
  double metric_to_cost;
  int stats_metric;
  int stats_dc_blocks;
  int stats_none_blocks;
  int stats_scan_blocks;
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

enum {
  SCHRO_MOTION_FIELD_HIER_REF0,
  SCHRO_MOTION_FIELD_HIER1_REF0,
  SCHRO_MOTION_FIELD_HIER2_REF0,
  SCHRO_MOTION_FIELD_HIER3_REF0,
  SCHRO_MOTION_FIELD_HIER_REF1,
  SCHRO_MOTION_FIELD_HIER1_REF1,
  SCHRO_MOTION_FIELD_HIER2_REF1,
  SCHRO_MOTION_FIELD_HIER3_REF1,
  SCHRO_MOTION_FIELD_DC,
  SCHRO_MOTION_FIELD_GLOBAL_REF0,
  SCHRO_MOTION_FIELD_GLOBAL_REF1,
  SCHRO_MOTION_FIELD_ZERO_REF0,
  SCHRO_MOTION_FIELD_ZERO_REF1,
  SCHRO_MOTION_FIELD_LAST
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

int schro_encoder_preference_get_range (SchroEncoder *encoder,
    SchroPrefEnum pref, int *min, int *max);
int schro_encoder_preference_get (SchroEncoder *encoder, SchroPrefEnum pref);
int schro_encoder_preference_set (SchroEncoder *encoder, SchroPrefEnum pref,
    int value);


#endif

