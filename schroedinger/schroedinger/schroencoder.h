
#ifndef __SCHRO_ENCODER_H__
#define __SCHRO_ENCODER_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schrobits.h>
#include <schroedinger/schrobuffer.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schroqueue.h>
#include <schroedinger/schromotion.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroEncoder SchroEncoder;
typedef struct _SchroEncoderParams SchroEncoderParams;
typedef struct _SchroEncoderTask SchroEncoderTask;
typedef struct _SchroEncoderFrame SchroEncoderFrame;

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

typedef enum {
  SCHRO_STATE_NEED_FRAME,
  SCHRO_STATE_HAVE_BUFFER,
  SCHRO_STATE_AGAIN,
  SCHRO_STATE_END_OF_STREAM
} SchroStateEnum;

typedef enum {
  SCHRO_ENCODER_FRAME_STATE_NEW,
  SCHRO_ENCODER_FRAME_STATE_INITED,
  SCHRO_ENCODER_FRAME_STATE_ENCODING,
  SCHRO_ENCODER_FRAME_STATE_DONE,
  SCHRO_ENCODER_FRAME_STATE_ENGINE_1,
  SCHRO_ENCODER_FRAME_STATE_FREE
} SchroEncoderFrameStateEnum;

struct _SchroEncoderParams {
  int ignore;
};

struct _SchroEncoderFrame {
  int refcount;

  int valid;
  SchroEncoderFrameStateEnum state;

  int start_access_unit;

  SchroPictureNumber frame_number;
  SchroFrame *original_frame;
  SchroFrame *downsampled_frames[5];
  SchroUpsampledFrame *reconstructed_frame;

  SchroBuffer *access_unit_buffer;
  SchroBuffer *output_buffer;
  int presentation_frame;
  int slot;
  int last_frame;

  SchroEncoderTask *task;

  int is_ref;
  int num_refs;
  SchroPictureNumber picture_number_ref0;
  SchroPictureNumber picture_number_ref1;
  int n_retire;
  SchroPictureNumber retire;
};

struct _SchroEncoder {
  SchroAsync *async;

  SchroPictureNumber next_frame_number;

  SchroQueue *frame_queue;

  //int frame_queue_index;

  SchroQueue *reference_queue;
  //SchroEncoderFrame *reference_frames[SCHRO_MAX_REFERENCE_FRAMES];
  //int n_reference_frames;

  int need_rap;

  int version_major;
  int version_minor;
  int profile;
  int level;

  int video_format_index;

  SchroVideoFormat video_format;
  SchroEncoderParams encoder_params;

  int end_of_stream;
  int end_of_stream_handled;
  int end_of_stream_pulled;
  int completed_eos;
  int prev_offset;

  SchroPictureNumber au_frame;
  int au_distance;
  int next_slot;

  int output_slot;

  SchroBuffer *inserted_buffer;
  int queue_depth;
  int queue_changed;

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
  int completed;

  SchroEncoder *encoder;
  SchroParams params;
  SchroEncoderFrame *encoder_frame;
  
  int outbuffer_size;
  SchroBuffer *outbuffer;
  SchroBits *bits;
  SchroFrame *encode_frame;

  SchroFrame *iwt_frame;
  SchroFrame *prediction_frame;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int subband_size;
  SchroBuffer *subband_buffer;
  SchroSubband subbands[1+SCHRO_MAX_TRANSFORM_DEPTH*3];

  SchroEncoderFrame *ref_frame0;
  SchroEncoderFrame *ref_frame1;

  int16_t *quant_data;

  SchroMotionField *motion_field;

  //SchroPredictionList *predict_lists;
  SchroMotionField *motion_fields[32];

  int slot;
  int is_ref;
  SchroPictureNumber frame_number;
  SchroPictureNumber reference_frame_number[2];

  int n_retire;
  SchroPictureNumber retire[SCHRO_MAX_REFERENCE_FRAMES];

  int presentation_frame;

  /* engine specific stuff */

  /* intra_only */

  /* backref */

  int stats_dc;
  int stats_global;
  int stats_motion;

  int estimated_entropy;
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
int schro_encoder_push_ready (SchroEncoder *encoder);
void schro_encoder_push_frame (SchroEncoder *encoder, SchroFrame *frame);
int schro_encoder_iterate (SchroEncoder *encoder);

SchroBuffer * schro_encoder_encode_auxiliary_data (SchroEncoder *encoder,
    void *data, int size);
void schro_encoder_copy_to_frame_buffer (SchroEncoder *encoder, SchroBuffer *buffer);
void schro_encoder_encode_access_unit_header (SchroEncoder *encoder, SchroBits *bits);
void schro_encoder_encode_parse_info (SchroBits *bits, int parse_code);
void schro_encoder_insert_buffer (SchroEncoder *encoder, SchroBuffer *buffer);

SchroBuffer * schro_encoder_pull (SchroEncoder *encoder,
    int *n_decodable_frames);

int schro_encoder_preference_get_range (SchroEncoder *encoder,
    SchroPrefEnum pref, int *min, int *max);
int schro_encoder_preference_get (SchroEncoder *encoder, SchroPrefEnum pref);
int schro_encoder_preference_set (SchroEncoder *encoder, SchroPrefEnum pref,
    int value);

void schro_encoder_init_subbands (SchroEncoderTask *task);
void schro_encoder_encode_subband (SchroEncoderTask *task, int component, int index);
void schro_encoder_encode_picture (SchroEncoderTask *task);

SchroEncoderTask * schro_encoder_task_new (SchroEncoder *encoder);
void schro_encoder_task_free (SchroEncoderTask *task);
SchroFrame * schro_encoder_frame_queue_get (SchroEncoder *encoder,
    SchroPictureNumber frame_number);
void schro_encoder_frame_queue_remove (SchroEncoder *encoder,
    SchroPictureNumber frame_number);
void schro_encoder_reference_add (SchroEncoder *encoder, SchroEncoderFrame *encoder_frame);
SchroEncoderFrame * schro_encoder_reference_get (SchroEncoder *encoder,
    SchroPictureNumber frame_number);
void schro_encoder_encode_picture_header (SchroEncoderTask *task);
SchroBuffer * schro_encoder_encode_end_of_stream (SchroEncoder *encoder);
void schro_encoder_clean_up_transform (SchroEncoderTask *task);
void schro_encoder_init_subbands (SchroEncoderTask *task);
void schro_encoder_choose_quantisers (SchroEncoderTask *task);
void schro_encoder_encode_subband (SchroEncoderTask *task, int component, int index);
SchroBuffer * schro_encoder_encode_access_unit (SchroEncoder *encoder);
void schro_encoder_output_push (SchroEncoder *encoder,
    SchroBuffer *buffer, int slot, int presentation_frame);

SchroEncoderFrame * schro_encoder_frame_new (void);
void schro_encoder_frame_ref (SchroEncoderFrame *frame);
void schro_encoder_frame_unref (SchroEncoderFrame *frame);

SCHRO_END_DECLS

#endif

