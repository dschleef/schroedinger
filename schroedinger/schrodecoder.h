
#ifndef __SCHRO_DECODER_H__
#define __SCHRO_DECODER_H__

#include <schroedinger/schrobuffer.h>
#include <schroedinger/schroparams.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schrounpack.h>
#include <schroedinger/schrobitstream.h>
#include <schroedinger/schroqueue.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroDecoder SchroDecoder;
typedef struct _SchroPicture SchroPicture;

#ifdef SCHRO_ENABLE_UNSTABLE_API
struct _SchroDecoder {
  /*< private >*/

  /* the list of reference pictures */
  SchroQueue *reference_queue;

  /* a list of frames provided by the app that we'll decode into */
  SchroQueue *output_queue;

  SchroPictureNumber next_frame_number;

  SchroPicture *picture;

  int major_version;
  int minor_version;
  int profile;
  int level;
  schro_bool interlaced_coding;
  SchroVideoFormat video_format;

  SchroQueue *frame_queue;

  SchroPictureNumber earliest_frame;

  int have_access_unit;
  int have_frame_number;

  double skip_value;
  double skip_ratio;

  int error;
  char *error_message;

  int has_md5;
  uint8_t md5_checksum[32];
};

struct _SchroPicture {
  int refcount;

  SchroDecoder *decoder;

  SchroBuffer *input_buffer;
  SchroParams params;
  SchroPictureNumber picture_number;
  int n_refs;
  SchroPictureNumber reference1;
  SchroPictureNumber reference2;
  SchroPictureNumber retired_picture_number;
  SchroUpsampledFrame *ref0;
  SchroUpsampledFrame *ref1;
  SchroFrame *planar_output_frame;

  int16_t *tmpbuf;
  int16_t *tmpbuf2;

  int parse_code;
  int next_parse_offset;
  int prev_parse_offset;

  SchroUnpack unpack;

  int zero_residual;

  SchroFrame *frame;
  SchroFrame *mc_tmp_frame;
  SchroMotion *motion;
  SchroFrame *output_picture;

};
#endif

enum {
  SCHRO_DECODER_OK,
  SCHRO_DECODER_ERROR,
  SCHRO_DECODER_EOS,
  SCHRO_DECODER_FIRST_ACCESS_UNIT,
  SCHRO_DECODER_NEED_BITS,
  SCHRO_DECODER_NEED_FRAME
};

SchroDecoder * schro_decoder_new (void);
void schro_decoder_free (SchroDecoder *decoder);
void schro_decoder_reset (SchroDecoder *decoder);
SchroVideoFormat * schro_decoder_get_video_format (SchroDecoder *decoder);
void schro_decoder_add_output_picture (SchroDecoder *decoder, SchroFrame *frame);
void schro_decoder_push (SchroDecoder *decoder, SchroBuffer *buffer);
SchroFrame *schro_decoder_pull (SchroDecoder *decoder);
int schro_decoder_is_parse_header (SchroBuffer *buffer);
int schro_decoder_is_access_unit (SchroBuffer *buffer);
int schro_decoder_is_intra (SchroBuffer *buffer);
int schro_decoder_is_picture (SchroBuffer *buffer);
int schro_decoder_iterate (SchroDecoder *decoder);

void schro_decoder_set_earliest_frame (SchroDecoder *decoder, SchroPictureNumber earliest_frame);
void schro_decoder_set_skip_ratio (SchroDecoder *decoder, double ratio);

#ifdef SCHRO_ENABLE_UNSTABLE_API

void schro_decoder_decode_parse_header (SchroDecoder *decoder);
void schro_decoder_decode_access_unit (SchroDecoder *decoder);
void schro_decoder_decode_picture_header (SchroPicture *picture);
void schro_decoder_decode_picture_prediction_parameters (SchroPicture *picture);
void schro_decoder_decode_block_data (SchroPicture *picture);
void schro_decoder_decode_transform_parameters (SchroPicture *picture);
void schro_decoder_decode_transform_data (SchroPicture *picture);
void schro_decoder_decode_lowdelay_transform_data (SchroPicture *picture);
void schro_decoder_iwt_transform (SchroDecoder *decoder, int component);
void schro_decoder_copy_from_frame_buffer (SchroDecoder *decoder, SchroBuffer *buffer);

void schro_decoder_subband_dc_predict (SchroFrameData *fd);

void schro_decoder_decode_lowdelay_transform_data_2 (SchroPicture *picture);

SchroPicture * schro_picture_new (SchroDecoder *decoder);
SchroPicture * schro_picture_ref (SchroPicture *picture);
void schro_picture_unref (SchroPicture *picture);





#endif

SCHRO_END_DECLS

#endif

