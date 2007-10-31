
#ifndef __SCHRO_MOTION_H__
#define __SCHRO_MOTION_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/schroparams.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroUpsampledFrame SchroUpsampledFrame;
struct _SchroUpsampledFrame {
  SchroFrame *frames[4];
};

#ifndef SCHRO_DISABLE_UNSTABLE_API

typedef struct _SchroObmc SchroObmc;
typedef struct _SchroObmcRegion SchroObmcRegion;
typedef struct _SchroMotion SchroMotion;

struct _SchroObmcRegion {
  int16_t *weights;
  int start_x;
  int start_y;
  int end_x;
  int end_y;
};

struct _SchroObmc {
  SchroObmcRegion regions[9];
  int16_t *region_data;
  int stride;
  int shift;
  int x_ramp;
  int y_ramp;
  int x_len;
  int y_len;
  int x_sep;
  int y_sep;
  uint8_t *tmpdata;
};

struct _SchroMotion {
  SchroUpsampledFrame *src1;
  SchroUpsampledFrame *src2;
  SchroMotionVector *motion_vectors;
  SchroParams *params;

  int sx_max;
  int sy_max;
  uint8_t *tmpdata;
  SchroObmc *obmc_luma;
  SchroObmc *obmc_chroma;
  uint8_t *blocks[3];
  int strides[3];
};

void schro_frame_copy_with_motion (SchroFrame *dest, SchroMotion *motion);
void schro_motion_dc_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred);
void schro_motion_vector_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred_x, int *pred_y, int mode);
int schro_motion_split_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y);
void schro_motion_field_get_global_prediction (SchroMotionField *mf,
    int x, int y, int *pred);
int schro_motion_get_mode_prediction (SchroMotionField *mf, int x, int y);
int schro_motion_verify (SchroMotion *mf);

void schro_obmc_init (SchroObmc *obmc, int x_len, int y_len, int x_sep,
    int y_sep);
void schro_obmc_cleanup (SchroObmc *obmc);

void schro_upsampled_frame_upsample (SchroUpsampledFrame *df);
SchroUpsampledFrame * schro_upsampled_frame_new (SchroFrame *frame);
void schro_upsampled_frame_free (SchroUpsampledFrame *df);

#endif

SCHRO_END_DECLS

#endif

