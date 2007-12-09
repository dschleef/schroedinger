
#ifndef __SCHRO_MOTION_H__
#define __SCHRO_MOTION_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/schroparams.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroObmc SchroObmc;
typedef struct _SchroObmcRegion SchroObmcRegion;
typedef struct _SchroMotionVector SchroMotionVector;
typedef struct _SchroMotionVectorDC SchroMotionVectorDC;
typedef struct _SchroMotionField SchroMotionField;
typedef struct _SchroMotion SchroMotion;

#ifdef SCHRO_ENABLE_UNSTABLE_API
struct _SchroMotionVector {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
  int16_t x1;
  int16_t y1;
  int16_t x2;
  int16_t y2;
};

struct _SchroMotionVectorDC {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
  uint16_t dc[3];
  uint16_t _padding1;
};

struct _SchroMotionField {
  int x_num_blocks;
  int y_num_blocks;
  SchroMotionVector *motion_vectors;
};

struct _SchroObmcRegion {
  int16_t *weights[3];
  int start_x;
  int start_y;
  int end_x;
  int end_y;
};

struct _SchroObmc {
  SchroObmcRegion regions[9];
  int16_t *region_data;
  int16_t *horiz_ramp;
  int16_t *vert_ramp;
  int x_ramp;
  int y_ramp;
  int x_len;
  int y_len;
  int x_sep;
  int y_sep;
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

  int ref_weight_precision;
  int ref1_weight;
  int ref2_weight;
  int mv_precision;
  int xoffset;
  int yoffset;
  int xbsep;
  int ybsep;
  int xblen;
  int yblen;
};

#define SCHRO_MOTION_GET_BLOCK(motion,x,y) \
  ((motion)->motion_vectors+(y)*(motion)->params->x_num_blocks + (x))
#define SCHRO_MOTION_GET_DC_BLOCK(motion,x,y) \
  ((SchroMotionVectorDC *)SCHRO_MOTION_GET_BLOCK(motion,x,y))

SchroMotion * schro_motion_new (SchroParams *params,
    SchroUpsampledFrame *ref1, SchroUpsampledFrame *ref2);
void schro_motion_free (SchroMotion *motion);

int schro_motion_verify (SchroMotion *mf);
void schro_motion_render_ref (SchroMotion *motion, SchroFrame *dest);
void schro_motion_render (SchroMotion *motion, SchroFrame *dest);

void schro_motion_vector_prediction (SchroMotion *motion,
    int x, int y, int *pred_x, int *pred_y, int mode);
int schro_motion_split_prediction (SchroMotion *motion, int x, int y);
int schro_motion_get_mode_prediction (SchroMotion *motion, int x, int y);
void schro_motion_dc_prediction (SchroMotion *motion,
    int x, int y, int *pred);
int schro_motion_get_global_prediction (SchroMotion *motion,
    int x, int y);

void schro_obmc_init (SchroObmc *obmc, int x_len, int y_len,
    int x_sep, int y_sep, int ref1_weight, int ref2_weight, int ref_shift);
void schro_obmc_cleanup (SchroObmc *obmc);


#endif

SCHRO_END_DECLS

#endif

