
#ifndef __SCHRO_MOTION_H__
#define __SCHRO_MOTION_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/schroparams.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroMotionVector SchroMotionVector;
typedef struct _SchroMotionVectorDC SchroMotionVectorDC;
typedef struct _SchroMotionField SchroMotionField;
typedef struct _SchroMotion SchroMotion;
typedef struct _SchroMotionScan SchroMotionScan;

#ifdef SCHRO_ENABLE_UNSTABLE_API
struct _SchroMotionVector {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
  union {
    struct {
      int16_t dx[2];
      int16_t dy[2];
    } vec;
    struct {
      int16_t dc[3];
    } dc;
  } u;
};

struct _SchroMotionField {
  int x_num_blocks;
  int y_num_blocks;
  SchroMotionVector *motion_vectors;
};

struct _SchroMotion {
  SchroUpsampledFrame *src1;
  SchroUpsampledFrame *src2;
  SchroMotionVector *motion_vectors;
  SchroParams *params;

  uint8_t *tmpdata;

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

  SchroFrameData block;
  SchroFrameData alloc_block;
  SchroFrameData obmc_weight;

  SchroFrameData alloc_block_ref[2];
  SchroFrameData block_ref[2];

  int weight_x[SCHRO_LIMIT_BLOCK_SIZE];
  int weight_y[SCHRO_LIMIT_BLOCK_SIZE];
  int width;
  int height;
  int max_fast_x;
  int max_fast_y;
};

#define SCHRO_MOTION_GET_BLOCK(motion,x,y) \
  ((motion)->motion_vectors+(y)*(motion)->params->x_num_blocks + (x))

SchroMotion * schro_motion_new (SchroParams *params,
    SchroUpsampledFrame *ref1, SchroUpsampledFrame *ref2);
void schro_motion_free (SchroMotion *motion);

int schro_motion_verify (SchroMotion *mf);
void schro_motion_render_ref (SchroMotion *motion, SchroFrame *dest);
void schro_motion_render (SchroMotion *motion, SchroFrame *dest);
void schro_motion_init_obmc_weight (SchroMotion *motion);

void schro_motion_render_fast (SchroMotion *motion, SchroFrame *dest);
int schro_motion_render_fast_allowed (SchroMotion *motion);

void schro_motion_vector_prediction (SchroMotion *motion,
    int x, int y, int *pred_x, int *pred_y, int mode);
int schro_motion_split_prediction (SchroMotion *motion, int x, int y);
int schro_motion_get_mode_prediction (SchroMotion *motion, int x, int y);
void schro_motion_dc_prediction (SchroMotion *motion,
    int x, int y, int *pred);
int schro_motion_get_global_prediction (SchroMotion *motion,
    int x, int y);

int schro_motion_vector_is_equal (SchroMotionVector *mv1, SchroMotionVector *mv2);


#endif

SCHRO_END_DECLS

#endif

