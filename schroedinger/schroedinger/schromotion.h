
#ifndef __SCHRO_MOTION_H__
#define __SCHRO_MOTION_H__

#include <schroedinger/schrobuffer.h>

typedef struct _SchroObmc SchroObmc;
typedef struct _SchroObmcRegion SchroObmcRegion;

struct _SchroObmcRegion {
  int16_t *weights;
  int start_x;
  int start_y;
  int end_x;
  int end_y;
};

struct _SchroObmc {
  SchroObmcRegion regions[9];
  int stride;
  int max_weight;
  int x_ramp;
  int y_ramp;
  int x_len;
  int y_len;
  int x_sep;
  int y_sep;
};

void schro_frame_copy_with_motion (SchroFrame *dest, SchroFrame *src1,
    SchroFrame *src2, SchroMotionVector *motion_vectors, SchroParams *params);
void schro_motion_dc_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred);
void schro_motion_vector_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y, int *pred_x, int *pred_y, int mode);
int schro_motion_split_prediction (SchroMotionVector *motion_vectors,
    SchroParams *params, int x, int y);

void schro_obmc_init (SchroObmc *obmc, int x_len, int y_len, int x_sep,
    int y_sep);
void schro_obmc_cleanup (SchroObmc *obmc);


#endif

