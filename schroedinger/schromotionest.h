
#ifndef __SCHRO_MOTIONEST_H__
#define __SCHRO_MOTIONEST_H__

#include <schroedinger/schroencoder.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroMotionEst SchroMotionEst;

#ifdef SCHRO_ENABLE_UNSTABLE_API

struct _SchroMotionEst {
  SchroEncoderFrame *encoder_frame;
  SchroParams *params;

  SchroUpsampledFrame *src0;
  SchroFrame *downsampled_src0[5];
  SchroUpsampledFrame *src1;
  SchroFrame *downsampled_src1[5];

  SchroMotionVector *motion_vectors;

  SchroMotionField *downsampled_mf[2][5];


};

SchroMotionEst *schro_motionest_new (SchroEncoderFrame *frame);
void schro_motionest_free (SchroMotionEst *me);



void schro_encoder_motion_predict (SchroEncoderFrame *frame);

void schro_encoder_global_estimation (SchroMotionEst *me);

SchroMotionField * schro_motion_field_new (int x_num_blocks, int y_num_blocks);
void schro_motion_field_free (SchroMotionField *field);
void schro_motion_field_scan (SchroMotionField *field, SchroParams *params, SchroFrame *frame, SchroFrame *ref, int dist);
void schro_motion_field_inherit (SchroMotionField *field, SchroMotionField *parent);
void schro_motion_field_copy (SchroMotionField *field, SchroMotionField *parent);
void schro_motion_field_global_estimation (SchroMotionField *mf,
    SchroGlobalMotion *gm, int mv_precision);

int schro_frame_get_metric (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2);
void schro_motion_field_lshift (SchroMotionField *mf, int n);

#endif

SCHRO_END_DECLS

#endif

