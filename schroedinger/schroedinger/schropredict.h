
#ifndef __SCHRO_PREDICT_H__
#define __SCHRO_PREDICT_H__

#include <schroedinger/schroencoder.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

#define SCHRO_PREDICTION_LIST_LENGTH 10

#define SCHRO_PREDICTION_METRIC_INVALID (-1)

struct _SchroPredictionVector {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int common : 1;
  uint8_t dc[3];
  int16_t dx;
  int16_t dy;
  int metric;
  int cost;
};

struct _SchroPredictionList {
  SchroPredictionVector vectors[SCHRO_PREDICTION_LIST_LENGTH];
};



void schro_encoder_motion_predict (SchroEncoderFrame *frame);

void schro_prediction_list_init (SchroPredictionList *pred);
void schro_prediction_list_insert (SchroPredictionList *pred,
    SchroPredictionVector *vec);
void schro_prediction_list_scan (SchroPredictionList *list, SchroFrame *frame,
    SchroFrame *ref, int refnum, int x, int y, int dx, int dy, int dist);

void schro_encoder_global_prediction (SchroEncoderFrame *frame);

SchroMotionField * schro_motion_field_new (int x_num_blocks, int y_num_blocks);
void schro_motion_field_free (SchroMotionField *field);
void schro_motion_field_scan (SchroMotionField *field, SchroParams *params, SchroFrame *frame, SchroFrame *ref, int dist);
void schro_motion_field_inherit (SchroMotionField *field, SchroMotionField *parent);
void schro_motion_field_copy (SchroMotionField *field, SchroMotionField *parent);
void schro_motion_field_global_prediction (SchroMotionField *mf,
    SchroGlobalMotion *gm, int mv_precision);
void schro_motion_field_calculate_stats (SchroMotionField *mf, SchroEncoderFrame *frame);

#endif

SCHRO_END_DECLS

#endif

