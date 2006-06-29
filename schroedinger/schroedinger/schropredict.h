
#ifndef __SCHRO_PREDICT_H__
#define __SCHRO_PREDICT_H__

#include <schroedinger/schroencoder.h>

#define SCHRO_PREDICTION_LIST_LENGTH 10

#define SCHRO_PREDICTION_METRIC_INVALID (-1)

typedef struct _SchroPredictionList SchroPredictionList;
typedef struct _SchroPredictionVector SchroPredictionVector;

struct _SchroPredictionVector {
  int dx;
  int dy;
  int metric;
};

struct _SchroPredictionList {
  int n_vectors;
  SchroPredictionVector vectors[SCHRO_PREDICTION_LIST_LENGTH];
};



void schro_encoder_motion_predict (SchroEncoder *encoder);

void schro_prediction_list_init (SchroPredictionList *pred);
void schro_prediction_list_insert (SchroPredictionList *pred,
    SchroPredictionVector *vec);
void schro_prediction_list_scan (SchroPredictionList *list, SchroFrame *frame,
    SchroFrame *ref, int x, int y, int sx, int sy, int sw, int sh);

#endif

