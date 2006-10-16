
#ifndef __SCHRO_PREDICT_H__
#define __SCHRO_PREDICT_H__

#include <schroedinger/schroencoder.h>

#define SCHRO_PREDICTION_LIST_LENGTH 10

#define SCHRO_PREDICTION_METRIC_INVALID (-1)

//typedef struct _SchroPredictionList SchroPredictionList;
//typedef struct _SchroPredictionVector SchroPredictionVector;

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



void schro_encoder_motion_predict (SchroEncoder *encoder);

void schro_prediction_list_init (SchroPredictionList *pred);
void schro_prediction_list_insert (SchroPredictionList *pred,
    SchroPredictionVector *vec);
void schro_prediction_list_scan (SchroPredictionList *list, SchroFrame *frame,
    SchroFrame *ref, int refnum, int x, int y, int dx, int dy, int dist);

#endif

