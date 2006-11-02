
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define SCHRO_METRIC_INVALID (1<<24)

void schro_encoder_hierarchical_prediction (SchroEncoderTask *task);
void schro_encoder_dc_prediction (SchroEncoderTask *task);


int cost (int value)
{
  int n;
  if (value == 0) return 1;
  if (value < 0) value = -value;
  value++;
  n = 0;
  while (value) {
    n+=2;
    value>>=1;
  }
  return n;
}

void
schro_encoder_motion_predict (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i;
  int j;
  SchroFrame *frame;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);

  if (task->motion_vectors == NULL) {
    task->motion_vectors = malloc(sizeof(SchroMotionVector)*
        params->x_num_blocks*params->y_num_blocks);
  }

  frame = task->encode_frame;

  SCHRO_ASSERT(params->num_refs > 0);

  schro_encoder_hierarchical_prediction (task);

  schro_encoder_dc_prediction (task);

  task->stats_metric = 0;
  task->stats_dc_blocks = 0;
  task->stats_none_blocks = 0;
  task->stats_scan_blocks = 0;
  for(j=0;j<params->y_num_blocks;j++){
    for(i=0;i<params->x_num_blocks;i++){
      int dc_pred[3];
      int pred1_x, pred1_y;
      int pred2_x, pred2_y;
      SchroMotionVector *mv;
      SchroPredictionVector *vec;
      int k;
      SchroPredictionList *list;
      int best_index = 0;
      int best_cost = 0;
      
      list = task->predict_lists + j*params->x_num_blocks + i;

      schro_motion_dc_prediction (task->motion_vectors,
          params, i, j, dc_pred);
      schro_motion_vector_prediction (task->motion_vectors,
          params, i, j, &pred1_x, &pred1_y, 1);
      schro_motion_vector_prediction (task->motion_vectors,
          params, i, j, &pred2_x, &pred2_y, 2);

      schro_prediction_list_scan (list, task->encode_frame,
          task->ref_frame0->frames[0], 1, i*8, j*8, pred1_x, pred1_y, 0);
      if (params->num_refs == 2) {
        schro_prediction_list_scan (list, task->encode_frame,
            task->ref_frame1->frames[0], 2, i*8, j*8, pred2_x, pred2_y, 0);
      }

      for(k=0;k<SCHRO_PREDICTION_LIST_LENGTH;k++){
        vec = list->vectors + k;

        if (vec->metric == SCHRO_METRIC_INVALID) break;

        switch(vec->pred_mode) {
          case 0:
            vec->cost = cost(vec->dc[0] - dc_pred[0]) +
              cost(vec->dc[1] - dc_pred[1]) + cost(vec->dc[2] - dc_pred[2]);
            vec->cost += 10;
            break;
          case 1:
            vec->cost = cost(vec->dx - pred1_x) + cost(vec->dy - pred1_y);
            break;
          case 2:
            vec->cost = cost(vec->dx - pred2_x) + cost(vec->dy - pred2_y);
            break;
          case 3:
            SCHRO_ASSERT(0);
            break;
        }
        vec->cost += vec->metric * task->metric_to_cost;

        if (k==0 || vec->cost < best_cost) {
          best_cost = vec->cost;
          best_index = k;
        }
      }

      /* FIXME choose based on cost as well */
      vec = list->vectors + best_index;

      mv = &task->motion_vectors[j*params->x_num_blocks + i];
      mv->pred_mode = vec->pred_mode;
      mv->using_global = 0;
      mv->split = 2;
      mv->common = 0;
      if (vec->pred_mode == 0) {
        mv->dc[0] = vec->dc[0];
        mv->dc[1] = vec->dc[1];
        mv->dc[2] = vec->dc[2];
        task->stats_dc_blocks++;
      } else {
        mv->x = vec->dx;
        mv->y = vec->dy;
        task->stats_scan_blocks++;
      }
      task->stats_metric += vec->metric;
    }
  }
}



void
schro_encoder_global_motion_predict (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i;
  int j;
  int sum_pred_x;
  int sum_pred_y;
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
  double sum_x, sum_y;

  sum_pred_x = 0;
  sum_pred_y = 0;

  pan_x = ((double)sum_pred_x)/(params->x_num_blocks*params->y_num_blocks);
  pan_y = ((double)sum_pred_y)/(params->x_num_blocks*params->y_num_blocks);

  mag_x = 0;
  mag_y = 0;
  skew_x = 0;
  skew_y = 0;
  sum_x = 0;
  sum_y = 0;
  for(j=0;j<params->y_num_blocks;j++) {
    for(i=0;i<params->x_num_blocks;i++) {
      double x;
      double y;

      x = i*params->xbsep_luma - (params->x_num_blocks/2 - 0.5);
      y = j*params->ybsep_luma - (params->y_num_blocks/2 - 0.5);

      mag_x += task->motion_vectors[j*params->x_num_blocks + i].x * x;
      mag_y += task->motion_vectors[j*params->x_num_blocks + i].y * y;

      skew_x += task->motion_vectors[j*params->x_num_blocks + i].x * y;
      skew_y += task->motion_vectors[j*params->x_num_blocks + i].y * x;

      sum_x += x * x;
      sum_y += y * y;
    }
  }
  if (sum_x != 0) {
    mag_x = mag_x/sum_x;
    skew_x = skew_x/sum_x;
  } else {
    mag_x = 0;
    skew_x = 0;
  }
  if (sum_y != 0) {
    mag_y = mag_y/sum_y;
    skew_y = skew_y/sum_y;
  } else {
    mag_y = 0;
    skew_y = 0;
  }

#if 0
  task->pan_x = pan_x;
  task->pan_y = pan_y;
  task->mag_x = mag_x;
  task->mag_y = mag_y;
  task->skew_x = skew_x;
  task->skew_y = skew_y;
#endif

  SCHRO_DEBUG("pan %g %g mag %g %g skew %g %g",
      pan_x, pan_y, mag_x, mag_y, skew_x, skew_y);

}


/* Prediction List */

void schro_prediction_list_init (SchroPredictionList *pred)
{
  int i;

  memset(pred,0,sizeof(*pred));
  for(i=0;i<SCHRO_PREDICTION_LIST_LENGTH;i++){
    pred->vectors[i].metric = SCHRO_METRIC_INVALID;
  }
}

void schro_prediction_list_insert (SchroPredictionList *pred,
    SchroPredictionVector *vec)
{
  int i;

  i = SCHRO_PREDICTION_LIST_LENGTH - 1;
  if ((vec->metric>>4) + vec->cost >=
      (pred->vectors[i].metric>>4) + pred->vectors[i].cost) {
    return;
  }

  for (i = SCHRO_PREDICTION_LIST_LENGTH - 2; i>=0; i--) {
    if ((vec->metric>>4) + vec->cost <
        (pred->vectors[i].metric>>4) + pred->vectors[i].cost) {
      pred->vectors[i+1] = pred->vectors[i];
    } else {
      pred->vectors[i+1] = *vec;
      return;
    }
  }
  pred->vectors[0] = *vec;
}

void
schro_prediction_list_scan (SchroPredictionList *list, SchroFrame *frame,
    SchroFrame *ref, int mode, int x, int y, int dx, int dy, int dist)
{
  int i,j;
  SchroPredictionVector vec;
  int xmin;
  int xmax;
  int ymin;
  int ymax;

  SCHRO_ASSERT(mode == 1 || mode == 2);

  xmin = MAX(0, x + dx - dist);
  ymin = MAX(0, y + dy - dist);
  xmax = MIN(frame->components[0].width - 8, x + dx + dist);
  ymax = MIN(frame->components[0].height - 8, y + dy + dist);

  for(j=ymin;j<=ymax;j++){
    for(i=xmin;i<=xmax;i++){
      vec.pred_mode = mode;
      vec.dx = i - x;
      vec.dy = j - y;
      vec.metric = schro_metric_haar (
          frame->components[0].data + x + y*frame->components[0].stride,
          frame->components[0].stride,
          ref->components[0].data + i + j*ref->components[0].stride,
          ref->components[0].stride, 8, 8);
      vec.cost = cost(vec.dx) + cost(vec.dy);
      schro_prediction_list_insert (list, &vec);
    }
  }
}


void
schro_encoder_hierarchical_prediction (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i;
  int j;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref0;
  SchroFrame *downsampled_ref1 = NULL;
  SchroFrame *downsampled[4];
  SchroFrame *downsampled_frame;
  SchroFrame *frame = task->encode_frame;
  SchroPredictionList *pred_lists;
  int shift;

  downsampled[0] = task->encode_frame;
  for(i=1;i<4;i++){
    downsampled[i] = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8,
        ROUND_UP_SHIFT(frame->components[0].width, i),
        ROUND_UP_SHIFT(frame->components[0].height, i),
        ROUND_UP_SHIFT(frame->components[0].width, i+1),
        ROUND_UP_SHIFT(frame->components[0].height, i+1));
    schro_frame_downsample(downsampled[i], downsampled[i-1], 1);
  }

  if (task->predict_lists == NULL) {
    task->predict_lists = malloc (params->x_num_blocks*params->y_num_blocks*
        sizeof(SchroPredictionList));
  }
  pred_lists = task->predict_lists;

  for(shift=3;shift>=0;shift--) {
    int skip = 1<<shift;

    downsampled_ref0 = task->ref_frame0->frames[shift];
    if (params->num_refs == 2) {
      downsampled_ref1 = task->ref_frame1->frames[shift];
    }
    downsampled_frame = downsampled[shift];

    x_blocks = ROUND_UP_SHIFT(params->x_num_blocks,shift);
    y_blocks = ROUND_UP_SHIFT(params->y_num_blocks,shift);

    if (shift == 3) {
      for(j=0;j<params->y_num_blocks;j+=(1<<shift)){
        for(i=0;i<params->x_num_blocks;i+=(1<<shift)){
          SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;
          int x = i>>shift;
          int y = j>>shift;

          schro_prediction_list_init (list);

          schro_prediction_list_scan (list,
              downsampled_frame, downsampled_ref0, 1, x*8, y*8, 0, 0, 12);
          if (params->num_refs == 2) {
            schro_prediction_list_scan (list,
                downsampled_frame, downsampled_ref1, 2, x*8, y*8, 0, 0, 12);
          }
        }
      }

    } else {
      /* copy from parent */
      for(j=0;j<params->y_num_blocks;j+=skip*2){
        for(i=0;i<params->x_num_blocks;i+=skip*2){
          SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;

          if (j+skip < params->y_num_blocks) {
            memcpy(list + skip*params->x_num_blocks, list,
                sizeof(SchroPredictionList));
          }
          if (i+skip < params->x_num_blocks) {
            memcpy(list + skip, list,
                sizeof(SchroPredictionList));
            if (j+skip < params->y_num_blocks) {
              memcpy(list + skip*(params->x_num_blocks + 1), list,
                  sizeof(SchroPredictionList));
            }
          }
        }
      }

      /* predict from parent */
      for(j=0;j<params->y_num_blocks;j+=skip){
        for(i=0;i<params->x_num_blocks;i+=skip){
          SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;
          int x = i>>shift;
          int y = j>>shift;
          int k;
          SchroPredictionList parent;

          memcpy (&parent, list, sizeof(SchroPredictionList));
          schro_prediction_list_init (list);

          for(k=0;k<4;k++){
            if (parent.vectors[k].metric == SCHRO_METRIC_INVALID) break;

            if (parent.vectors[k].pred_mode == 1) {
              schro_prediction_list_scan (list,
                  downsampled_frame, downsampled_ref0, 1, x*8, y*8,
                  parent.vectors[k].dx*2, parent.vectors[k].dy*2, 2);
            } else {
              schro_prediction_list_scan (list,
                  downsampled_frame, downsampled_ref1, 2, x*8, y*8,
                  parent.vectors[k].dx*2, parent.vectors[k].dy*2, 2);
            }
          }
        }
      }

#if 0
      /* predict from neighbor */
      for(j=0;j<params->y_num_blocks;j+=skip){
        for(i=0;i<params->x_num_blocks;i+=skip){
          SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;
          int x = i>>shift;
          int y = j>>shift;
          int k;

          for(k=0;k<4;k++){
            const int di[4] = { -1, 0, 0, 1 };
            const int dj[4] = { 0, -1, 1, 0 };
            int si = i + di[k] * skip;
            int sj = j + dj[k] * skip;
            SchroPredictionList *neighbor;

            if (si >= 0 && si < params->x_num_blocks &&
                sj >= 0 && sj < params->y_num_blocks) {
              neighbor = pred_lists + sj*params->x_num_blocks + si;
              schro_prediction_list_scan (list, downsampled_frame,
                  (neighbor->vectors[0].pred_mode == 1) ? downsampled_ref0 : downsampled_ref1,
                  neighbor->vectors[0].pred_mode, x*8, y*8,
                  neighbor->vectors[0].dx, neighbor->vectors[0].dy, 2);
            }
          }
        }
      }
#endif

      /* predict from average of neighbors? */

#if 1
      /* predict zero vector */
      for(j=0;j<params->y_num_blocks;j+=skip){
        for(i=0;i<params->x_num_blocks;i+=skip){
          SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;
          int x = i>>shift;
          int y = j>>shift;

          schro_prediction_list_scan (list, downsampled_frame,
              downsampled_ref0, 1, x*8, y*8, 0, 0, 0);
          if (params->num_refs == 2) {
            schro_prediction_list_scan (list, downsampled_frame,
                downsampled_ref1, 2, x*8, y*8, 0, 0, 0);
          }
        }
      }
#endif

      /* predict from elsewhere? */
    }
  }

  schro_frame_free(downsampled[1]);
  schro_frame_free(downsampled[2]);
  schro_frame_free(downsampled[3]);
}

static int
schro_block_average (uint8_t *dest, SchroFrameComponent *comp,
    int x, int y, int w, int h)
{
  int xmax = MIN(x + w, comp->width);
  int ymax = MIN(y + h, comp->height);
  int i,j;
  int n = 0;
  int sum = 0;
  int ave;

  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      sum += SCHRO_GET(comp->data, j*comp->stride + i, uint8_t);
    }
    n += xmax - x;
  }

  if (n == 0) {
    return SCHRO_METRIC_INVALID;
  }

  ave = (sum + n/2)/n;

  sum = 0;
  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      sum += abs(ave - SCHRO_GET(comp->data, j*comp->stride + i, uint8_t));
    }
  }

  *dest = ave;
  return sum;
}

void
schro_encoder_dc_prediction (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  int i;
  int j;
  SchroPredictionVector vec;
  SchroFrame *frame = task->encode_frame;
  SchroPredictionList *pred_lists = task->predict_lists;

  vec.pred_mode = 0;

  for(j=0;j<params->y_num_blocks;j++){
    for(i=0;i<params->x_num_blocks;i++){
      SchroPredictionList *list = pred_lists + j*params->x_num_blocks + i;

      vec.metric = schro_block_average (vec.dc + 0, frame->components + 0,
          i*8, j*8, 8, 8);
      vec.metric += schro_block_average (vec.dc + 1, frame->components + 1,
          i*4, j*4, 4, 4);
      vec.metric += schro_block_average (vec.dc + 2, frame->components + 2,
          i*4, j*4, 4, 4);
      vec.cost = cost(vec.dc[0] - 128) + cost(vec.dc[1] - 128) +
        cost(vec.dc[2] - 128);

      schro_prediction_list_insert (list, &vec);
    }
  }
}

