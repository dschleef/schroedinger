
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static void predict_dc (SchroMotionVector *mv, SchroFrame *frame,
    int x, int y, int w, int h);

void schro_encoder_hierarchical_prediction (SchroEncoder *encoder);


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
schro_encoder_motion_predict (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i;
  int j;
  SchroFrame *ref_frame;
  SchroFrame *frame;
  int sum_pred_x;
  int sum_pred_y;
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
  double sum_x, sum_y;

  SCHRO_ASSERT(params->x_num_mb != 0);
  SCHRO_ASSERT(params->y_num_mb != 0);

  if (encoder->motion_vectors == NULL) {
    encoder->motion_vectors = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
  if (encoder->motion_vectors_dc == NULL) {
    encoder->motion_vectors_dc = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }

  ref_frame = encoder->ref_frame0;
  if (!ref_frame) {
    SCHRO_ERROR("no reference frame");
  }
  frame = encoder->encode_frame;

  SCHRO_ASSERT(params->num_refs > 0);

  schro_encoder_hierarchical_prediction (encoder);

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      int w,h;

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;
      
      w = CLAMP(encoder->video_format.width - x, 0, params->xbsep_luma);
      h = CLAMP(encoder->video_format.height - y, 0, params->ybsep_luma);

      predict_dc (&encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i],
          frame, x, y, w, h);
    }
  }

  encoder->stats_metric = 0;
  encoder->stats_dc_blocks = 0;
  encoder->stats_none_blocks = 0;
  encoder->stats_scan_blocks = 0;
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int cost_dc;
      int cost_scan;
      int pred[3];
      int pred_x, pred_y;
      SchroMotionVector *mv;

      schro_motion_dc_prediction (encoder->motion_vectors,
          params, i, j, pred);
      mv = &encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i];
      cost_dc = cost(mv->dc[0] - pred[0]) + cost(mv->dc[1] - pred[1]) +
        cost(mv->dc[2] - pred[2]);
      cost_dc += encoder->metric_to_cost * mv->metric;
#if 0
mv->dc[0] = 128;
mv->dc[1] = 128;
mv->dc[2] = 128;
#endif
      /* FIXME the metric underestimates the cost of DC blocks, so we
       * bump it up a bit here */
      //cost_dc += 64;
      cost_dc += 10;

      schro_motion_vector_prediction (encoder->motion_vectors,
          params, i, j, &pred_x, &pred_y);
      mv = &encoder->motion_vectors[j*(4*params->x_num_mb) + i];
      cost_scan = cost(mv->x - pred_x) + cost(mv->y - pred_y);
      cost_scan += encoder->metric_to_cost * mv->metric;

      if (cost_dc < cost_scan) {
        memcpy (&encoder->motion_vectors[j*(4*params->x_num_mb) + i],
            &encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i],
            sizeof(SchroMotionVector));
        encoder->stats_dc_blocks++;
      } else {
        encoder->stats_scan_blocks++;
      }
      encoder->stats_metric += 
          encoder->motion_vectors[j*(4*params->x_num_mb) + i].metric;

      encoder->motion_vectors[j*(4*params->x_num_mb) + i].split = 2;
    }
  }

  sum_pred_x = 0;
  sum_pred_y = 0;

  pan_x = ((double)sum_pred_x)/(16*params->x_num_mb*params->y_num_mb);
  pan_y = ((double)sum_pred_y)/(16*params->x_num_mb*params->y_num_mb);

  mag_x = 0;
  mag_y = 0;
  skew_x = 0;
  skew_y = 0;
  sum_x = 0;
  sum_y = 0;
  for(j=0;j<4*params->y_num_mb;j++) {
    for(i=0;i<4*params->x_num_mb;i++) {
      double x;
      double y;

      x = i*params->xbsep_luma - (2*params->x_num_mb - 0.5);
      y = j*params->ybsep_luma - (2*params->y_num_mb - 0.5);

      mag_x += encoder->motion_vectors[j*(4*params->x_num_mb) + i].x * x;
      mag_y += encoder->motion_vectors[j*(4*params->x_num_mb) + i].y * y;

      skew_x += encoder->motion_vectors[j*(4*params->x_num_mb) + i].x * y;
      skew_y += encoder->motion_vectors[j*(4*params->x_num_mb) + i].y * x;

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

  encoder->pan_x = pan_x;
  encoder->pan_y = pan_y;
  encoder->mag_x = mag_x;
  encoder->mag_y = mag_y;
  encoder->skew_x = skew_x;
  encoder->skew_y = skew_y;

  SCHRO_DEBUG("pan %g %g mag %g %g skew %g %g",
      pan_x, pan_y, mag_x, mag_y, skew_x, skew_y);

}

static void
predict_dc (SchroMotionVector *mv, SchroFrame *frame, int x, int y,
    int width, int height)
{
  int stride;
  int sum;

  if (height == 0 || width == 0) {
    mv->pred_mode = 0;
    mv->metric = 1000000;
    return;
  }

  SCHRO_ASSERT(x + width <= frame->components[0].width);
  SCHRO_ASSERT(y + height <= frame->components[0].height);

  stride = frame->components[0].stride;
  sum = schro_metric_sum_u8 (frame->components[0].data + x + y * stride,
      stride, width, height);
  mv->dc[0] = (sum+height*width/2)/(height*width);
  mv->metric = schro_metric_haar_const (frame->components[0].data + x + y*stride,
      stride, mv->dc[0], width, height);

  width/=2;
  height/=2;
  x/=2;
  y/=2;

  stride = frame->components[1].stride;
  sum = schro_metric_sum_u8 (frame->components[1].data + x + y * stride,
      stride, width, height);
  mv->dc[1] = (sum+height*width/2)/(height*width);

  stride = frame->components[2].stride;
  sum = schro_metric_sum_u8 (frame->components[2].data + x + y * stride,
      stride, width, height);
  mv->dc[2] = (sum+height*width/2)/(height*width);

  mv->pred_mode = 0;
}


/* Prediction List */

void schro_prediction_list_init (SchroPredictionList *pred)
{
  pred->n_vectors = 0;
}

void schro_prediction_list_insert (SchroPredictionList *pred,
    SchroPredictionVector *vec)
{
  int i;

  if (pred->n_vectors == 0) {
    pred->vectors[0] = *vec;
    pred->n_vectors = 1;
    return;
  }
  if (pred->n_vectors == SCHRO_PREDICTION_LIST_LENGTH &&
      vec->metric > pred->vectors[SCHRO_PREDICTION_LIST_LENGTH-1].metric) {
    return;
  }
  if (pred->n_vectors < SCHRO_PREDICTION_LIST_LENGTH) pred->n_vectors++;
  for(i=pred->n_vectors-2;i>=0;i--) {
    if (vec->metric < pred->vectors[i].metric) {
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
    SchroFrame *ref, int refnum, int x, int y, int sx, int sy, int sw, int sh)
{
  int i,j;
  SchroPredictionVector vec;
  int xmin;
  int xmax;
  int ymin;
  int ymax;

  xmin = MAX(0, sx);
  ymin = MAX(0, sy);
  xmax = MIN(frame->components[0].width - 8, sx + sw);
  ymax = MIN(frame->components[0].height - 8, sy + sh);

  for(j=ymin;j<=ymax;j++){
    for(i=xmin;i<=xmax;i++){
      vec.ref = refnum;
      vec.dx = i - x;
      vec.dy = j - y;
      vec.metric = schro_metric_haar (
          frame->components[0].data + x + y*frame->components[0].stride,
          frame->components[0].stride,
          ref->components[0].data + i + j*ref->components[0].stride,
          ref->components[0].stride, 8, 8);
      schro_prediction_list_insert (list, &vec);
    }
  }
}


void
schro_encoder_hierarchical_prediction (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i;
  int j;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref0;
  SchroFrame *downsampled_ref1 = NULL;
  SchroFrame *downsampled_frame;
  SchroPredictionList *pred_lists;
  SchroPredictionList *prev_pred_lists;
  int shift;

  prev_pred_lists = NULL;

  for(shift=3;shift>=0;shift--) {
    /* FIXME downsampled size is wrong */
    downsampled_ref0 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        encoder->video_format.width>>shift, encoder->video_format.height>>shift, 2, 2);
    schro_frame_downsample (downsampled_ref0, encoder->ref_frame0, shift);

    if (params->num_refs == 2) {
      downsampled_ref1 = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
          encoder->video_format.width>>shift, encoder->video_format.height>>shift, 2, 2);
      schro_frame_downsample (downsampled_ref1, encoder->ref_frame1, shift);
    }

    downsampled_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        encoder->video_format.width>>shift, encoder->video_format.height>>shift, 2, 2);
    schro_frame_downsample (downsampled_frame, encoder->encode_frame, shift);

    x_blocks = params->x_num_blocks>>shift;
    y_blocks = params->y_num_blocks>>shift;

    pred_lists = malloc (x_blocks*y_blocks*sizeof(SchroPredictionList));
    for(j=0;j<y_blocks;j++){
      for(i=0;i<x_blocks;i++){
        schro_prediction_list_init (pred_lists + j*x_blocks + i);
      }
    }

    if (prev_pred_lists != NULL) {
      for(j=0;j<y_blocks;j++){
        for(i=0;i<x_blocks;i++){
          int parent_j = j>>1;
          int parent_i = i>>1;
          int k;
          SchroPredictionList *list;

          list = prev_pred_lists + parent_j*(x_blocks>>1) + parent_i;
          if (list->n_vectors == 0) {
            /* probably on bottom or side, so we'll pull from neighbor
             * parent */
            if (j >= y_blocks + 2) {
              parent_j--;
            }
            if (i >= x_blocks + 2) {
              parent_i--;
            }
            list = prev_pred_lists + parent_j*(x_blocks>>1) + parent_i;
          }
          for(k=0;k<MIN(4,list->n_vectors);k++){
            if (list->vectors[k].ref == 0) {
#define SIZE2 1
              schro_prediction_list_scan (pred_lists + j*x_blocks + i,
                  downsampled_frame, downsampled_ref0, 0, i*8, j*8,
                  i*8 - SIZE2 + list->vectors[k].dx*2,
                  j*8 - SIZE2 + list->vectors[k].dy*2, 2*SIZE2+1, 2*SIZE2+1);
            } else {
              schro_prediction_list_scan (pred_lists + j*x_blocks + i,
                  downsampled_frame, downsampled_ref1, 1, i*8, j*8,
                  i*8 - SIZE2 + list->vectors[k].dx*2,
                  j*8 - SIZE2 + list->vectors[k].dy*2, 2*SIZE2+1, 2*SIZE2+1);
            }
          }
        }
      }
    }
    for(j=0;j<y_blocks;j++){
      for(i=0;i<x_blocks;i++){
        SchroPredictionList *list = pred_lists + j*x_blocks + i;
        if (list->n_vectors == 0) {
#define SIZE 8
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref0, 0, i*8, j*8,
              i*8 - SIZE, j*8 - SIZE, 2*SIZE + 1, 2*SIZE + 1);
          if (params->num_refs == 2) {
            schro_prediction_list_scan (pred_lists + j*x_blocks + i,
                downsampled_frame, downsampled_ref1, 1, i*8, j*8,
                i*8 - SIZE, j*8 - SIZE, 2*SIZE + 1, 2*SIZE + 1);
          }
        }
      }
    }
#if 0
    /* This is expensive */
    for(j=0;j<y_blocks;j++){
      for(i=0;i<x_blocks;i++){
        if(j>0) {
          SchroPredictionList *neighbor_list = pred_lists + (j-1)*x_blocks + i;
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref, i*8, j*8,
              i*8 - 2 + neighbor_list->vectors[0].dx,
              j*8 - 2 + neighbor_list->vectors[0].dy, 5, 5);
        }
        if(i>0) {
          SchroPredictionList *neighbor_list = pred_lists + j*x_blocks + i-1;
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref, i*8, j*8,
              i*8 - 2 + neighbor_list->vectors[0].dx,
              j*8 - 2 + neighbor_list->vectors[0].dy, 5, 5);
        }
        if(j+1 < y_blocks) {
          SchroPredictionList *neighbor_list = pred_lists + (j+1)*x_blocks + i;
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref, i*8, j*8,
              i*8 - 2 + neighbor_list->vectors[0].dx,
              j*8 - 2 + neighbor_list->vectors[0].dy, 5, 5);
        }
        if(i+1 < x_blocks) {
          SchroPredictionList *neighbor_list = pred_lists + j*x_blocks + i+1;
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref, i*8, j*8,
              i*8 - 2 + neighbor_list->vectors[0].dx,
              j*8 - 2 + neighbor_list->vectors[0].dy, 5, 5);
        }
      }
    }
#endif

    schro_frame_free (downsampled_ref0);
    if (downsampled_ref1) {
      schro_frame_free (downsampled_ref1);
    }
    schro_frame_free (downsampled_frame);

    if (prev_pred_lists) {
      free(prev_pred_lists);
    }
    prev_pred_lists = pred_lists;
    pred_lists = NULL;
  }

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){ 
      SchroMotionVector *mv;
      SchroPredictionList *list;

      mv = &encoder->motion_vectors[j*(4*params->x_num_mb) + i];
      list = prev_pred_lists + MIN(j,y_blocks-1)*x_blocks + MIN(i,x_blocks-1);

      mv->pred_mode = list->vectors[0].ref + 1;
      mv->using_global = 0;
      mv->split = 2;
      mv->common = 0;
      mv->x = list->vectors[0].dx;
      mv->y = list->vectors[0].dy;
      mv->metric = list->vectors[0].metric;
    }
  }

  free(prev_pred_lists);
}

