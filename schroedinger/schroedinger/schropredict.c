
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define CLAMP(x,a,b) ((x)<(a) ? (a) : ((x)>(b) ? (b) : (x)))

static void predict_dc (SchroMotionVector *mv, SchroFrame *frame,
    int x, int y, int w, int h);
#if 0
static void predict_motion (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h);
static void predict_motion_none (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h);
static void predict_motion_search (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h);
#endif

#if 0
static void schro_encoder_rough_global_prediction (SchroEncoder *encoder);
static void schro_block_search (SchroMotionVector *est, SchroFrame *frame,
    SchroFrame *ref, int x, int y, int distance);
#endif

void schro_encoder_rough_global_prediction_2 (SchroEncoder *encoder);


#if 0
static void
schro_encoder_choose_split (SchroEncoder *encoder, int x, int y)
{
  SchroParams *params = &encoder->params;
  SchroMotionVector *mv0;
  SchroMotionVector *mv;
  int i,j;
 
  mv0 = &encoder->motion_vectors[y*(4*params->x_num_mb) + x];

  if (mv0->pred_mode == 0) {
    for(j=0;j<4;j++){
      for(i=0;i<4;i++){
        mv = &encoder->motion_vectors[(y+j)*(4*params->x_num_mb) + (x+j)];
        if (mv0->dc[0] != mv->dc[0] ||
            mv0->dc[1] != mv->dc[1] ||
            mv0->dc[2] != mv->dc[2]) {
          mv->split = 2;
          return;
        }
      }
    }
    mv->split = 0;
  } else {
    for(j=0;j<4;j++){
      for(i=0;i<4;i++){
        mv = &encoder->motion_vectors[(y+j)*(4*params->x_num_mb) + (x+j)];
        if (mv0->pred_mode != mv->pred_mode ||
            mv0->x != mv->x ||
            mv0->y != mv->y) {
          mv->split = 2;
          return;
        }
      }
    }
    mv->split = 0;
  }
}
#endif

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
#if 0
  if (encoder->motion_vectors_none == NULL) {
    encoder->motion_vectors_none = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
  if (encoder->motion_vectors_scan == NULL) {
    encoder->motion_vectors_scan = malloc(sizeof(SchroMotionVector)*
        params->x_num_mb*params->y_num_mb*16);
  }
#endif

  ref_frame = encoder->ref_frame0;
  if (!ref_frame) {
    SCHRO_ERROR("no reference frame");
  }
  frame = encoder->encode_frame;

#if 0
  schro_encoder_rough_global_prediction (encoder);
#endif
  schro_encoder_rough_global_prediction_2 (encoder);

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      int w,h;

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;
      
      w = CLAMP(params->width - x, 0, params->xbsep_luma);
      h = CLAMP(params->height - y, 0, params->ybsep_luma);

      predict_dc (&encoder->motion_vectors_dc[j*(4*params->x_num_mb) + i],
          frame, x, y, w, h);

#if 0
      predict_motion_none (&encoder->motion_vectors_none[j*(4*params->x_num_mb) + i],
          frame, ref_frame, x, y, w, h);

      predict_motion_search (&encoder->motion_vectors_scan[j*(4*params->x_num_mb) + i],
          frame, ref_frame, x, y, w, h);

      (void)&predict_motion;
#endif
    }
  }

  encoder->stats_metric = 0;
  encoder->stats_dc_blocks = 0;
  encoder->stats_none_blocks = 0;
  encoder->stats_scan_blocks = 0;
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int cost_dc;
#if 0
      int cost_none;
#endif
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
      cost_dc += 64;

      schro_motion_vector_prediction (encoder->motion_vectors,
          params, i, j, &pred_x, &pred_y);
#if 0
      mv = &encoder->motion_vectors_none[j*(4*params->x_num_mb) + i];
      cost_none = cost(mv->x - pred_x) + cost(mv->y - pred_y);
      cost_none += encoder->metric_to_cost * mv->metric;

      mv = &encoder->motion_vectors_scan[j*(4*params->x_num_mb) + i];
      cost_scan = cost(mv->x - pred_x) + cost(mv->y - pred_y);
      cost_scan += encoder->metric_to_cost * mv->metric;
#endif
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
#if 0
  for(j=0;j<4*params->y_num_mb;j+=4) {
    for(i=0;i<4*params->x_num_mb;i+=4) {
      schro_encoder_choose_split (encoder, i, j);
    }
  }
#endif

  sum_pred_x = 0;
  sum_pred_y = 0;
#if 0
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      SchroMotionVector *mv =
        &encoder->motion_vectors[j*(4*params->x_num_mb) + i];

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      predict_dc (mv, frame, x, y, params->xbsep_luma, params->ybsep_luma);

      predict_motion (mv, frame, ref_frame, x, y,
          params->xbsep_luma, params->ybsep_luma);

      if (mv->pred_mode != 0) {
        sum_pred_x += mv->x;
        sum_pred_y += mv->y;
      }
    }
  }
#endif

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

#if 0
static int
calculate_metric (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  return metric;
}
#endif

#if 0
static int
calculate_metric2 (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2, int width, int height)
{
  int i;
  int j;
  int metric = 0;
  uint8_t *a;
  int a_stride;
  uint8_t *b;
  int b_stride;

  a_stride = frame1->components[0].stride;
  a = frame1->components[0].data + x1 + y1 * a_stride;
  b_stride = frame2->components[0].stride;
  b = frame2->components[0].data + x2 + y2 * b_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  return metric;
}
#endif

static int
calculate_haar_metric (int16_t *diff)
{
  int i,j;
  int a,b;
  int metric;

if(1){
  for(j=0;j<8;j++){
    for(i=0;i<8;i+=2){
      a = (diff[8*j+i] + diff[8*j+i+1])/2;
      b = diff[8*j+i+1] - a;
      diff[8*j+i] = a;
      diff[8*j+i+1] = b;
    }
  }
  for(j=0;j<8;j++){
    for(i=0;i<8;i+=2){
      a = (diff[j+8*i] + diff[j+8*(i+1)])/2;
      b = diff[j+8*(i+1)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+1)] = b;
    }
  }
  for(j=0;j<8;j+=2){
    for(i=0;i<8;i+=4){
      a = (diff[8*j+i] + diff[8*j+i+2])/2;
      b = diff[8*j+i+2] - a;
      diff[8*j+i] = a;
      diff[8*j+i+2] = b;
    }
  }
  for(j=0;j<8;j+=2){
    for(i=0;i<8;i+=4){
      a = (diff[j+8*i] + diff[j+8*(i+2)])/2;
      b = diff[j+8*(i+2)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+2)] = b;
    }
  }
  for(j=0;j<8;j+=4){
    for(i=0;i<8;i+=8){
      a = (diff[8*j+i] + diff[8*j+i+4])/2;
      b = diff[8*j+i+4] - a;
      diff[8*j+i] = a;
      diff[8*j+i+4] = b;
    }
  }
  for(j=0;j<8;j+=4){
    for(i=0;i<8;i+=8){
      a = (diff[j+8*i] + diff[j+8*(i+4)])/2;
      b = diff[j+8*(i+4)] - a;
      diff[j+8*i] = a;
      diff[j+8*(i+4)] = b;
    }
  }
}
  metric = 0;
  for(j=0;j<8;j++){
    for(i=0;i<8;i++){
      metric+=abs(diff[j*8+i]);
    }
  }

  return metric;
}

static int
calculate_metric3 (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2, int width, int height)
{
  int16_t diff[64];
  int i;
  int j;
  uint8_t *a;
  int a_stride;
  uint8_t *b;
  int b_stride;

  a_stride = frame1->components[0].stride;
  a = frame1->components[0].data + x1 + y1 * a_stride;
  b_stride = frame2->components[0].stride;
  b = frame2->components[0].data + x2 + y2 * b_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      diff[j*8+i] = a[j*a_stride + i] - b[j*b_stride + i];
    }
  }

  return calculate_haar_metric(diff);
}

static int
calculate_dc_metric (SchroFrame *frame1, int x1, int y1, int dc_value,
    int width, int height)
{
  int16_t diff[64];
  int i;
  int j;
  uint8_t *a;
  int a_stride;

  a_stride = frame1->components[0].stride;
  a = frame1->components[0].data + x1 + y1 * a_stride;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      diff[j*8+i] = a[j*a_stride + i] - dc_value;
    }
  }

  return calculate_haar_metric(diff);
}

#if 0
static void
predict_motion_search (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int dx, dy;
  //uint8_t *data = frame->components[0].data;
  //int stride = frame->components[0].stride;
  //uint8_t *ref_data = reference_frame->components[0].data;
  //int ref_stride = reference_frame->components[0].stride;
  int metric;
  int min_metric;
  int step_size;

#if 0
  min_metric = calculate_metric (data + y * stride + x, stride,
      ref_data + y * ref_stride + x, ref_stride, w, h);
  *pred_x = 0;
  *pred_y = 0;

  printf("mp %d %d metric %d\n", x, y, min_metric);
#endif

  dx = 0;
  dy = 0;
  step_size = 4;
  while (step_size > 0) {
    static const int hx[5] = { 0, 0, -1, 0, 1 };
    static const int hy[5] = { 0, -1, 0, 1, 0 };
    int px, py;
    int min_index;
    int i;

    min_index = 0;
    min_metric = calculate_metric2 (frame, x, y, reference_frame, x+dx, y+dy,
        w, h);
    for(i=1;i<5;i++){
      px = x + dx + hx[i] * step_size;
      py = y + dy + hy[i] * step_size;
      if (px < 0 || py < 0 || 
          px + w > reference_frame->components[0].width ||
          py + h > reference_frame->components[0].height) {
        continue;
      }

      metric = calculate_metric2 (frame, x, y, reference_frame, px, py,
          w, h);

      if (metric < min_metric) {
        min_metric = metric;
        min_index = i;
      }
    }

    if (min_index == 0) {
      step_size >>= 1;
    } else {
      dx += hx[min_index] * step_size;
      dy += hy[min_index] * step_size;
    }
  }
  mv->x = dx;
  mv->y = dy;
  mv->metric = min_metric;
  mv->pred_mode = 1;
}
#endif

#if 0
static void
predict_motion_scan (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int dx,dy;
  int metric;

  for(dy = -4; dy <= 4; dy++) {
    for(dx = -4; dx <= 4; dx++) {
      if (y + dy < 0) continue;
      if (x + dx < 0) continue;
      if (y + dy + h > reference_frame->components[0].height) continue;
      if (x + dx + w > reference_frame->components[0].width) continue;

      metric = calculate_metric2 (frame, x, y, reference_frame,
          x + dx, y + dy, w, h);

      if (metric < mv->metric) {
        mv->metric = metric;
        mv->x = dx;
        mv->y = dy;
        mv->pred_mode = 1;
      }

    }
  }
}
#endif

#if 0
static void
predict_motion_none (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int metric;

  metric = calculate_metric2 (frame, x, y, reference_frame, x, y, w, h);
  mv->x = 0;
  mv->y = 0;
  mv->metric = metric;
  mv->pred_mode = 1;
}
#endif

#if 0
static void
predict_motion (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *reference_frame, int x, int y, int w, int h)
{
  int how = 2;

  switch(how) {
    case 0:
      predict_motion_scan (mv, frame, reference_frame, x, y, w, h);
      break;
    case 1:
      predict_motion_search (mv, frame, reference_frame, x, y, w, h);
      break;
    case 2:
      predict_motion_none (mv, frame, reference_frame, x, y, w, h);
      break;
  }
}
#endif

static void
predict_dc (SchroMotionVector *mv, SchroFrame *frame, int x, int y,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;
  uint8_t *a;
  int a_stride;
  int sum;

  if (height == 0 || width == 0) {
    mv->pred_mode = 0;
    mv->metric = 1000000;
    return;
  }

  SCHRO_ASSERT(x + width <= frame->components[0].width);
  SCHRO_ASSERT(y + height <= frame->components[0].height);

  a_stride = frame->components[0].stride;
  a = frame->components[0].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[0] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[0]);
    }
  }
  mv->metric = calculate_dc_metric(frame,x,y,mv->dc[0],width,height);

  width/=2;
  height/=2;
  x/=2;
  y/=2;

  a_stride = frame->components[1].stride;
  a = frame->components[1].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[1] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[1]);
    }
  }

  a_stride = frame->components[2].stride;
  a = frame->components[2].data + x + y * a_stride;
  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      sum += a[j*a_stride + i];
    }
  }
  mv->dc[2] = (sum+height*width/2)/(height*width);
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - mv->dc[2]);
    }
  }

  mv->pred_mode = 0;
}


#if 0
static void
schro_encoder_rough_global_prediction (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i;
  int j;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref;
  SchroFrame *downsampled_frame;
  SchroMotionVector *motion_vectors;

  downsampled_ref = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
      params->width/8, params->height/8, 2, 2);
  schro_frame_downsample (downsampled_ref, encoder->ref_frame0, 3);

  downsampled_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
      params->width/8, params->height/8, 2, 2);
  schro_frame_downsample (downsampled_frame, encoder->encode_frame, 3);

  x_blocks = params->width/8/8;
  y_blocks = params->height/8/8;

  motion_vectors = malloc (x_blocks*y_blocks*sizeof(SchroMotionVector));
  memset (motion_vectors, 0, x_blocks*y_blocks*sizeof(SchroMotionVector));

  for(j=0;j<y_blocks;j++){
    for(i=0;i<x_blocks;i++){
      schro_block_search (motion_vectors + j*x_blocks + i,
          downsampled_frame, downsampled_ref, i*8, j*8, 8);
    }
  }

  free(motion_vectors);
}
#endif

#if 0
static void
schro_block_search (SchroMotionVector *est, SchroFrame *frame,
    SchroFrame *ref, int x, int y, int distance)
{
  int i,j;
  int min_metric;
  int new_dx;
  int new_dy;

  new_dx = est->x;
  new_dy = est->y;
  min_metric = calculate_metric2 (frame, x, y, ref, x + est->x, y + est->y,
      8, 8);

  for(j=est->y-distance;j<=est->y+distance;j++){
    if (y + j < 0 || y + j + 8 >= frame->components[0].height) continue;
    for(i=est->x - distance;i<=est->x + distance;i++){
      int metric;

      if (x + i < 0 || x + i + 8 >= frame->components[0].width) continue;
      metric = calculate_metric2 (frame, x, y, ref, x + i, y + j, 8, 8);
      if (metric < min_metric) {
        new_dx = i;
        new_dy = j;
        min_metric = metric;
      }
    }
  }

  est->x = new_dx;
  est->y = new_dx;
  est->metric = min_metric;
}
#endif



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

#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define MAX(a,b) ((a)>(b) ? (a) : (b))

void
schro_prediction_list_scan (SchroPredictionList *list, SchroFrame *frame,
    SchroFrame *ref, int x, int y, int sx, int sy, int sw, int sh)
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
      vec.dx = i - x;
      vec.dy = j - y;
      vec.metric = calculate_metric3 (frame, x, y, ref, i, j, 8, 8);
      schro_prediction_list_insert (list, &vec);
    }
  }
}


void
schro_encoder_rough_global_prediction_2 (SchroEncoder *encoder)
{
  SchroParams *params = &encoder->params;
  int i;
  int j;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref;
  SchroFrame *downsampled_frame;
  SchroPredictionList *pred_lists;
  SchroPredictionList *prev_pred_lists;
  int shift;

  prev_pred_lists = NULL;

  for(shift=3;shift>=0;shift--) {
    downsampled_ref = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        params->width>>shift, params->height>>shift, 2, 2);
    schro_frame_downsample (downsampled_ref, encoder->ref_frame0, shift);

    downsampled_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
        params->width>>shift, params->height>>shift, 2, 2);
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
          for(k=0;k<MIN(2,list->n_vectors);k++){
            schro_prediction_list_scan (pred_lists + j*x_blocks + i,
                downsampled_frame, downsampled_ref, i*8, j*8,
                i*8 - 2 + list->vectors[k].dx*2,
                j*8 - 2 + list->vectors[k].dy*2, 5, 5);
          }
        }
      }
    }
    for(j=0;j<y_blocks;j++){
      for(i=0;i<x_blocks;i++){
        SchroPredictionList *list = pred_lists + j*x_blocks + i;
        if (list->n_vectors == 0) {
          schro_prediction_list_scan (pred_lists + j*x_blocks + i,
              downsampled_frame, downsampled_ref, i*8, j*8,
              i*8 - 8, j*8 - 8, 17, 17);
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

    schro_frame_free (downsampled_ref);
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

      mv->pred_mode = 1;
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

