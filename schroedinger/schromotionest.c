
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schrophasecorrelation.h>
#include <liboil/liboil.h>
#include <string.h>
#include <math.h>

#define DC_BIAS 50
#define DC_METRIC 50
#define BIDIR_LIMIT (10*8*8)

#define SCHRO_METRIC_INVALID_2 0x7fffffff

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

void schro_encoder_bigblock_estimation (SchroMotionEst *me);
void schro_motion_field_set (SchroMotionField *field, int split, int pred_mode);
void schro_motionest_rough_scan_nohint (SchroMotionEst *me,
    int shift, int ref, int distance);
void schro_motionest_rough_scan_hint (SchroMotionEst *me,
    int shift, int ref, int distance);
static SchroFrame * get_downsampled(SchroEncoderFrame *frame, int i);

void schro_motion_calculate_stats (SchroMotion *motion, SchroEncoderFrame *frame);


SchroMotionEst *
schro_motionest_new (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroMotionEst *me;
  int n;

  me = schro_malloc0 (sizeof(SchroMotionEst));

  me->encoder_frame = frame;
  me->params = &frame->params;

  me->downsampled_src0[0] = frame->ref_frame[0]->filtered_frame;
  me->downsampled_src0[1] = frame->ref_frame[0]->downsampled_frames[0];
  me->downsampled_src0[2] = frame->ref_frame[0]->downsampled_frames[1];
  me->downsampled_src0[3] = frame->ref_frame[0]->downsampled_frames[2];
  me->downsampled_src0[4] = frame->ref_frame[0]->downsampled_frames[3];

  if (me->params->num_refs > 1) {
    me->downsampled_src1[0] = frame->ref_frame[1]->filtered_frame;
    me->downsampled_src1[1] = frame->ref_frame[1]->downsampled_frames[0];
    me->downsampled_src1[2] = frame->ref_frame[1]->downsampled_frames[1];
    me->downsampled_src1[3] = frame->ref_frame[1]->downsampled_frames[2];
    me->downsampled_src1[4] = frame->ref_frame[1]->downsampled_frames[3];
  }

  n = params->x_num_blocks * params->y_num_blocks / 16;
  me->sblocks = schro_malloc0(sizeof(SchroBlock)*n);



  return me;
}

void
schro_motionest_free (SchroMotionEst *me)
{
#if 0
  int ref;
  int i;

  for(ref=0;ref<2;ref++){
    for(i=0;i<5;i++){
      if (me->downsampled_mf[ref][i]) {
        schro_motion_field_free (me->downsampled_mf[ref][i]);
      }
    }
  }
#endif

  schro_free (me->sblocks);

  schro_free (me);
}


void
schro_encoder_motion_predict_rough (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroEncoder *encoder = frame->encoder;
  int ref;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);
  SCHRO_ASSERT(params->num_refs > 0);

  for(ref=0;ref<params->num_refs;ref++){
    frame->rme[ref] = schro_rough_me_new (frame, frame->ref_frame[ref]);
    schro_rough_me_heirarchical_scan (frame->rme[ref]);

    if (encoder->enable_phasecorr_estimation) {
      frame->phasecorr[ref] = schro_phasecorr_new (frame,
          frame->ref_frame[ref]);
      schro_encoder_phasecorr_estimation (frame->phasecorr[ref]);
    }
  }
  if (encoder->enable_global_motion) {
    schro_encoder_global_estimation (frame);
  }

  frame->me = schro_motionest_new (frame);

  frame->motion = schro_motion_new (params, NULL, NULL);
  frame->me->motion = frame->motion;

#if 0
  for(ref=0;ref<params->num_refs;ref++){
    schro_motionest_rough_scan_nohint (frame->me, 3, ref, 12);
    schro_motionest_rough_scan_hint (frame->me, 2, ref, 2);
    schro_motionest_rough_scan_hint (frame->me, 1, ref, 2);
  }
#endif

}


void
schro_encoder_motion_predict_pel (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);
  SCHRO_ASSERT(params->num_refs > 0);

  schro_encoder_bigblock_estimation (frame->me);

  schro_motion_calculate_stats (frame->motion, frame);
  frame->estimated_mc_bits = schro_motion_estimate_entropy (frame->motion);

  frame->badblock_ratio = (double)frame->me->badblocks/(params->x_num_blocks*params->y_num_blocks/16);
}

void
schro_encoder_motion_refine_block_subpel (SchroEncoderFrame *frame,
    SchroBlock *block, int i, int j)
{
  SchroParams *params = &frame->params;
  int skip;
  int ii, jj;

  skip = 4 >> block->mv[0][0].split;
  for(jj=0;jj<4;jj+=skip){
    for(ii=0;ii<4;ii+=skip){
      if (block->mv[jj][ii].pred_mode & 1) {
        block->mv[jj][ii].dx[0] <<= params->mv_precision;
        block->mv[jj][ii].dy[0] <<= params->mv_precision;
      }
      if (block->mv[jj][ii].pred_mode & 2) {
        block->mv[jj][ii].dx[1] <<= params->mv_precision;
        block->mv[jj][ii].dy[1] <<= params->mv_precision;
      }
    }
  }

  if (block->mv[0][0].split < 3) {
    for(jj=0;jj<4;jj+=skip){
      for(ii=0;ii<4;ii+=skip){
        if (block->mv[jj][ii].pred_mode == 1 || block->mv[jj][ii].pred_mode == 2) {
          SchroUpsampledFrame *ref_upframe;
          SchroFrameData orig;
          SchroFrameData ref_fd;
          int dx,dy;
          int x,y;
          int metric;
          int width, height;
          int min_metric;
          int min_dx, min_dy;
          int ref;

          ref = block->mv[jj][ii].pred_mode - 1;
          ref_upframe = frame->ref_frame[ref]->upsampled_original_frame;

          x = MAX((i+ii)*frame->params.xbsep_luma, 0);
          y = MAX((j+jj)*frame->params.ybsep_luma, 0);
          width = skip*frame->params.xbsep_luma;
          height = skip*frame->params.ybsep_luma;

          schro_frame_get_subdata (get_downsampled (frame, 0), &orig, 0, x, y);

          min_metric = 0x7fffffff;
          min_dx = 0;
          min_dy = 0;
          for(dx=-1;dx<=1;dx++) {
            for(dy=-1;dy<=1;dy++) {
              schro_upsampled_frame_get_subdata_prec1 (ref_upframe, 0,
                  2*x + block->mv[jj][ii].dx[ref] + dx,
                  2*y + block->mv[jj][ii].dy[ref] + dy,
                  &ref_fd);

              metric = schro_metric_get (&orig, &ref_fd, width, height);
              if (metric < min_metric) {
                min_dx = dx;
                min_dy = dy;
                min_metric = metric;
              }
            }
          }
          block->mv[ii][ii].dx[ref] += min_dx;
          block->mv[jj][ii].dy[ref] += min_dy;
          block->error = metric;
        }
      }
    }
  }

  schro_block_fixup (block);
}

void
schro_encoder_motion_predict_subpel (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int i;
  int j;

  SCHRO_ASSERT(frame->upsampled_original_frame);
  SCHRO_ASSERT(frame->ref_frame[0]->upsampled_original_frame);
  if (frame->ref_frame[1]) {
    SCHRO_ASSERT(frame->ref_frame[1]->upsampled_original_frame);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      SchroBlock block = { 0 };

      schro_motion_copy_from (frame->me->motion, i, j, &block);
      schro_encoder_motion_refine_block_subpel (frame, &block, i, j);
      schro_block_fixup (&block);
      schro_motion_copy_to (frame->me->motion, i, j, &block);
    }
  }
}

void
schro_motion_calculate_stats (SchroMotion *motion, SchroEncoderFrame *frame)
{
  int i,j;
  SchroMotionVector *mv;
  int ref1 = 0;
  int ref2 = 0;
  int bidir = 0;

  frame->stats_dc = 0;
  frame->stats_global = 0;
  frame->stats_motion = 0;
  for(j=0;j<motion->params->y_num_blocks;j++){
    for(i=0;i<motion->params->x_num_blocks;i++){
      mv = SCHRO_MOTION_GET_BLOCK(motion,i,j);
      if (mv->pred_mode == 0) {
        frame->stats_dc++;
      } else {
        if (mv->using_global) {
          frame->stats_global++;
        } else {
          frame->stats_motion++;
        }
        if (mv->pred_mode == 1) {
          ref1++;
        } else if (mv->pred_mode == 2) {
          ref2++;
        } else {
          bidir++;
        }
      }
    }
  }
  SCHRO_DEBUG("dc %d global %d motion %d ref1 %d ref2 %d bidir %d",
      frame->stats_dc, frame->stats_global, frame->stats_motion,
      ref1, ref2, bidir);
}


SchroMotionField *
schro_motion_field_new (int x_num_blocks, int y_num_blocks)
{
  SchroMotionField *mf;

  mf = schro_malloc0 (sizeof(SchroMotionField));
  mf->x_num_blocks = x_num_blocks;
  mf->y_num_blocks = y_num_blocks;
  mf->motion_vectors = schro_malloc0 (sizeof(SchroMotionVector)*
      x_num_blocks*y_num_blocks);

  return mf;
}

void
schro_motion_field_free (SchroMotionField *field)
{
  schro_free (field->motion_vectors);
  schro_free (field);
}

#if 0
void
schro_motion_field_set (SchroMotionField *field, int split, int pred_mode)
{
  SchroMotionVector *mv;
  int i;
  int j;

  for(j=0;j<field->y_num_blocks;j++){
    for(i=0;i<field->x_num_blocks;i++){
      mv = field->motion_vectors + j*field->x_num_blocks + i;
      memset (mv, 0, sizeof (*mv));
      mv->split = split;
      mv->pred_mode = pred_mode;
      mv->metric = SCHRO_METRIC_INVALID;
    }
  }
}
#endif

#if 0
void
schro_motion_field_copy (SchroMotionField *field, SchroMotionField *parent)
{
  SchroMotionVector *mv;
  SchroMotionVector *pv;
  int i;
  int j;

  for(j=0;j<field->y_num_blocks;j++){
    for(i=0;i<field->x_num_blocks;i++){
      mv = field->motion_vectors + j*field->x_num_blocks + i;
      pv = parent->motion_vectors + (j>>1)*parent->x_num_blocks + (i>>1);
      *mv = *pv;
    }
  }
}
#endif

#if 0
void
schro_motion_field_dump (SchroMotionField *field)
{
  SchroMotionVector *mv;
  int i;
  int j;

  for(j=0;j<field->y_num_blocks;j++){
    for(i=0;i<field->x_num_blocks;i++){
      mv = field->motion_vectors + j*field->x_num_blocks + i;
      printf("%d %d %d %d\n", i, j, mv->dx[0], mv->dy[0]);
    }
  }
  exit(0);
}
#endif

static SchroFrame *
get_downsampled(SchroEncoderFrame *frame, int i)
{
  SCHRO_ASSERT(frame->have_downsampling);

  if (i==0) {
    return frame->filtered_frame;
  }
  return frame->downsampled_frames[i-1];
}

static int
schro_block_average (int16_t *dest, SchroFrameData *comp,
    int x, int y, int w, int h)
{
  int xmax = MIN(x + w, comp->width);
  int ymax = MIN(y + h, comp->height);
  int i,j;
  int n = 0;
  int sum = 0;
  int ave;

  if (x >= comp->width || y >= comp->height) return SCHRO_METRIC_INVALID_2;

  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      sum += SCHRO_GET(comp->data, j*comp->stride + i, uint8_t);
    }
    n += xmax - x;
  }

  if (n == 0) {
    return SCHRO_METRIC_INVALID_2;
  }

  ave = (sum + n/2)/n;

  sum = 0;
  for(j=y;j<ymax;j++){
    for(i=x;i<xmax;i++){
      sum += abs(ave - SCHRO_GET(comp->data, j*comp->stride + i, uint8_t));
    }
  }

  *dest = ave - 128;
  return sum;
}


static void
schro_motionest_superblock_scan_one (SchroMotionEst *me, int ref, int distance,
    SchroBlock *block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  //hint_mf = me->downsampled_mf[ref][2];
  hint_mf = me->encoder_frame->rme[ref]->motion_fields[2];

  scan.x = i * params->xbsep_luma;
  scan.y = j * params->ybsep_luma;
  scan.block_width = MIN(4*params->xbsep_luma, scan.frame->width - scan.x);
  scan.block_height = MIN(4*params->ybsep_luma, scan.frame->height - scan.y);
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[0][0];
  hint_mv = motion_field_get (hint_mf, i, j);

  dx = hint_mv->dx[ref];
  dy = hint_mv->dy[ref];

  schro_metric_scan_setup (&scan, dx, dy, distance);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->dx[ref] = 0;
    mv->dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  block->error = schro_metric_scan_get_min (&scan, &dx, &dy);
  mv->metric = block->error/16;

  mv->split = 0;
  mv->pred_mode = 1<<ref;
  mv->using_global = 0;
  mv->dx[ref] = dx;
  mv->dy[ref] = dy;

  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_predicted (SchroMotionEst *me, int ref,
    SchroBlock *block, int i, int j)
{
  SchroMotionVector *mv;
  int pred_x, pred_y;

  schro_motion_vector_prediction (me->motion, i, j, &pred_x, &pred_y, (1<<ref));

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 1<<ref;
  mv->using_global = 0;
  mv->dx[ref] = pred_x;
  mv->dy[ref] = pred_y;
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  block->entropy = 0;
  schro_block_fixup (block);

  block->valid = (block->error != SCHRO_METRIC_INVALID_2);
}

static void
schro_motionest_superblock_biref_zero (SchroMotionEst *me,
    SchroBlock *block, int i, int j)
{
  SchroMotionVector *mv;

  mv = &block->mv[0][0];
  mv->split = 0;
  mv->pred_mode = 3;
  mv->using_global = 0;
  mv->dx[0] = 0;
  mv->dy[0] = 0;
  mv->dx[1] = 0;
  mv->dy[1] = 0;
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = (block->error != SCHRO_METRIC_INVALID_2);
}

static void
schro_motionest_superblock_dc (SchroMotionEst *me,
    SchroBlock *block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVectorDC *mvdc;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mvdc = (SchroMotionVectorDC *)&block->mv[0][0];
  mvdc->split = 0;
  mvdc->pred_mode = 0;

  metric = schro_block_average (&mvdc->dc[0], frame->components + 0,
      i * params->xbsep_luma, j * params->ybsep_luma,
      4 * params->xbsep_luma, 4 * params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w = params->xbsep_luma>>SCHRO_CHROMA_FORMAT_H_SHIFT(params->video_format->chroma_format);
  chroma_h = params->ybsep_luma>>SCHRO_CHROMA_FORMAT_V_SHIFT(params->video_format->chroma_format);
  schro_block_average (&mvdc->dc[1], frame->components + 1,
      i * chroma_w, j * chroma_h, 4 * chroma_w, 4 * chroma_h);
  schro_block_average (&mvdc->dc[2], frame->components + 2,
      i * chroma_w, j * chroma_h, 4 * chroma_w, 4 * chroma_h);

  mvdc->metric = metric/16;
  block->error = metric;
  /* FIXME magic parameter */
  block->error += 4 * 2 * me->params->xbsep_luma * 10;

  schro_block_fixup (block);

  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
  block->valid = TRUE;
}

static void
schro_motionest_superblock_dc_predicted (SchroMotionEst *me,
    SchroBlock *block, int i, int j)
{
  SchroMotionVectorDC *mvdc;
  int pred[3];

  schro_motion_dc_prediction (me->motion, i, j, pred);

  mvdc = (SchroMotionVectorDC *)&block->mv[0][0];
  mvdc->split = 0;
  mvdc->pred_mode = 0;
  mvdc->dc[0] = pred[0];
  mvdc->dc[1] = pred[1];
  mvdc->dc[2] = pred[2];

  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  /* FIXME magic parameter */
  block->error += 4 * 2 * me->params->xbsep_luma * 10;
  mvdc->metric = block->error/16;

  schro_block_fixup (block);
  block->entropy = 0;
  block->valid = TRUE;
}

#ifdef unused
static void
schro_motion_splat_4x4 (SchroMotion *motion, int i, int j)
{
  SchroMotionVector *mv;

  mv = SCHRO_MOTION_GET_BLOCK (motion, i, j);
  mv[1] = mv[0];
  mv[2] = mv[0];
  mv[3] = mv[0];
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j+1), mv, 4*sizeof(*mv));
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j+2), mv, 4*sizeof(*mv));
  memcpy (SCHRO_MOTION_GET_BLOCK (motion, i, j+3), mv, 4*sizeof(*mv));
}
#endif

#ifdef unused
static void
motion_field_splat_4x4 (SchroMotionField *mf, int i, int j)
{
  SchroMotionVector *mv;

  mv = motion_field_get (mf, i, j);
  mv[1] = mv[0];
  mv[2] = mv[0];
  mv[3] = mv[0];
  memcpy (motion_field_get (mf, i, j+1), mv, 4*sizeof(*mv));
  memcpy (motion_field_get (mf, i, j+2), mv, 4*sizeof(*mv));
  memcpy (motion_field_get (mf, i, j+3), mv, 4*sizeof(*mv));
}
#endif

#ifdef unused
static void
schro_motionest_block_scan_one (SchroMotionEst *me, int ref, int distance,
    SchroBlock *block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;
  int ii, jj;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  block->error = 0;
  block->valid = TRUE;
  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      mv = &block->mv[jj][ii];
      hint_mv = motion_field_get (hint_mf, i + (ii&2), j + (jj&2));

      dx = hint_mv->dx[ref];
      dy = hint_mv->dy[ref];

      scan.x = (i + ii) * params->xbsep_luma;
      scan.y = (j + jj) * params->ybsep_luma;
      schro_metric_scan_setup (&scan, dx, dy, distance);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[ref] = 0;
        mv->dy[ref] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        block->error += mv->metric;
        block->valid = FALSE;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      block->error += mv->metric;
      block->valid &= (mv->metric != SCHRO_METRIC_INVALID);

      mv->split = 2;
      mv->pred_mode = 1<<ref;
      mv->using_global = 0;
      mv->dx[ref] = dx;
      mv->dy[ref] = dy;
    }
  }

  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
}
#endif


#define MAGIC_SUPERBLOCK_METRIC 5
#define MAGIC_BLOCK_METRIC 50

#define TRYBLOCK \
      score = tryblock.entropy + me->lambda * tryblock.error; \
      if (tryblock.valid && score < min_score) { \
        memcpy (&block, &tryblock, sizeof(block)); \
        min_score = score; \
      }

static void
schro_motionest_block_scan (SchroMotionEst *me, int ref, int distance,
    SchroBlock *block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[jj][ii];
  hint_mv = motion_field_get (hint_mf, i + (ii&2), j + (jj&2));

  dx = hint_mv->dx[ref];
  dy = hint_mv->dy[ref];

  scan.x = (i + ii) * params->xbsep_luma;
  scan.y = (j + jj) * params->ybsep_luma;
  schro_metric_scan_setup (&scan, dx, dy, distance);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->dx[ref] = 0;
    mv->dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
  block->error = mv->metric;
  block->valid = (mv->metric != SCHRO_METRIC_INVALID);

  mv->split = 2;
  mv->pred_mode = 1<<ref;
  mv->using_global = 0;
  mv->dx[ref] = dx;
  mv->dy[ref] = dy;

  schro_block_fixup (block);

  mv = SCHRO_MOTION_GET_BLOCK (me->motion, i + ii, j + jj);
  *mv = block->mv[jj][ii];
  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
}

static void
schro_motionest_block_dc (SchroMotionEst *me,
    SchroBlock *block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVectorDC *mvdc;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mvdc = (SchroMotionVectorDC *)&(block->mv[jj][ii]);
  mvdc->split = 2;
  mvdc->pred_mode = 0;

  metric = schro_block_average (&mvdc->dc[0], frame->components + 0,
      (i + ii) * params->xbsep_luma, (j + jj) * params->ybsep_luma,
      params->xbsep_luma, params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w = params->xbsep_luma>>SCHRO_CHROMA_FORMAT_H_SHIFT(params->video_format->chroma_format);
  chroma_h = params->ybsep_luma>>SCHRO_CHROMA_FORMAT_V_SHIFT(params->video_format->chroma_format);
  schro_block_average (&mvdc->dc[1], frame->components + 1,
      (i + ii) * chroma_w, (j+jj) * chroma_h, chroma_w, chroma_h);
  schro_block_average (&mvdc->dc[2], frame->components + 2,
      (i + ii) * chroma_w, (j+jj) * chroma_h, chroma_w, chroma_h);

  mvdc->metric = metric;
  block->error = metric;
  /* FIXME magic parameter */
  block->error += 4 * 2 * me->params->xbsep_luma * 10;

  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
  block->valid = TRUE;
}

void
schro_motionest_superblock_block (SchroMotionEst *me,
    SchroBlock *p_block, int i, int j)
{
  SchroParams *params = me->params;
  int ii,jj;
  SchroBlock block = { 0 };
  int total_error = 0;

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      block.mv[jj][ii].split = 2;
      block.mv[jj][ii].pred_mode = 1;
      block.mv[jj][ii].dx[0] = 0;
      block.mv[jj][ii].dy[0] = 0;
    }
  }
  schro_motion_copy_to (me->motion, i, j, &block);

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      schro_motionest_block_scan (me, 0, 4, &block, i, j, ii, jj);
      min_score = block.entropy + me->lambda * block.error;

      if (params->num_refs > 1) {
        memcpy (&tryblock, &block, sizeof(block));
        schro_motionest_block_scan (me, 1, 4, &tryblock, i, j, ii, jj);
        TRYBLOCK
      }

      memcpy (&tryblock, &block, sizeof(block));
      schro_motionest_block_dc (me, &tryblock, i, j, ii, jj);
      TRYBLOCK

      total_error += block.error;
    }
  }
  block.entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, &block);
  block.error = total_error;

  memcpy (p_block, &block, sizeof(block));
}

static void
schro_motionest_subsuperblock_scan (SchroMotionEst *me, int ref, int distance,
    SchroBlock *block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

  hint_mf = me->encoder_frame->rme[ref]->motion_fields[1];

  scan.block_width = 2*params->xbsep_luma;
  scan.block_height = 2*params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  mv = &block->mv[jj][ii];
  hint_mv = motion_field_get (hint_mf, i + (ii&2), j + (jj&2));

  dx = hint_mv->dx[ref];
  dy = hint_mv->dy[ref];

  scan.x = (i + ii) * params->xbsep_luma;
  scan.y = (j + jj) * params->ybsep_luma;
  schro_metric_scan_setup (&scan, dx, dy, distance);
  if (scan.scan_width <= 0 || scan.scan_height <= 0) {
    mv->dx[ref] = 0;
    mv->dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    block->error += mv->metric;
    block->valid = FALSE;
    return;
  }

  schro_metric_scan_do_scan (&scan);
  mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
  block->error = mv->metric;
  block->valid = (mv->metric != SCHRO_METRIC_INVALID);

  mv->split = 1;
  mv->pred_mode = 1<<ref;
  mv->using_global = 0;
  mv->dx[ref] = dx;
  mv->dy[ref] = dy;

  schro_block_fixup (block);

  mv = SCHRO_MOTION_GET_BLOCK (me->motion, i + ii, j + jj);
  *mv = block->mv[jj][ii];
  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
}

static void
schro_motionest_subsuperblock_dc (SchroMotionEst *me,
    SchroBlock *block, int i, int j, int ii, int jj)
{
  SchroParams *params = me->params;
  SchroMotionVectorDC *mvdc;
  int chroma_w, chroma_h;
  SchroFrame *frame;
  int metric;

  frame = get_downsampled (me->encoder_frame, 0);

  mvdc = (SchroMotionVectorDC *)&block->mv[jj][ii];
  mvdc->split = 1;
  mvdc->pred_mode = 0;

  metric = schro_block_average (&mvdc->dc[0], frame->components + 0,
      (i + ii) * params->xbsep_luma, (j + jj) * params->ybsep_luma,
      2*params->xbsep_luma, 2*params->ybsep_luma);
  if (metric == SCHRO_METRIC_INVALID_2) {
    block->valid = FALSE;
    return;
  }
  chroma_w = params->xbsep_luma>>SCHRO_CHROMA_FORMAT_H_SHIFT(params->video_format->chroma_format);
  chroma_h = params->ybsep_luma>>SCHRO_CHROMA_FORMAT_V_SHIFT(params->video_format->chroma_format);
  schro_block_average (&mvdc->dc[1], frame->components + 1,
      (i + ii) * chroma_w, (j+jj) * chroma_h, 2*chroma_w, 2*chroma_h);
  schro_block_average (&mvdc->dc[2], frame->components + 2,
      (i + ii) * chroma_w, (j+jj) * chroma_h, 2*chroma_w, 2*chroma_h);

  mvdc->metric = metric / 4;
  block->error = metric;
  /* FIXME magic parameter */
  block->error += 4 * 2 * me->params->xbsep_luma * 10;

#if 0
  block->error = schro_motionest_superblock_get_metric (me, block, i, j);
  /* FIXME magic parameter */
  block->error += 4 * 2 * me->params->xbsep_luma * 10;
  mvdc->metric = block->error/16;
#endif

  block->entropy = schro_motion_block_estimate_entropy (me->motion,
      i + ii, j + jj);
  block->valid = TRUE;
}

void
schro_motionest_superblock_subsuperblock (SchroMotionEst *me,
    SchroBlock *p_block, int i, int j)
{
  SchroParams *params = me->params;
  int ii,jj;
  SchroBlock block = { 0 };
  int total_error = 0;

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      block.mv[jj][ii].split = 1;
      block.mv[jj][ii].pred_mode = 1;
      block.mv[jj][ii].dx[0] = 0;
      block.mv[jj][ii].dy[0] = 0;
    }
  }
  schro_motion_copy_to (me->motion, i, j, &block);

  for(jj=0;jj<4;jj+=2){
    for(ii=0;ii<4;ii+=2){
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      schro_motionest_subsuperblock_scan (me, 0, 4, &block, i, j, ii, jj);
      min_score = block.entropy + me->lambda * block.error;

      if (params->num_refs > 1) {
        memcpy (&tryblock, &block, sizeof(block));
        schro_motionest_subsuperblock_scan (me, 1, 4, &tryblock, i, j, ii, jj);
        TRYBLOCK

#if 0
        memcpy (&tryblock, &block, sizeof(block));
        schro_motionest_block_biref_zero (me, 1, &tryblock, i, j, ii, jj);
        TRYBLOCK
#endif
      }

if (1) {
      memcpy (&tryblock, &block, sizeof(block));
      schro_motionest_subsuperblock_dc (me, &tryblock, i, j, ii, jj);
      TRYBLOCK
}

      total_error += block.error;
    }
  }
  block.entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, &block);
  block.error = total_error;

  memcpy (p_block, &block, sizeof(block));
}

void schro_motionest_superblock_phasecorr1 (SchroMotionEst *me, int ref,
    SchroBlock *block, int i, int j);
void schro_motionest_superblock_global (SchroMotionEst *me, int ref,
    SchroBlock *block, int i, int j);

void
schro_encoder_bigblock_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int i,j;
  double total_error = 0;
  int block_size;
  int block_threshold;

  me->lambda = me->encoder_frame->encoder->magic_mc_lambda;

  block_size = 16 * params->xbsep_luma * params->ybsep_luma;
  block_threshold = params->xbsep_luma * params->ybsep_luma *
    me->encoder_frame->encoder->magic_block_search_threshold;

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      SchroBlock block = { 0 };
      SchroBlock tryblock = { 0 };
      double score;
      double min_score;

      /* base 119 s */
      schro_motionest_superblock_predicted (me, 0, &block, i, j);
      min_score = block.entropy + me->lambda * block.error;
      if (params->num_refs > 1) {
        schro_motionest_superblock_predicted (me, 1, &tryblock, i, j);
        TRYBLOCK
      }

      /* 16 s */
      schro_motionest_superblock_scan_one (me, 0, 4, &tryblock, i, j);
      TRYBLOCK
      if (params->num_refs > 1) {
        schro_motionest_superblock_scan_one (me, 1, 4, &tryblock, i, j);
        TRYBLOCK
      }

      /* 2.5 s */
      schro_motionest_superblock_dc_predicted (me, &tryblock, i, j);
      TRYBLOCK
      schro_motionest_superblock_dc (me, &tryblock, i, j);
      TRYBLOCK

      /* 3.0 s */
      if (params->num_refs > 1) {
        schro_motionest_superblock_biref_zero (me, &tryblock, i, j);
        TRYBLOCK
      }

      if (min_score > block_threshold) {
        schro_motionest_superblock_subsuperblock (me, &tryblock, i, j);
        TRYBLOCK

        schro_motionest_superblock_block (me, &tryblock, i, j);
        TRYBLOCK
      }

      if (me->encoder_frame->encoder->enable_phasecorr_estimation) {
        schro_motionest_superblock_phasecorr1 (me, 0, &tryblock, i, j);
        TRYBLOCK
        if (params->num_refs > 1) {
          schro_motionest_superblock_phasecorr1 (me, 1, &tryblock, i, j);
          TRYBLOCK
        }
      }

      if (me->encoder_frame->encoder->enable_global_motion) {
        schro_motionest_superblock_global (me, 0, &tryblock, i, j);
        TRYBLOCK
        if (params->num_refs > 1) {
          schro_motionest_superblock_global (me, 1, &tryblock, i, j);
          TRYBLOCK
        }
      }

      if (block.error > 10*block_size) {
        me->badblocks++;
      }

      schro_block_fixup (&block);
      schro_motion_copy_to (me->motion, i, j, &block);

      total_error += (double)block.error*block.error/(double)(block_size * block_size);
    }
  }

  me->encoder_frame->mc_error = total_error/(240.0*240.0)/
    (params->x_num_blocks*params->y_num_blocks/16);

  /* magic parameter */
  me->encoder_frame->mc_error *= 2.5;
}

int
schro_motion_block_estimate_entropy (SchroMotion *motion, int i, int j)
{
  SchroMotionVector *mv;
  int entropy = 0;

  mv = SCHRO_MOTION_GET_BLOCK (motion, i, j);

  if (mv->split == 0 && (i&3 || j&3)) return 0;
  if (mv->split == 1 && (i&1 || j&1)) return 0;

  if (mv->pred_mode == 0) {
    SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;
    int pred[3];

    schro_motion_dc_prediction (motion, i, j, pred);

    entropy += schro_pack_estimate_sint (mvdc->dc[0] - pred[0]);
    entropy += schro_pack_estimate_sint (mvdc->dc[1] - pred[1]);
    entropy += schro_pack_estimate_sint (mvdc->dc[2] - pred[2]);

    return entropy;
  }
  if (mv->using_global) return 0;
  if (mv->pred_mode & 1) {
    int pred_x, pred_y;
    schro_motion_vector_prediction (motion, i, j, &pred_x, &pred_y, 1);
    entropy += schro_pack_estimate_sint (mv->dx[0] - pred_x);
    entropy += schro_pack_estimate_sint (mv->dy[0] - pred_y);
  }
  if (mv->pred_mode & 2) {
    int pred_x, pred_y;
    schro_motion_vector_prediction (motion, i, j, &pred_x, &pred_y, 2);
    entropy += schro_pack_estimate_sint (mv->dx[1] - pred_x);
    entropy += schro_pack_estimate_sint (mv->dy[1] - pred_y);
  }
  return entropy;
}

int
schro_motion_estimate_entropy (SchroMotion *motion)
{
  SchroParams *params = motion->params;
  int i,j;
  int entropy = 0;

  for (j=0;j<params->y_num_blocks;j++){
    for (i=0;i<params->x_num_blocks;i++){
      entropy += schro_motion_block_estimate_entropy (motion, i, j);
    }
  }

  return entropy;
}

#ifdef unused
int
schro_motion_superblock_estimate_entropy (SchroMotion *motion, int i, int j)
{
  int ii,jj;
  int entropy = 0;

  for(jj=j;jj<j+4;jj++){
    for(ii=i;ii<i+4;ii++){
      entropy += schro_motion_block_estimate_entropy (motion, ii, jj);
    }
  }

  return entropy;
}
#endif

int
schro_motion_superblock_try_estimate_entropy (SchroMotion *motion, int i,
    int j, SchroBlock *block)
{
  int ii,jj;
  int entropy = 0;
  SchroBlock save_block;

  schro_motion_copy_from (motion, i, j, &save_block);
  schro_motion_copy_to (motion, i, j, block);
  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      entropy += schro_motion_block_estimate_entropy (motion, i + ii, j + jj);
    }
  }
  schro_motion_copy_to (motion, i, j, &save_block);

  return entropy;
}

int
schro_motionest_superblock_get_metric (SchroMotionEst *me,
    SchroBlock *block, int i, int j)
{
  SchroMotionVector *mv;
  SchroFrameData orig;
  int width, height;
  int xmin, ymin;
  int xmax, ymax;

  xmin = MAX(i*me->params->xbsep_luma, 0);
  ymin = MAX(j*me->params->ybsep_luma, 0);
  xmax = MIN((i+4)*me->params->xbsep_luma, me->encoder_frame->filtered_frame->width);
  ymax = MIN((j+4)*me->params->ybsep_luma, me->encoder_frame->filtered_frame->height);

  schro_frame_get_subdata (get_downsampled (me->encoder_frame, 0), &orig,
      0, xmin, ymin);

  width = xmax - xmin;
  height = ymax - ymin;

  mv = &block->mv[0][0];

  if (mv->pred_mode == 0) {
    SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

    return schro_metric_get_dc (&orig, mvdc->dc[0], width, height);
  }
  if (mv->pred_mode == 1 || mv->pred_mode == 2) {
    SchroFrame *ref_frame;
    SchroFrameData ref_data;
    int ref;

    ref = mv->pred_mode - 1;

    ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], 0);

    if (xmin + mv->dx[ref] < -ref_frame->extension ||
        ymin + mv->dy[ref] < -ref_frame->extension ||
        xmax + mv->dx[ref] > me->encoder_frame->filtered_frame->width + ref_frame->extension ||
        ymax + mv->dy[ref] > me->encoder_frame->filtered_frame->height + ref_frame->extension) {
      /* bailing because it's "hard" */
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (ref_frame, &ref_data,
        0, xmin + mv->dx[ref], ymin + mv->dy[ref]);

    return schro_metric_get (&orig, &ref_data, width, height);
  }

  if (mv->pred_mode == 3) {
    SchroFrame *ref0_frame;
    SchroFrame *ref1_frame;
    SchroFrameData ref0_data;
    SchroFrameData ref1_data;

    ref0_frame = get_downsampled (me->encoder_frame->ref_frame[0], 0);
    ref1_frame = get_downsampled (me->encoder_frame->ref_frame[1], 0);

    if (xmin + mv->dx[0] < -ref0_frame->extension ||
        ymin + mv->dy[0] < -ref0_frame->extension ||
        xmax + mv->dx[0] > me->encoder_frame->filtered_frame->width + ref0_frame->extension ||
        ymax + mv->dy[0] > me->encoder_frame->filtered_frame->height + ref0_frame->extension ||
        xmin + mv->dx[1] < -ref1_frame->extension ||
        ymin + mv->dy[1] < -ref1_frame->extension ||
        xmax + mv->dx[1] > me->encoder_frame->filtered_frame->width + ref1_frame->extension ||
        ymax + mv->dy[1] > me->encoder_frame->filtered_frame->height + ref1_frame->extension) {
      /* bailing because it's "hard" */
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (ref0_frame,
        &ref0_data, 0, xmin + mv->dx[0], ymin + mv->dy[0]);
    schro_frame_get_subdata (ref1_frame,
        &ref1_data, 0, xmin + mv->dx[1], ymin + mv->dy[1]);

    return schro_metric_get_biref (&orig, &ref0_data, 1, &ref1_data, 1, 1, width, height);
  }

  SCHRO_ASSERT(0);

  return SCHRO_METRIC_INVALID_2;
}

#ifdef unused
int
schro_block_check (SchroBlock *block)
{
  SchroMotionVector *sbmv;
  SchroMotionVector *bmv;
  SchroMotionVector *mv;
  int i,j;

  sbmv = &block->mv[0][0];
  for(j=0;j<4;j++){
    for(i=0;i<4;i++){
      mv = &block->mv[j][i];

      switch (sbmv->split) {
        case 0:
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
            SCHRO_ERROR("mv(%d,%d) not equal to superblock mv", i, j);
            return 0;
          }
          break;
        case 1:
          bmv = &block->mv[(j&~1)][(i&~1)];
          if (!schro_motion_vector_is_equal (mv, bmv)) {
            SCHRO_ERROR("mv(%d,%d) not equal to 2-block mv", i, j);
            return 0;
          }
          break;
        case 2:
          break;
        default:
          SCHRO_ERROR("mv(%d,%d) has bad split", i, j);
          return 0;
          break;
      }
    }
  }

  return 1;
}
#endif

void
schro_block_fixup (SchroBlock *block)
{
  SchroMotionVector *mv;

  mv = &block->mv[0][0];
  if (mv->split == 0) {
    mv[1] = mv[0];
    mv[2] = mv[0];
    mv[3] = mv[0];
    memcpy (mv + 4, mv, 4*sizeof(*mv));
    memcpy (mv + 8, mv, 4*sizeof(*mv));
    memcpy (mv + 12, mv, 4*sizeof(*mv));
  }
  if (mv->split == 1) {
    mv[1] = mv[0];
    mv[3] = mv[2];
    memcpy (mv + 4, mv, 4*sizeof(*mv));
    mv[9] = mv[8];
    mv[11] = mv[10];
    memcpy (mv + 12, mv + 8, 4*sizeof(*mv));
  }
}

void
schro_motion_copy_from (SchroMotion *motion, int i, int j, SchroBlock *block)
{
  SchroMotionVector *mv;
  int ii,jj;

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      mv = SCHRO_MOTION_GET_BLOCK (motion, i + ii, j + jj);
      block->mv[jj][ii] = *mv;
    }
  }
}

void
schro_motion_copy_to (SchroMotion *motion, int i, int j, SchroBlock *block)
{
  SchroMotionVector *mv;
  int ii,jj;

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      mv = SCHRO_MOTION_GET_BLOCK (motion, i + ii, j + jj);
      *mv = block->mv[jj][ii];
    }
  }
}

