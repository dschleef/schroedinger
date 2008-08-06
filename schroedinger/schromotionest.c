
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
void schro_motion_global_metric (SchroMotionField *mf, SchroFrame *frame,
    SchroFrame *ref);
void schro_motionest_rough_scan_nohint (SchroMotionEst *me,
    int shift, int ref, int distance);
void schro_motionest_rough_scan_hint (SchroMotionEst *me,
    int shift, int ref, int distance);
void schro_rough_me_heirarchical_scan_nohint (SchroRoughME *rme, int shift,
    int distance);
void schro_rough_me_heirarchical_scan_hint (SchroRoughME *rme, int shift,
    int distance);
static SchroFrame * get_downsampled(SchroEncoderFrame *frame, int i);

#ifdef unused
void schro_motion_predict_subpixel (SchroMotion *motion, SchroFrame *frame,
    SchroMotionField *mf);
#endif
void schro_motion_calculate_stats (SchroMotion *motion, SchroEncoderFrame *frame);

#ifdef unused
static void motion_field_splat_4x4 (SchroMotionField *mf, int i, int j);
static void motion_field_splat_2x2 (SchroMotionField *mf, int i, int j);
#endif

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

SchroRoughME *
schro_rough_me_new (SchroEncoderFrame *frame, SchroEncoderFrame *ref)
{
  SchroRoughME *rme;

  rme = schro_malloc0 (sizeof(SchroRoughME));

  rme->encoder_frame = frame;
  rme->ref_frame = ref;

  return rme;
}

void
schro_rough_me_free (SchroRoughME *rme)
{
  int i;
  for(i=0;i<SCHRO_MAX_HIER_LEVELS;i++){
    if (rme->motion_fields[i]) schro_motion_field_free (rme->motion_fields[i]);
  }
  schro_free (rme);
}

void
schro_rough_me_heirarchical_scan (SchroRoughME *rme)
{
  SchroParams *params = &rme->encoder_frame->params;
  int i;
  int n_levels = rme->encoder_frame->encoder->downsample_levels;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);
  SCHRO_ASSERT(params->num_refs > 0);

  schro_rough_me_heirarchical_scan_nohint (rme, n_levels, 12);
  for (i=n_levels-1; i >= 1; i--) {
    schro_rough_me_heirarchical_scan_hint (rme, i, 4);
  }
}

void
schro_rough_me_heirarchical_scan_nohint (SchroRoughME *rme, int shift,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroParams *params = &rme->encoder_frame->params;
  int i;
  int j;
  int skip;

  scan.frame = get_downsampled (rme->encoder_frame, shift);
  scan.ref_frame = get_downsampled (rme->ref_frame, shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

  schro_motion_field_set (mf, 0, 1);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  skip = 1<<shift;
  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      int dx, dy;

      scan.x = (i>>shift) * params->xbsep_luma;
      scan.y = (j>>shift) * params->ybsep_luma;
      scan.block_width = MIN(scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN(scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, 0, 0, distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[0] = 0 << shift;
        mv->dy[0] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#if 0
      /* this code skips blocks that are off the edge.  Instead, we
       * scan smaller block sizes */
      if (scan.x + scan.block_width >= scan.ref_frame->width ||
          scan.y + scan.block_height >= scan.ref_frame->height) {
        mv->dx[0] = 0 << shift;
        mv->dy[0] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#endif

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      dx <<= shift;
      dy <<= shift;

      mv->dx[0] = dx;
      mv->dy[0] = dy;
    }
  }

  rme->motion_fields[shift] = mf;
}

void
schro_rough_me_heirarchical_scan_hint (SchroRoughME *rme, int shift,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroMotionField *hint_mf;
  SchroParams *params = &rme->encoder_frame->params;
  SchroMotionVector zero_mv;
  int i;
  int j;
  int skip;
  unsigned int hint_mask;

  scan.frame = get_downsampled (rme->encoder_frame, shift);
  scan.ref_frame = get_downsampled (rme->ref_frame, shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  hint_mf = rme->motion_fields[shift+1];

  schro_motion_field_set (mf, 0, 1);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  memset (&zero_mv, 0, sizeof(zero_mv));

  hint_mask = ~((1<<(shift + 1))-1);
  skip = 1<<shift;
  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      SchroFrameData orig;
      SchroFrameData ref_data;
#define LIST_LENGTH 10
      SchroMotionVector *hint_mv[LIST_LENGTH];
      int m;
      int n = 0;
      int dx, dy;
      int min_m;
      int min_metric;

      schro_frame_get_subdata (scan.frame, &orig,
          0, i*params->xbsep_luma >> shift,
          j*params->ybsep_luma >> shift);

      /* always test the zero vector */
      hint_mv[n] = &zero_mv;
      n++;

      /* inherit from nearby parents */
      /* This overly clever bit of code checks the parents of the diagonal
       * neighbors, which corresponds to the nearest parents. */
      for(m=0;m<4;m++) {
        int l = (i + skip*(-1 + 2*(m&1)))&hint_mask;
        int k = (j + skip*(-1 + (m&2)))&hint_mask;
        if (l >= 0 && l < params->x_num_blocks &&
            k >= 0 && k < params->y_num_blocks) {
          hint_mv[n] = motion_field_get (hint_mf, l, k);
          n++;
        }
      }

      /* inherit from neighbors (only towards SE) */
      if (i > 0) {
        hint_mv[n] = motion_field_get (mf, i-skip, j);
        n++;
      }
      if (j > 0) {
        hint_mv[n] = motion_field_get (mf, i, j - skip);
        n++;
      }
      if (i > 0 && j > 0) {
        hint_mv[n] = motion_field_get (mf, i - skip, j - skip);
        n++;
      }

      SCHRO_ASSERT(n <= LIST_LENGTH);

      min_m = 0;
      min_metric = SCHRO_METRIC_INVALID;
      for(m = 0; m < n; m++) {
        int metric;
        int width, height;
        int x,y;

        dx = hint_mv[m]->dx[0];
        dy = hint_mv[m]->dy[0];


        x = (i*params->xbsep_luma + dx) >> shift;
        y = (j*params->ybsep_luma + dy) >> shift;
        if (x < 0 || y < 0) {
          //SCHRO_ERROR("ij %d %d dx dy %d %d", i, j, dx, dy);
          continue;
        }

        schro_frame_get_subdata (scan.ref_frame,
            &ref_data, 0,
            (i*params->xbsep_luma + dx) >> shift,
            (j*params->ybsep_luma + dy) >> shift);

        width = MIN(params->xbsep_luma, orig.width);
        height = MIN(params->ybsep_luma, orig.height);
        if (width == 0 || height == 0) continue;
        if (ref_data.width < width || ref_data.height < height) continue;

        metric = schro_metric_get (&orig, &ref_data, width, height);

        if (metric < min_metric) {
          min_metric = metric;
          min_m = m;
        }
      }

      dx = hint_mv[min_m]->dx[0] >> shift;
      dy = hint_mv[min_m]->dy[0] >> shift;

      scan.x = (i>>shift) * params->xbsep_luma;
      scan.y = (j>>shift) * params->ybsep_luma;
      scan.block_width = MIN(scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN(scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, dx, dy, distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[0] = 0;
        mv->dy[0] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      dx <<= shift;
      dy <<= shift;

      mv->dx[0] = dx;
      mv->dy[0] = dy;
    }
  }

  rme->motion_fields[shift] = mf;
}

void
schro_encoder_motion_predict_rough (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  int n;
  int ref;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);
  SCHRO_ASSERT(params->num_refs > 0);

  for(ref=0;ref<params->num_refs;ref++){
    frame->rme[ref] = schro_rough_me_new (frame, frame->ref_frame[0]);
    schro_rough_me_heirarchical_scan (frame->rme[ref]);
  }

  frame->me = schro_motionest_new (frame);

  frame->motion = schro_motion_new (params, NULL, NULL);
  frame->me->motion = frame->motion;

  frame->motion_field_list = schro_list_new_full ((SchroListFreeFunc)schro_motion_field_free, NULL);
  n = 0;

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

#if 0
    if (frame->encoder->enable_phasecorr_estimation) {
      schro_encoder_phasecorr_estimation (frame->me);
    }
    if (params->have_global_motion) {
      schro_encoder_global_estimation (frame->me);
#endif

  schro_motion_calculate_stats (frame->motion, frame);
  frame->estimated_mc_bits = schro_motion_estimate_entropy (frame->motion);

  schro_list_free (frame->motion_field_list);

  frame->badblock_ratio = (double)frame->me->badblocks/(params->x_num_blocks*params->y_num_blocks/16);
}

void
schro_motion_field_lshift (SchroMotionField *mf, int n)
{
  int i,j;
  SchroMotionVector *mv;

  for(j=0;j<mf->y_num_blocks;j++){
    for(i=0;i<mf->x_num_blocks;i++){
      mv = motion_field_get(mf,i,j);

      if (mv->using_global || mv->pred_mode == 0) continue;
      if (mv->pred_mode & 3) {
        mv->dx[0] <<= n;
        mv->dy[0] <<= n;
        mv->dx[1] <<= n;
        mv->dy[1] <<= n;
      }
    }
  }
}

#if 0
static void
schro_motion_predict_subpixel (SchroMotion *motion, SchroFrame *frame,
    SchroMotionField *mf)
{
  int i,j;
  SchroMotionVector *mv;
  int x,y;
  SchroUpsampledFrame *uf;

  for(j=0;j<motion->params->y_num_blocks;j++){
    for(i=0;i<motion->params->x_num_blocks;i++){
      int metric;
      int dx, dy;

      mv = motion_field_get(mf,i,j);

      if (mv->pred_mode & 1) {
        uf = motion->src1;
        dx = mv->x1;
        dy = mv->y1;
      } else {
        uf = motion->src2;
        dx = mv->x2;
        dy = mv->y2;
      }

      x = i * motion->params->xblen_luma;
      y = j * motion->params->yblen_luma;

      schro_motion_x_get_block (motion, 0, uf, x, y, dx, dy);
      metric = schro_metric_absdiff_u8 (motion->blocks[0], motion->strides[0],
          frame->components[0].data + x + y*frame->components[0].stride,
          frame->components[0].stride, 8, 8);

      SCHRO_ERROR("%d %d", metric, mv->metric);

#if 0
      for (l=-n;l<=n;l++){
        for (k=-n;k<=n;k++){

        }
      }
#endif
    }
  }
}
#endif

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

void
schro_encoder_global_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  SchroMotionField *mf, *mf_orig;
  int i;

  SCHRO_ERROR("Global prediction is broken.  Please try again later");

  for(i=0;i<params->num_refs;i++) {
    //mf_orig = me->downsampled_mf[i][0];
    mf_orig = me->encoder_frame->rme[i]->motion_fields[0];
    mf = schro_motion_field_new (mf_orig->x_num_blocks, mf_orig->y_num_blocks);

    memcpy (mf->motion_vectors, mf_orig->motion_vectors,
        sizeof(SchroMotionVector)*mf->x_num_blocks*mf->y_num_blocks);
    schro_motion_field_global_estimation (mf,
        &me->encoder_frame->params.global_motion[i],
        params->mv_precision);
    if (i == 0) {
      schro_motion_global_metric (mf,
          me->encoder_frame->filtered_frame,
          me->encoder_frame->ref_frame[0]->filtered_frame);
    } else {
      schro_motion_global_metric (mf, me->encoder_frame->filtered_frame,
          me->encoder_frame->ref_frame[1]->filtered_frame);
    }
    schro_list_append (me->encoder_frame->motion_field_list, mf);
  }
}

void
schro_motion_global_metric (SchroMotionField *field, SchroFrame *frame,
    SchroFrame *ref)
{
  SchroMotionVector *mv;
  int i;
  int j;
  int x,y;

  for(j=0;j<field->y_num_blocks;j++){
    for(i=0;i<field->x_num_blocks;i++){
      mv = field->motion_vectors + j*field->x_num_blocks + i;

      x = i*8 + mv->dx[0];
      y = j*8 + mv->dy[0];
#if 0
      mv->metric = schro_metric_absdiff_u8 (
            frame->components[0].data + x + y*frame->components[0].stride,
            frame->components[0].stride,
            ref->components[0].data + i*8 + j*8*ref->components[0].stride,
            ref->components[0].stride, 8, 8);
#endif
mv->metric = 0;
    }
  }
}

void
schro_motion_field_global_estimation (SchroMotionField *mf,
    SchroGlobalMotion *gm, int mv_precision)
{
  int i;
  int j;
  int k;
  SchroMotionVector *mv;

  for(j=0;j<mf->y_num_blocks;j++) {
    for(i=0;i<mf->x_num_blocks;i++) {
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;

      mv->using_global = 1;

      /* HACK */
      if (j >= mf->y_num_blocks - 8 || i >= mf->x_num_blocks - 8) {
        mv->using_global = 0;
      }
    }
  }

  for(k=0;k<4;k++){
    double m_x, m_y;
    double m_f, m_g;
    double pan_x, pan_y;
    double ave_x, ave_y;
    double m_fx, m_fy, m_gx, m_gy;
    double m_xx, m_yy;
    double a00, a01, a10, a11;
    double sum2;
    double stddev2;
    int n = 0;

    SCHRO_DEBUG("step %d", k);
    m_x = 0;
    m_y = 0;
    m_f = 0;
    m_g = 0;
    for(j=0;j<mf->y_num_blocks;j++) {
      for(i=0;i<mf->x_num_blocks;i++) {
        mv = mf->motion_vectors + j*mf->x_num_blocks + i;
        if (mv->using_global) {
          m_f += mv->dx[0];
          m_g += mv->dy[0];
          m_x += i*8;
          m_y += j*8;
          n++;
        }
      }
    }
    pan_x = m_f / n;
    pan_y = m_g / n;
    ave_x = m_x / n;
    ave_y = m_y / n;

    SCHRO_DEBUG("pan %f %f ave %f %f n %d", pan_x, pan_y, ave_x, ave_y, n);

    m_fx = 0;
    m_fy = 0;
    m_gx = 0;
    m_gy = 0;
    m_xx = 0;
    m_yy = 0;
    n = 0;
    for(j=0;j<mf->y_num_blocks;j++) {
      for(i=0;i<mf->x_num_blocks;i++) {
        mv = mf->motion_vectors + j*mf->x_num_blocks + i;
        if (mv->using_global) {
          m_fx += (mv->dx[0] - pan_x) * (i*8 - ave_x);
          m_fy += (mv->dx[0] - pan_x) * (j*8 - ave_y);
          m_gx += (mv->dy[0] - pan_y) * (i*8 - ave_x);
          m_gy += (mv->dy[0] - pan_y) * (j*8 - ave_y);
          m_xx += (i*8 - ave_x) * (i*8 - ave_x);
          m_yy += (j*8 - ave_y) * (j*8 - ave_y);
          n++;
        }
      }
    }
    SCHRO_DEBUG("m_fx %f m_gx %f m_xx %f n %d", m_fx, m_gx, m_xx, n);
    a00 = m_fx / m_xx;
    a01 = m_fy / m_yy;
    a10 = m_gx / m_xx;
    a11 = m_gy / m_yy;

    pan_x -= a00*ave_x + a01*ave_y;
    pan_y -= a10*ave_x + a11*ave_y;

    SCHRO_DEBUG("pan %f %f a[] %f %f %f %f", pan_x, pan_y, a00, a01, a10, a11);

    sum2 = 0;
    for(j=0;j<mf->y_num_blocks;j++) {
      for(i=0;i<mf->x_num_blocks;i++) {
        mv = mf->motion_vectors + j*mf->x_num_blocks + i;
        if (mv->using_global) {
          double dx, dy;
          dx = mv->dx[0] - (pan_x + a00 * i + a01 * j);
          dy = mv->dy[0] - (pan_y + a10 * i + a11 * j);
          sum2 += dx * dx + dy * dy;
        }
      }
    }

    stddev2 = sum2/n;
    SCHRO_DEBUG("stddev %f", sqrt(sum2/n));

    if (stddev2 < 1) stddev2 = 1;

    n = 0;
    for(j=0;j<mf->y_num_blocks;j++) {
      for(i=0;i<mf->x_num_blocks;i++) {
        double dx, dy;
        mv = mf->motion_vectors + j*mf->x_num_blocks + i;
        dx = mv->dx[0] - (pan_x + a00 * i + a01 * j);
        dy = mv->dy[0] - (pan_y + a10 * i + a11 * j);
        mv->using_global = (dx * dx + dy * dy < stddev2*16);
        n += mv->using_global;
      }
    }
    SCHRO_DEBUG("using n = %d", n);

    gm->b0 = rint(pan_x*(0.125*(1<<mv_precision)));
    gm->b1 = rint(pan_y*(0.125*(1<<mv_precision)));
    gm->a_exp = 16;
    gm->a00 = rint((1.0 + a00/8) * (1<<(gm->a_exp + mv_precision)));
    gm->a01 = rint(a01/8 * (1<<(gm->a_exp + mv_precision)));
    gm->a10 = rint(a10/8 * (1<<(gm->a_exp + mv_precision)));
    gm->a11 = rint((1.0 + a11/8) * (1<<(gm->a_exp + mv_precision)));
  }

  for(j=0;j<mf->y_num_blocks;j++) {
    for(i=0;i<mf->x_num_blocks;i++) {
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;
      mv->using_global = 1;
      //mv->dx[0] = gm->b0 + ((gm->a00 * (i*8) + gm->a01 * (j*8))>>gm->a_exp) - i*8;
      //mv->dy[0] = gm->b1 + ((gm->a10 * (i*8) + gm->a11 * (j*8))>>gm->a_exp) - j*8;
      mv->dx[0] = 0;
      mv->dy[0] = 0;
    }
  }
}


static void
schro_motion_vector_scan (SchroMotionVector *mv, SchroFrame *frame,
    SchroFrame *ref, int x, int y, int dist)
{
  int i,j;
  int xmin;
  int xmax;
  int ymin;
  int ymax;
  int metric;
  int dx, dy;
  uint32_t metric_array[100];

  dx = mv->dx[0];
  dy = mv->dy[0];
  xmin = MAX(0, x + dx - dist);
  ymin = MAX(0, y + dy - dist);
  xmax = MIN(frame->width - 8, x + dx + dist);
  ymax = MIN(frame->height - 8, y + dy + dist);

  mv->metric = 256*8*8;

  if (xmin > xmax || ymin > ymax) return;

  if (ymax - ymin + 1 <= 100) {
    for(i=xmin;i<xmax;i++){
      oil_sad8x8_8xn_u8 (metric_array,
          frame->components[0].data + x + y*frame->components[0].stride,
          frame->components[0].stride,
          ref->components[0].data + i + ymin*ref->components[0].stride,
          ref->components[0].stride,
          ymax - ymin + 1);
      for(j=ymin;j<=ymax;j++){
        metric = metric_array[j-ymin] + abs(i - x) + abs(j - y);
        if (metric < mv->metric) {
          mv->dx[0] = i - x;
          mv->dy[0] = j - y;
          mv->metric = metric;
        }
      }
    }
  } else {
    SCHRO_ERROR("increase scan limit, please");
    for(j=ymin;j<=ymax;j++){
      for(i=xmin;i<=xmax;i++){

        metric = schro_metric_absdiff_u8 (
            frame->components[0].data + x + y*frame->components[0].stride,
            frame->components[0].stride,
            ref->components[0].data + i + j*ref->components[0].stride,
            ref->components[0].stride, 8, 8);
        metric += abs(i - x) + abs(j - y);
        if (metric < mv->metric) {
          mv->dx[0] = i - x;
          mv->dy[0] = j - y;
          mv->metric = metric;
        }
      }
    }
  }
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

void
schro_motion_field_scan (SchroMotionField *field, SchroParams *params,
    SchroFrame *frame, SchroFrame *ref, int dist)
{
  SchroMotionVector *mv;
  int i;
  int j;

  for(j=0;j<field->y_num_blocks;j++){
    for(i=0;i<field->x_num_blocks;i++){
      mv = field->motion_vectors + j*field->x_num_blocks + i;

      schro_motion_vector_scan (mv, frame, ref, i*params->xbsep_luma,
          j*params->ybsep_luma, dist);
    }
  }
}

void
schro_motion_field_inherit (SchroMotionField *field,
    SchroMotionField *parent)
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
      mv->dx[0] *= 2;
      mv->dy[0] *= 2;
    }
  }
}

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

void
schro_encoder_dc_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int i;
  int j;
  int luma_w, luma_h;
  int chroma_w, chroma_h;
  SchroMotionField *motion_field;
  SchroFrame *orig_frame = me->encoder_frame->filtered_frame;

  motion_field = schro_motion_field_new (params->x_num_blocks,
      params->y_num_blocks);

  luma_w = params->xbsep_luma;
  luma_h = params->xbsep_luma;
  chroma_w = luma_w>>SCHRO_CHROMA_FORMAT_H_SHIFT(params->video_format->chroma_format);
  chroma_h = luma_h>>SCHRO_CHROMA_FORMAT_V_SHIFT(params->video_format->chroma_format);

  for(j=0;j<params->y_num_blocks;j++){
    for(i=0;i<params->x_num_blocks;i++){
      SchroMotionVectorDC *mvdc;
#if 0
      int x,y;
      uint8_t const_data[16];
#endif

      mvdc = (SchroMotionVectorDC *)(motion_field->motion_vectors + j*motion_field->x_num_blocks + i);

      memset(mvdc, 0, sizeof(*mvdc));
      mvdc->pred_mode = 0;
      mvdc->split = 2;
      mvdc->using_global = 0;
      schro_block_average (&mvdc->dc[0], orig_frame->components + 0, i*luma_w, j*luma_h, luma_w, luma_h);
      schro_block_average (&mvdc->dc[1], orig_frame->components + 1, i*chroma_w, j*chroma_h, chroma_w, chroma_h);
      schro_block_average (&mvdc->dc[2], orig_frame->components + 2, i*chroma_w, j*chroma_h, chroma_w, chroma_h);

#if 0
      memset (const_data, mvdc->dc[0] + 128, 16);

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;
      mvdc->metric = schro_metric_absdiff_u8 (
          orig_frame->components[0].data + x + y*orig_frame->components[0].stride,
          orig_frame->components[0].stride,
          const_data, 0, 8, 8);
      mvdc->metric += DC_BIAS;
#else
      mvdc->metric = DC_METRIC*8*8;
#endif
    }
  }

  schro_list_append (me->encoder_frame->motion_field_list, motion_field);
}

int
schro_frame_get_metric (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2)
{
  int metric;

  /* FIXME handle out-of-frame vectors */
  if (x1 < 0 || y1 < 0 || x1+8 > frame1->width ||
      y1+8 > frame1->height) return 64*255;
  if (x2 < 0 || y2 < 0 || x2+8 > frame2->width ||
      y2+8 > frame2->height) return 64*255;

  metric = schro_metric_absdiff_u8 (
      frame1->components[0].data + x1 + y1*frame1->components[0].stride,
      frame1->components[0].stride,
      frame2->components[0].data + x2 + y2*frame2->components[0].stride,
      frame2->components[0].stride, 8, 8);
  //metric += abs(x1 - x2) + abs(y1 - y2);

  return metric;
}

#if 0
void
schro_motionest_rough_scan_nohint (SchroMotionEst *me, int shift, int ref,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroParams *params = me->params;
  int i;
  int j;
  int skip;

  scan.frame = get_downsampled (me->encoder_frame, shift);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

  schro_motion_field_set (mf, 0, 1<<ref);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  skip = 1<<shift;
  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      int dx, dy;

      scan.x = (i>>shift) * params->xbsep_luma;
      scan.y = (j>>shift) * params->ybsep_luma;
      scan.block_width = MIN(scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN(scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, 0, 0, distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[ref] = 0 << shift;
        mv->dy[ref] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#if 0
      /* this code skips blocks that are off the edge.  Instead, we
       * scan smaller block sizes */
      if (scan.x + scan.block_width >= scan.ref_frame->width ||
          scan.y + scan.block_height >= scan.ref_frame->height) {
        mv->dx[ref] = 0 << shift;
        mv->dy[ref] = 0 << shift;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }
#endif

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      dx <<= shift;
      dy <<= shift;

      mv->dx[ref] = dx;
      mv->dy[ref] = dy;

      me->hier_score += (mv->metric>10*12*12);
    }
  }

  me->downsampled_mf[ref][shift] = mf;
}

void
schro_motionest_rough_scan_hint (SchroMotionEst *me, int shift, int ref,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroMotionField *hint_mf;
  SchroParams *params = me->params;
  SchroMotionVector zero_mv;
  int i;
  int j;
  int skip;
  unsigned int hint_mask;

  scan.frame = get_downsampled (me->encoder_frame, shift);
  scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame[ref], shift);

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  hint_mf = me->downsampled_mf[ref][shift+1];

  schro_motion_field_set (mf, 0, 1<<ref);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  memset (&zero_mv, 0, sizeof(zero_mv));

  hint_mask = ~((1<<(shift + 1))-1);
  skip = 1<<shift;
  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      SchroFrameData orig;
      SchroFrameData ref_data;
#define LIST_LENGTH 10
      SchroMotionVector *hint_mv[LIST_LENGTH];
      int m;
      int n = 0;
      int dx, dy;
      int min_m;
      int min_metric;

      schro_frame_get_subdata (scan.frame, &orig,
          0, i*me->params->xbsep_luma >> shift,
          j*me->params->ybsep_luma >> shift);

      /* always test the zero vector */
      hint_mv[n] = &zero_mv;
      n++;

      /* inherit from nearby parents */
      /* This overly clever bit of code checks the parents of the diagonal
       * neighbors, which corresponds to the nearest parents. */
      for(m=0;m<4;m++) {
        int l = (i-1 + 2*(m&1))&hint_mask;
        int k = (j-1 + (m&2))&hint_mask;
        if (l >= 0 && l < params->x_num_blocks &&
            k >= 0 && k < params->y_num_blocks) {
          hint_mv[n] = motion_field_get (hint_mf, l, k);
          n++;
        }
      }

      /* inherit from neighbors (only towards SE) */
      if (i > 0) {
        hint_mv[n] = motion_field_get (mf, i-1, j);
        n++;
      }
      if (j > 0) {
        hint_mv[n] = motion_field_get (mf, i, j - 1);
        n++;
      }
      if (i > 0 && j > 0) {
        hint_mv[n] = motion_field_get (mf, i - 1, j - 1);
        n++;
      }

      SCHRO_ASSERT(n <= LIST_LENGTH);

      min_m = 0;
      min_metric = SCHRO_METRIC_INVALID;
      for(m = 0; m < n; m++) {
        int metric;
        int width, height;

        dx = hint_mv[m]->dx[ref];
        dy = hint_mv[m]->dy[ref];

        schro_frame_get_subdata (scan.ref_frame,
            &ref_data, 0,
            (i*me->params->xbsep_luma + dx) >> shift,
            (j*me->params->ybsep_luma + dy) >> shift);

        width = MIN(me->params->xbsep_luma, orig.width);
        height = MIN(me->params->ybsep_luma, orig.height);
        if (width == 0 || height == 0) continue;
        if (ref_data.width < width || ref_data.height < height) continue;

        metric = schro_metric_get (&orig, &ref_data, width, height);

        if (metric < min_metric) {
          min_metric = metric;
          min_m = m;
        }
      }

      dx = hint_mv[min_m]->dx[ref] >> shift;
      dy = hint_mv[min_m]->dy[ref] >> shift;

      scan.x = (i>>shift) * params->xbsep_luma;
      scan.y = (j>>shift) * params->ybsep_luma;
      scan.block_width = MIN(scan.frame->width - scan.x, params->xbsep_luma);
      scan.block_height = MIN(scan.frame->height - scan.y, params->ybsep_luma);
      schro_metric_scan_setup (&scan, dx, dy, distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[ref] = 0;
        mv->dy[ref] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      dx <<= shift;
      dy <<= shift;

      mv->dx[ref] = dx;
      mv->dy[ref] = dy;
    }
  }

  me->downsampled_mf[ref][shift] = mf;
}
#endif

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
  chroma_w = params->xbsep_luma>>SCHRO_CHROMA_FORMAT_H_SHIFT(params->video_format->chroma_format);
  chroma_h = params->ybsep_luma>>SCHRO_CHROMA_FORMAT_V_SHIFT(params->video_format->chroma_format);
  schro_block_average (&mvdc->dc[1], frame->components + 1,
      i * chroma_w, j * chroma_h, 4 * chroma_w, 4 * chroma_h);
  schro_block_average (&mvdc->dc[2], frame->components + 2,
      i * chroma_w, j * chroma_h, 4 * chroma_w, 4 * chroma_h);

  mvdc->metric = metric/16;
  block->error = metric;
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
motion_field_splat_2x2 (SchroMotionField *mf, int i, int j)
{
  SchroMotionVector *mv;

  mv = motion_field_get (mf, i, j);
  mv[1] = mv[0];
  memcpy (motion_field_get (mf, i, j+1), mv, 2*sizeof(*mv));
}
#endif

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

  //hint_mf = me->downsampled_mf[ref][1];
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


#define MAGIC_SUPERBLOCK_METRIC 5
#define MAGIC_BLOCK_METRIC 50

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
      double min_score1;

#define TRYBLOCK \
      score = tryblock.entropy + me->lambda * tryblock.error; \
      if (tryblock.valid && score < min_score) { \
        memcpy (&block, &tryblock, sizeof(block)); \
        min_score = score; \
      }

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
        min_score1 = min_score;
        schro_motionest_block_scan_one (me, 0, 4, &tryblock, i, j);
        TRYBLOCK
        //schro_dump(SCHRO_DUMP_MOTIONEST, "%g %g %g\n", min_score1, min_score, score);
        if (params->num_refs > 1) {
          schro_motionest_block_scan_one (me, 1, 4, &tryblock, i, j);
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

  schro_frame_get_subdata (get_downsampled (me->encoder_frame, 0), &orig,
      0, i*me->params->xbsep_luma, j*me->params->ybsep_luma);

  width = MIN(4*me->params->xbsep_luma, orig.width);
  height = MIN(4*me->params->ybsep_luma, orig.height);

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

    if (i*me->params->xbsep_luma + mv->dx[ref] < 0 ||
        j*me->params->ybsep_luma + mv->dy[ref] < 0) {
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (ref_frame, &ref_data,
        0, i*me->params->xbsep_luma + mv->dx[ref],
        j*me->params->ybsep_luma + mv->dy[ref]);

    if (ref_data.width < width || ref_data.height < height) {
      return SCHRO_METRIC_INVALID_2;
    }

    return schro_metric_get (&orig, &ref_data, width, height);
  }

  if (mv->pred_mode == 3) {
    SchroFrameData ref0_data;
    SchroFrameData ref1_data;

    if (i*me->params->xbsep_luma + mv->dx[0] < 0 ||
        j*me->params->ybsep_luma + mv->dy[0] < 0 ||
        i*me->params->xbsep_luma + mv->dx[1] < 0 ||
        j*me->params->ybsep_luma + mv->dy[1] < 0) {
      return SCHRO_METRIC_INVALID_2;
    }

    schro_frame_get_subdata (get_downsampled (me->encoder_frame->ref_frame[0], 0),
        &ref0_data, 0, i*me->params->xbsep_luma + mv->dx[0],
        j*me->params->ybsep_luma + mv->dy[0]);
    schro_frame_get_subdata (get_downsampled (me->encoder_frame->ref_frame[1], 0),
        &ref1_data, 0, i*me->params->xbsep_luma + mv->dx[1],
        j*me->params->ybsep_luma + mv->dy[1]);

    if (ref0_data.width < width || ref0_data.height < height ||
        ref1_data.width < width || ref1_data.height < height) {
      return SCHRO_METRIC_INVALID_2;
    }

    return schro_metric_get_biref (&orig, &ref0_data, 1, &ref1_data, 1, 1, width, height);
  }

  SCHRO_ASSERT(0);

  return SCHRO_METRIC_INVALID_2;
}

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

