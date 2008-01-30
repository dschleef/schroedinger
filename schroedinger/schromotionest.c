
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

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

void schro_encoder_bigblock_estimation (SchroMotionEst *me);
void schro_encoder_bigblock2_estimation (SchroMotionEst *me);
void schro_encoder_hierarchical_estimation (SchroMotionEst *me);
void schro_encoder_zero_estimation (SchroMotionEst *me);
void schro_encoder_dc_estimation (SchroMotionEst *me);
void schro_encoder_fullscan_estimation (SchroMotionEst *me);
void schro_motion_field_set (SchroMotionField *field, int split, int pred_mode);
void schro_motion_global_metric (SchroMotionField *mf, SchroFrame *frame,
    SchroFrame *ref);
void schro_motionest_rough_scan_nohint (SchroMotionEst *me,
    int shift, int ref, int distance);
void schro_motionest_rough_scan_hint (SchroMotionEst *me,
    int shift, int ref, int distance);

void schro_motion_merge (SchroMotion *motion, SchroList *_list);
void schro_motion_cleanup (SchroMotion *motion, int x, int y);
void schro_motion_combine (SchroMotion *motion);
#if 0
void schro_motion_predict_subpixel (SchroMotion *motion, SchroFrame *frame,
    SchroMotionField *mf);
#endif
void schro_motion_calculate_stats (SchroMotion *motion, SchroEncoderFrame *frame);

static void motion_field_splat_4x4 (SchroMotionField *mf, int i, int j);
#if 0
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

  me->src0 = frame->ref_frame0->reconstructed_frame;
  me->downsampled_src0[0] = frame->ref_frame0->filtered_frame;
  me->downsampled_src0[1] = frame->ref_frame0->downsampled_frames[0];
  me->downsampled_src0[2] = frame->ref_frame0->downsampled_frames[1];
  me->downsampled_src0[3] = frame->ref_frame0->downsampled_frames[2];
  me->downsampled_src0[4] = frame->ref_frame0->downsampled_frames[3];

  if (me->params->num_refs > 1) {
    me->src1 = frame->ref_frame1->reconstructed_frame;
    me->downsampled_src1[0] = frame->ref_frame1->filtered_frame;
    me->downsampled_src1[1] = frame->ref_frame1->downsampled_frames[0];
    me->downsampled_src1[2] = frame->ref_frame1->downsampled_frames[1];
    me->downsampled_src1[3] = frame->ref_frame1->downsampled_frames[2];
    me->downsampled_src1[4] = frame->ref_frame1->downsampled_frames[3];
  }

  n = params->x_num_blocks * params->y_num_blocks / 16;
  me->sblocks = schro_malloc0(sizeof(SchroBlock)*n);



  return me;
}

void
schro_motionest_free (SchroMotionEst *me)
{
  int ref;
  int i;

  for(ref=0;ref<2;ref++){
    for(i=0;i<5;i++){
      if (me->downsampled_mf[ref][i]) {
        schro_motion_field_free (me->downsampled_mf[ref][i]);
      }
    }
  }

  schro_free (me->sblocks);

  schro_free (me);
}


void
schro_encoder_motion_predict (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroMotionEst *me;
  int n;

  SCHRO_ASSERT(params->x_num_blocks != 0);
  SCHRO_ASSERT(params->y_num_blocks != 0);
  SCHRO_ASSERT(params->num_refs > 0);

  me = schro_motionest_new (frame);

  frame->motion = schro_motion_new (params,
      frame->ref_frame0->reconstructed_frame,
      (params->num_refs > 1) ? frame->ref_frame1->reconstructed_frame : NULL);
  me->motion = frame->motion;

  frame->motion_field_list = schro_list_new_full ((SchroListFreeFunc)schro_motion_field_free, NULL);
  n = 0;

  if (frame->encoder->enable_bigblock_estimation) {
    int ref;

    for(ref=0;ref<params->num_refs;ref++){
      schro_motionest_rough_scan_nohint (me, 3, ref, 12);
      schro_motionest_rough_scan_hint (me, 2, ref, 2);
      schro_motionest_rough_scan_hint (me, 1, ref, 2);
    }

    //schro_encoder_bigblock_estimation (me);
    schro_encoder_bigblock2_estimation (me);
  } else {
    schro_encoder_dc_estimation (me);

    if (frame->encoder->enable_phasecorr_estimation) {
      schro_encoder_phasecorr_estimation (me);
    }

    if (frame->encoder->enable_hierarchical_estimation) {
      schro_encoder_hierarchical_estimation (me);
    }

    if (frame->encoder->enable_zero_estimation) {
      schro_encoder_zero_estimation (me);
    }

    if (frame->encoder->enable_fullscan_estimation) {
      schro_encoder_fullscan_estimation (me);
    }

    if (params->have_global_motion) {
      schro_encoder_global_estimation (me);
    }

    schro_motion_merge (frame->motion, frame->motion_field_list);
    schro_motion_cleanup (frame->motion,
        (params->video_format->width + params->xbsep_luma - 1)/params->xbsep_luma,
        (params->video_format->height + params->ybsep_luma - 1)/params->ybsep_luma);

    schro_motion_combine (frame->motion);
  }

  schro_motion_calculate_stats (frame->motion, frame);
  frame->estimated_mc_bits = schro_motion_estimate_entropy (frame->motion);

  schro_list_free (frame->motion_field_list);

  schro_motionest_free (me);
}

void
schro_motion_merge (SchroMotion *motion, SchroList *list)
{
  int i,j,k;
  SchroMotionVector *mv;
  SchroMotionVector *mvk;
  SchroMotionField **fields = (SchroMotionField **)list->members;
  int n = schro_list_get_size (list);

  for(k=0;k<n;k++){
    SCHRO_ASSERT(fields[k]->x_num_blocks == motion->params->x_num_blocks);
    SCHRO_ASSERT(fields[k]->y_num_blocks == motion->params->y_num_blocks);
  }

  for(j=0;j<motion->params->y_num_blocks;j++){
    for(i=0;i<motion->params->x_num_blocks;i++){
      mv = SCHRO_MOTION_GET_BLOCK(motion,i,j);

      mvk = &fields[0]->motion_vectors[j*motion->params->x_num_blocks + i];
      *mv = *mvk;
      for(k=1;k<n;k++){
        mvk = &fields[k]->motion_vectors[j*motion->params->x_num_blocks + i];
        if (mvk->metric < mv->metric) {
          *mv = *mvk;
        }
      }
      if (mv->pred_mode == 2) {
        mv->dx[1] = mv->dx[0];
        mv->dy[1] = mv->dy[0];
      }
      SCHRO_ASSERT (!(mv->pred_mode == 0 && mv->using_global));
    }
  }
}

void
schro_motion_cleanup (SchroMotion *motion, int x_blocks, int y_blocks)
{
  int i,j;
  SchroMotionVector *mv;

  SCHRO_DEBUG("motion field cleanup %dx%d to %dx%d",
      x_blocks, y_blocks,
      motion->params->x_num_blocks, motion->params->y_num_blocks);
  if (x_blocks < motion->params->x_num_blocks) {
    for(j=0;j<motion->params->y_num_blocks;j++){
      for(i=x_blocks;i<motion->params->x_num_blocks;i++){
        mv = SCHRO_MOTION_GET_BLOCK(motion,i,j);
        *mv = *SCHRO_MOTION_GET_BLOCK(motion,i-1,j);
      }
    }
  }
  if (y_blocks < motion->params->y_num_blocks) {
    for(j=y_blocks;j<motion->params->y_num_blocks;j++){
      for(i=0;i<motion->params->x_num_blocks;i++){
        mv = SCHRO_MOTION_GET_BLOCK(motion,i,j);
        *mv = *SCHRO_MOTION_GET_BLOCK(motion,i,j-1);
      }
    }
  }
}

void
schro_motion_combine (SchroMotion *motion)
{
  int i,j,k,l;
  SchroMotionVector *mv;
  SchroMotionVector *mv2;

  for(j=0;j<motion->params->y_num_blocks;j+=4){
    for(i=0;i<motion->params->x_num_blocks;i+=4){
      mv = SCHRO_MOTION_GET_BLOCK(motion,i,j);
      if (mv->pred_mode == 0) {
#if 0
        int dc[3] = { 0, 0, 0 };
        for(k=0;k<4;k++){
          for(l=0;l<4;l++){
            if (mv[k*mf->x_num_blocks + l].pred_mode != 0) goto next_mb;
            dc[0] += mv[k*mf->x_num_blocks + l].u.dc[0];
            dc[1] += mv[k*mf->x_num_blocks + l].u.dc[1];
            dc[2] += mv[k*mf->x_num_blocks + l].u.dc[2];
          }
        }
        mv->split = 0;
        mv->u.dc[0] = (dc[0] + 8)>>4;
        mv->u.dc[1] = (dc[1] + 8)>>4;
        mv->u.dc[2] = (dc[2] + 8)>>4;
#else
        continue;
#endif
      } else if (mv->using_global) {
        for(k=0;k<4;k++){
          for(l=0;l<4;l++){
            mv2 = SCHRO_MOTION_GET_BLOCK(motion,i+k,j+l);
            if (!mv2->using_global ||
                mv2->pred_mode != mv->pred_mode) {
              goto next_mb;
            }
          }
        }
        mv->split = 0;
      } else {
        goto next_mb;
      }

      for(k=0;k<4;k++){
        for(l=0;l<4;l++){
          mv2 = SCHRO_MOTION_GET_BLOCK(motion,i+k,j+l);
          if (mv != mv2) {
            *mv2 = *mv;
          }
        }
      }
next_mb:
      do {} while (0);
    }
  }
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
      if (mv->pred_mode & 1) {
        mv->dx[0] <<= n;
        mv->dy[0] <<= n;
      }
      if (mv->pred_mode & 2) {
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
    mf_orig = me->downsampled_mf[i][0];
    mf = schro_motion_field_new (mf_orig->x_num_blocks, mf_orig->y_num_blocks);

    memcpy (mf->motion_vectors, mf_orig->motion_vectors,
        sizeof(SchroMotionVector)*mf->x_num_blocks*mf->y_num_blocks);
    schro_motion_field_global_estimation (mf,
        &me->encoder_frame->params.global_motion[i],
        params->mv_precision);
    if (i == 0) {
      schro_motion_global_metric (mf,
          me->encoder_frame->filtered_frame,
          me->encoder_frame->ref_frame0->filtered_frame);
    } else {
      schro_motion_global_metric (mf, me->encoder_frame->filtered_frame,
          me->encoder_frame->ref_frame1->filtered_frame);
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
    return SCHRO_METRIC_INVALID;
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
  chroma_w = luma_w>>params->video_format->chroma_h_shift;
  chroma_h = luma_h>>params->video_format->chroma_v_shift;

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

void
schro_encoder_hierarchical_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int ref;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref;
  SchroFrame *downsampled_frame;
  int shift;
  SchroMotionField *mf;
  SchroMotionField *parent_mf = NULL;
  SchroMotionField *mf1 = NULL;
  SchroMotionField *mf2 = NULL;

  for(ref=0;ref<params->num_refs;ref++){

    shift = 3;
    if (ref == 0) {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame0,shift);
    } else {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame1,shift);
    }
    downsampled_frame = get_downsampled(me->encoder_frame,shift);

    x_blocks = params->x_num_blocks>>shift;
    y_blocks = params->y_num_blocks>>shift;
    parent_mf = schro_motion_field_new (x_blocks, y_blocks);

    schro_motion_field_set (parent_mf, 2, 1<<ref);
    schro_motion_field_scan (parent_mf, params, downsampled_frame, downsampled_ref,
        12);

    for(shift=2;shift>=0;shift--) {
      int i,j;
      SchroMotionVector *mv;

      x_blocks = params->x_num_blocks>>shift;
      y_blocks = params->y_num_blocks>>shift;

      mf = schro_motion_field_new (x_blocks, y_blocks);

      if (ref == 0) {
        downsampled_ref = get_downsampled(me->encoder_frame->ref_frame0,shift);
      } else {
        downsampled_ref = get_downsampled(me->encoder_frame->ref_frame1,shift);
      }
      downsampled_frame = get_downsampled(me->encoder_frame,shift);

      for(j=0;j<mf->y_num_blocks;j++){
        for(i=0;i<mf->x_num_blocks;i++){
#define LIST_LENGTH 20
          int list_x[LIST_LENGTH], list_y[LIST_LENGTH];
          int n = 0;
          int l, k;
          int x, y;
          int metric;

          /* always test the zero vector */
          list_x[n] = 0;
          list_y[n] = 0;
          n++;

          /* inherit from parent */
          for(k=0;k<4;k++){
            int l = (i-1+2*(k&1))>>1;
            int m = (j-1+(k&2))>>1;
            if (l >= 0 && l < parent_mf->x_num_blocks &&
                m >= 0 && m < parent_mf->y_num_blocks) {
              mv = motion_field_get(parent_mf, l, m);
              list_x[n] = mv->dx[0] * 2;
              list_y[n] = mv->dy[0] * 2;
              n++;
            }
          }

          /* inherit from neighbors (only towards SE) */
          if (i > 0) {
            mv = motion_field_get (mf, i-1, j);
            list_x[n] = mv->dx[0];
            list_y[n] = mv->dy[0];
            n++;
          }
          if (j > 0) {
            mv = motion_field_get (mf, i, j-1);
            list_x[n] = mv->dx[0];
            list_y[n] = mv->dy[0];
            n++;
          }
          if (i > 0 && j > 0) {
            mv = motion_field_get (mf, i-1, j-1);
            list_x[n] = mv->dx[0];
            list_y[n] = mv->dy[0];
            n++;
          }
          
          SCHRO_ASSERT(n<=LIST_LENGTH);
          metric = schro_frame_get_metric (downsampled_frame,
              i * 8, j * 8, downsampled_ref, i*8 + list_x[0],
              j*8 + list_y[0]);
          x = list_x[0];
          y = list_y[0];
          for (l = 1; l < n; l++) {
            int m;

            m = schro_frame_get_metric (downsampled_frame,
                i * 8, j * 8, downsampled_ref, i*8 + list_x[l],
                j*8 + list_y[l]);
            if (m < metric) {
              metric = m;
              x = list_x[l];
              y = list_y[l];
            }
          }

          mv = motion_field_get (mf, i, j);
          mv->dx[0] = x;
          mv->dy[0] = y;
          mv->metric = metric;
          mv->split = 2;
          mv->pred_mode = (1<<ref);
        }
      }

      schro_motion_field_scan (mf, params, downsampled_frame, downsampled_ref, 4);

      schro_motion_field_free (parent_mf);
      parent_mf = mf;
    }

    schro_motion_field_lshift (parent_mf, params->mv_precision);
#if 0
    schro_motion_predict_subpixel (frame->motion, frame->filtered_frame,
        parent_mf);
#endif
    schro_list_append (me->encoder_frame->motion_field_list, parent_mf);
    
    if (ref == 0) {
      mf1 = parent_mf;
    } else {
      mf2 = parent_mf;
    }
  }

  if (params->num_refs == 2) {
    int i,j;
    SchroMotionVector *mv;
    SchroMotionVector *mv1;
    SchroMotionVector *mv2;

    mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
    for(j=0;j<mf->y_num_blocks;j++){
      for(i=0;i<mf->x_num_blocks;i++){
        mv = motion_field_get (mf, i, j);
        mv1 = motion_field_get (mf1, i, j);
        mv2 = motion_field_get (mf2, i, j);

        *mv = *mv1;
        mv->pred_mode = 3;
        mv->dx[1] = mv2->dx[0];
        mv->dy[1] = mv2->dy[0];
        if (mv1->metric < BIDIR_LIMIT && mv2->metric < BIDIR_LIMIT) {
          mv->metric = MIN(mv1->metric, mv2->metric) - 1;
        } else {
          mv->metric = 32000;
        }
      }
    }
    schro_list_append (me->encoder_frame->motion_field_list, mf);
  }
}

void
schro_encoder_zero_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int ref;
  int x_blocks;
  int y_blocks;
  SchroFrame *downsampled_ref;
  SchroFrame *downsampled_frame;
  SchroMotionField *mf;

  for(ref=0;ref<params->num_refs;ref++){
    int i,j;
    SchroMotionVector *mv;

    x_blocks = params->x_num_blocks;
    y_blocks = params->y_num_blocks;

    mf = schro_motion_field_new (x_blocks, y_blocks);

    if (ref == 0) {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame0,0);
    } else {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame1,0);
    }
    downsampled_frame = get_downsampled(me->encoder_frame,0);

    for(j=0;j<mf->y_num_blocks;j++){
      for(i=0;i<mf->x_num_blocks;i++){
        int metric;
        
        metric = schro_frame_get_metric (downsampled_frame,
            i * params->xbsep_luma, j * params->ybsep_luma,
            downsampled_ref, i*params->xbsep_luma, j*params->ybsep_luma);

        mv = motion_field_get (mf, i, j);
        mv->dx[0] = 0;
        mv->dy[0] = 0;
        mv->metric = metric;
        mv->split = 2;
        mv->pred_mode = (1<<ref);
      }
    }

    schro_motion_field_lshift (mf, params->mv_precision);
    schro_list_append (me->encoder_frame->motion_field_list, mf);
  }
}

void
schro_encoder_fullscan_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int ref;
  SchroFrame *downsampled_ref;
  SchroFrame *downsampled_frame;
  SchroMotionField *mf;

  for(ref=0;ref<params->num_refs;ref++){
    mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

    if (ref == 0) {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame0,0);
    } else {
      downsampled_ref = get_downsampled(me->encoder_frame->ref_frame1,0);
    }
    downsampled_frame = get_downsampled(me->encoder_frame,0);

    schro_motion_field_set (mf, 2, (1<<ref));
    schro_motion_field_scan (mf, params, downsampled_frame,
        downsampled_ref, 20);

    schro_motion_field_lshift (mf, params->mv_precision);
    schro_list_append (me->encoder_frame->motion_field_list, mf);
  }
}


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
  if (ref == 0) {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, shift);
  } else {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, shift);
  }

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
      schro_metric_scan_setup (&scan, 0, 0, distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[0] = 0 << shift;
        mv->dy[0] = 0 << shift;
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

void
schro_motionest_rough_scan_hint (SchroMotionEst *me, int shift, int ref,
    int distance)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  SchroParams *params = me->params;
  int i;
  int j;
  int skip;
  unsigned int hint_mask;

  scan.frame = get_downsampled (me->encoder_frame, shift);
  if (ref == 0) {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, shift);
  } else {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, shift);
  }

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  hint_mf = me->downsampled_mf[ref][shift+1];

  schro_motion_field_set (mf, 0, 1<<ref);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  hint_mask = ~((1<<(shift + 1))-1);
  skip = 1<<shift;
  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      int dx, dy;

      hint_mv = motion_field_get (hint_mf, i&hint_mask, j&hint_mask);
      dx = hint_mv->dx[0]>>shift;
      dy = hint_mv->dy[0]>>shift;

      scan.x = (i>>shift) * params->xbsep_luma;
      scan.y = (j>>shift) * params->ybsep_luma;
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

      mv->dx[ref] = dx;
      mv->dy[ref] = dy;
    }
  }

  me->downsampled_mf[ref][shift] = mf;
}

static void
schro_motionest_superblock_scan_one (SchroMotionEst *me, int ref, int distance,
    SchroBlock *block, int i, int j)
{
  SchroParams *params = me->params;
  SchroMotionVector *mv;
  SchroMetricScan scan2;
  SchroMotionField *hint_mf;
  SchroMotionVector *hint_mv;
  int dx, dy;

  scan2.frame = get_downsampled (me->encoder_frame, 0);
  if (ref == 0) {
    scan2.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, 0);
  } else {
    scan2.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, 0);
  }

  hint_mf = me->downsampled_mf[ref][2];

  scan2.block_width = 4*params->xbsep_luma;
  scan2.block_height = 4*params->ybsep_luma;
  scan2.gravity_scale = 0;
  scan2.gravity_x = 0;
  scan2.gravity_y = 0;

  mv = &block->mv[0][0];
  hint_mv = motion_field_get (hint_mf, i, j);
  
  dx = hint_mv->dx[0];
  dy = hint_mv->dy[0];

  scan2.x = i * params->xbsep_luma;
  scan2.y = j * params->ybsep_luma;
  schro_metric_scan_setup (&scan2, dx, dy, distance);
  if (scan2.scan_width <= 0 || scan2.scan_height <= 0) {
    mv->dx[ref] = 0;
    mv->dy[ref] = 0;
    mv->metric = SCHRO_METRIC_INVALID;
    return;
  }

  schro_metric_scan_do_scan (&scan2);
  block->error = schro_metric_scan_get_min (&scan2, &dx, &dy);
  mv->metric = block->error/16;

  mv->split = 0;
  mv->pred_mode = 1<<ref;
  mv->using_global = 0;
  mv->dx[ref] = dx;
  mv->dy[ref] = dy;

  schro_block_fixup (block);
  block->entropy = schro_motion_superblock_try_estimate_entropy (me->motion,
      i, j, block);
}

static SchroMotionField *
schro_motionest_superblock_scan (SchroMotionEst *me, int ref, int distance)
{
  SchroMotionField *mf;
  SchroMotionVector *mv;
  SchroParams *params = me->params;
  int i;
  int j;
  int skip = 4;
  SchroBlock block;

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
  schro_motion_field_set (mf, 0, 1<<ref);

  for(j=0;j<params->y_num_blocks;j+=skip){
    for(i=0;i<params->x_num_blocks;i+=skip){
      mv = motion_field_get (mf, i, j);

      schro_motionest_superblock_scan_one (me, ref, distance, &block,
          i, j);

      *mv = block.mv[0][0];
      motion_field_splat_4x4 (mf, i, j);
    }
  }

  return mf;
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
  chroma_w = params->xbsep_luma>>params->video_format->chroma_h_shift;
  chroma_h = params->xbsep_luma>>params->video_format->chroma_v_shift;
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
}

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
  if (ref == 0) {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, 0);
  } else {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, 0);
  }

  hint_mf = me->downsampled_mf[ref][1];

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  for(jj=0;jj<4;jj++){
    for(ii=0;ii<4;ii++){
      mv = &block->mv[jj][ii];
      hint_mv = motion_field_get (hint_mf, i + (ii&2), j + (jj&2));
      
      dx = hint_mv->dx[0];
      dy = hint_mv->dy[0];

      scan.x = (i + ii) * params->xbsep_luma;
      scan.y = (j + jj) * params->ybsep_luma;
      schro_metric_scan_setup (&scan, dx, dy, distance);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[ref] = 0;
        mv->dy[ref] = 0;
        mv->metric = SCHRO_METRIC_INVALID;
        block->error += mv->metric;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);
      block->error += mv->metric;

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

static SchroMotionField *
schro_motionest_block_scan (SchroMotionEst *me, SchroMotionField *mask_mf,
    int ref)
{
  SchroMetricScan scan;
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroMotionVector *hint_mv;
  SchroMotionField *hint_mf;
  SchroMotionVector *mask_mv;
  SchroParams *params = me->params;
  int i;
  int j;
  int distance = 12;

  scan.frame = get_downsampled (me->encoder_frame, 0);
  if (ref == 0) {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, 0);
  } else {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, 0);
  }

  hint_mf = me->downsampled_mf[ref][2];
  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

  schro_motion_field_set (mf, 2, 1<<ref);

  scan.block_width = params->xbsep_luma;
  scan.block_height = params->ybsep_luma;
  scan.gravity_scale = 0;
  scan.gravity_x = 0;
  scan.gravity_y = 0;

  for(j=0;j<params->y_num_blocks;j++){
    for(i=0;i<params->x_num_blocks;i++){
      int dx, dy;

      mask_mv = motion_field_get (mask_mf, i&(~3), j&(~3));
      /* FIXME */
      //if (mask_mv->metric != SCHRO_METRIC_INVALID) continue;

      hint_mv = motion_field_get (hint_mf, i, j);

      scan.x = i * params->xbsep_luma;
      scan.y = j * params->ybsep_luma;
      schro_metric_scan_setup (&scan, hint_mv->dx[0], hint_mv->dy[0],
          distance);

      mv = motion_field_get (mf, i, j);
      if (scan.scan_width <= 0 || scan.scan_height <= 0) {
        mv->dx[0] = 0;
        mv->dy[0] = 0;
        mv->split = 2;
        mv->metric = SCHRO_METRIC_INVALID;
        continue;
      }

      schro_metric_scan_do_scan (&scan);
      mv->metric = schro_metric_scan_get_min (&scan, &dx, &dy);

      mv->dx[0] = dx;
      mv->dy[0] = dy;

      mv->split = 2;
      mv->pred_mode = 1<<ref;
      if (mv->metric > (scan.block_width * scan.block_height)*50) {
        mv->metric = SCHRO_METRIC_INVALID;
      }
    }
  }

  return mf;
}

static void
schro_encoder_dc_predict (SchroEncoderFrame *frame, int i, int j)
{
  SchroParams *params = &frame->params;
  int luma_w, luma_h;
  int chroma_w, chroma_h;
  SchroFrame *orig_frame = frame->filtered_frame;
  SchroMotionVectorDC *mvdc;

  luma_w = params->xbsep_luma;
  luma_h = params->xbsep_luma;
  chroma_w = luma_w>>params->video_format->chroma_h_shift;
  chroma_h = luma_h>>params->video_format->chroma_v_shift;

  mvdc = SCHRO_MOTION_GET_DC_BLOCK (frame->motion, i, j);

  memset(mvdc, 0, sizeof(*mvdc));
  mvdc->pred_mode = 0;
  mvdc->split = 2;
  mvdc->using_global = 0;
  schro_block_average (&mvdc->dc[0], orig_frame->components + 0,
      i*luma_w, j*luma_h, luma_w, luma_h);
  schro_block_average (&mvdc->dc[1], orig_frame->components + 1,
      i*chroma_w, j*chroma_h, chroma_w, chroma_h);
  schro_block_average (&mvdc->dc[2], orig_frame->components + 2,
      i*chroma_w, j*chroma_h, chroma_w, chroma_h);

  mvdc->metric = DC_METRIC*8*8;
}

#define MAGIC_SUPERBLOCK_METRIC 5
#define MAGIC_BLOCK_METRIC 50

void
schro_encoder_bigblock2_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  int i,j;
  int total_error = 0;

  me->lambda = 0.1;

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      SchroBlock blocks[10];
      int n;
      double score;
      int k;
      double min_score;
      int min_k;

      memset (blocks, 0, sizeof(blocks));

      n = 0;
      schro_motionest_superblock_scan_one (me, 0, 4, blocks + n, i, j);
      n++;
      schro_motionest_superblock_predicted (me, 0, blocks + n, i, j);
      n++;
      schro_motionest_block_scan_one (me, 0, 12, blocks + n, i, j);
      n++;
      schro_motionest_superblock_dc (me, blocks + n, i, j);
      n++;
      schro_motionest_superblock_dc_predicted (me, blocks + n, i, j);
      n++;
      if (params->num_refs > 1) {
        schro_motionest_superblock_scan_one (me, 1, 4, blocks + n, i, j);
        n++;
        schro_motionest_superblock_predicted (me, 1, blocks + n, i, j);
        n++;
        schro_motionest_superblock_biref_zero (me, blocks + n, i, j);
        n++;
        schro_motionest_block_scan_one (me, 1, 12, blocks + n, i, j);
        n++;
      }

      min_k = 0;
      min_score = blocks[0].entropy + me->lambda * blocks[0].error;
      for(k=0;k<n;k++){
        score = blocks[k].entropy + me->lambda * blocks[k].error;
        if (score < min_score) {
          min_k = k;
          min_score = score;
        }
      }
      schro_block_fixup (blocks + min_k);
      schro_motion_copy_to (me->motion, i, j, blocks + min_k);
      
      total_error += blocks[min_k].error;
    }
  }

  me->encoder_frame->mc_error = total_error;
}

void
schro_encoder_bigblock_estimation (SchroMotionEst *me)
{
  SchroMotionField *mf1;
  SchroMotionField *mf2 = NULL;
  SchroMotionField *mf3;
  SchroMotionField *mf4 = NULL;
  SchroMotionVector *mv;
  SchroMotionVector *mv1;
  SchroMotionVector *mv2;
  SchroParams *params = me->params;
  int i,j;
  int k,l;
  int n;

  n = params->xbsep_luma * params->ybsep_luma;

  mf1 = schro_motionest_superblock_scan (me, 0, 4);
  schro_motion_field_lshift (mf1, params->mv_precision);

  mf3 = schro_motionest_block_scan (me, mf1, 0);

  if (me->encoder_frame->num_refs > 1) {
    mf2 = schro_motionest_superblock_scan (me, 1, 4);
    schro_motion_field_lshift (mf2, params->mv_precision);

    mf4 = schro_motionest_block_scan (me, mf2, 1);
  }

  for(j=0;j<params->y_num_blocks;j+=4){
    for(i=0;i<params->x_num_blocks;i+=4){
      mv = SCHRO_MOTION_GET_BLOCK (me->motion, i, j);

      if (me->encoder_frame->num_refs == 1) {
        mv1 = motion_field_get (mf1, i, j);
        if (mv1->metric < n*MAGIC_SUPERBLOCK_METRIC) {
          *mv = *mv1;
          mv->pred_mode = 1;
          schro_motion_splat_4x4 (me->motion, i, j);
          continue;
        }
      } else {
        int pred_mode = 0;

        mv1 = motion_field_get (mf1, i, j);
        mv2 = motion_field_get (mf2, i, j);

        if (mv1->metric < n*MAGIC_SUPERBLOCK_METRIC) pred_mode |= 1;
        if (mv2->metric < n*MAGIC_SUPERBLOCK_METRIC) pred_mode |= 2;

        if (pred_mode != 0) {
          mv->pred_mode = pred_mode;
          mv->split = 0;
          mv->dx[0] = mv1->dx[0];
          mv->dy[0] = mv1->dy[0];
          mv->dx[1] = mv2->dx[0];
          mv->dy[1] = mv2->dy[0];
          mv->metric = MIN(mv1->metric, mv2->metric);

          schro_motion_splat_4x4 (me->motion, i, j);
          continue;
        }
      }

      for(l=j;l<j+4;l++) {
        for(k=i;k<i+4;k++) {
          mv = SCHRO_MOTION_GET_BLOCK (me->motion, k, l);

          if (me->encoder_frame->num_refs == 1) {
            mv1 = motion_field_get (mf3, k, l);
            if (mv1->metric < n*MAGIC_BLOCK_METRIC) {
              *mv = *mv1;
              mv->pred_mode = 1;
              mv->split = 2;
              continue;
            }
          } else {
            int pred_mode = 0;

            mv1 = motion_field_get (mf3, k, l);
            mv2 = motion_field_get (mf4, k, l);

            if (mv1->metric < n*MAGIC_BLOCK_METRIC) pred_mode |= 1;
            if (mv2->metric < n*MAGIC_BLOCK_METRIC) pred_mode |= 2;

            if (pred_mode == 3) {
              if (mv1->metric < mv2->metric/2) {
                pred_mode = 1;
              } else if (mv2->metric < mv1->metric/2) {
                pred_mode = 2;
              }
            }

            if (pred_mode != 0) {
              mv->pred_mode = pred_mode;
              mv->split = 2;
              mv->dx[0] = mv1->dx[0];
              mv->dy[0] = mv1->dy[0];
              mv->dx[1] = mv2->dx[0];
              mv->dy[1] = mv2->dy[0];
              mv->metric = MIN(mv1->metric, mv2->metric);
              continue;
            }
          }

          schro_encoder_dc_predict (me->encoder_frame, k, l);
        }
      }
    }
  }
  schro_motion_cleanup (me->motion,
      (params->video_format->width + params->xbsep_luma - 1)/params->xbsep_luma,
      (params->video_format->height + params->ybsep_luma - 1)/params->ybsep_luma);

  schro_motion_field_free (mf1);
  schro_motion_field_free (mf3);
  if (me->encoder_frame->num_refs > 1) {
    schro_motion_field_free (mf2);
    schro_motion_field_free (mf4);
  }

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
  SchroMetricScan scan;
  SchroMotionVector *mv;
  int ref;
  int metric;

  mv = &block->mv[0][0];

  if (mv->pred_mode == 0) {
    int xmax;
    int ymax;
    int ii,jj;
    SchroFrame *frame;
    SchroFrameData *comp;
    SchroMotionVectorDC *mvdc = (SchroMotionVectorDC *)mv;

    frame = get_downsampled (me->encoder_frame, 0);
    comp = frame->components + 0;

    xmax = MIN((i+4)*me->params->xbsep_luma, comp->width);
    ymax = MIN((j+4)*me->params->ybsep_luma, comp->height);

    metric = 0;
    for(jj=j*me->params->ybsep_luma;jj<ymax;jj++){
      for(ii=i*me->params->xbsep_luma;ii<xmax;ii++){
        metric += abs (mvdc->dc[0] - SCHRO_GET(comp->data, jj*comp->stride + ii, uint8_t));
      }
    }
    return metric;
  }

  if (mv->pred_mode == 3) {
    SchroFrameData *comp;
    SchroFrameData *comp_ref0;
    SchroFrameData *comp_ref1;
    int xmax;
    int ymax;
    int ii,jj;
    int val;
    int ref_val;

    SCHRO_ASSERT(mv->dx[0] == 0);
    SCHRO_ASSERT(mv->dx[1] == 0);
    SCHRO_ASSERT(mv->dy[0] == 0);
    SCHRO_ASSERT(mv->dy[1] == 0);

    comp = get_downsampled (me->encoder_frame, 0)->components + 0;
    comp_ref0 = get_downsampled (me->encoder_frame->ref_frame0, 0)->components + 0;
    comp_ref1 = get_downsampled (me->encoder_frame->ref_frame1, 0)->components + 0;

    xmax = MIN((i+4)*me->params->xbsep_luma, comp->width);
    ymax = MIN((j+4)*me->params->ybsep_luma, comp->height);

    metric = 0;
    for(jj=j*me->params->ybsep_luma;jj<ymax;jj++){
      for(ii=i*me->params->xbsep_luma;ii<xmax;ii++){
        val = SCHRO_GET(comp->data, jj*comp->stride + ii, uint8_t);
        ref_val = (SCHRO_GET(comp_ref0->data, jj*comp->stride + ii, uint8_t) +
            SCHRO_GET(comp_ref1->data, jj*comp->stride + ii, uint8_t))/2;
        metric += abs (val - ref_val);
      }
    }
    return metric;
  }

  if (mv->pred_mode == 1) {
    ref = 0;
  } else {
    ref = 1;
  }

  scan.frame = get_downsampled (me->encoder_frame, 0);
  if (ref == 0) {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame0, 0);
  } else {
    scan.ref_frame = get_downsampled (me->encoder_frame->ref_frame1, 0);
  }


  /* FIXME this shouldn't use SchroMetricScan */
  if (mv->split == 0) {
    int dx, dy;

    scan.block_width = 4*me->params->xbsep_luma;
    scan.block_height = 4*me->params->ybsep_luma;
    scan.gravity_scale = 0;
    scan.gravity_x = 0;
    scan.gravity_y = 0;

    scan.x = i * me->params->xbsep_luma;
    scan.y = j * me->params->ybsep_luma;
    schro_metric_scan_setup (&scan, mv->dx[ref], mv->dy[ref], 0);
    if (scan.scan_width <= 0 || scan.scan_height <= 0) {
      mv->dx[ref] = 0;
      mv->dy[ref] = 0;
      mv->metric = SCHRO_METRIC_INVALID;
      return SCHRO_METRIC_INVALID;
    }

    schro_metric_scan_do_scan (&scan);
    metric = schro_metric_scan_get_min (&scan, &dx, &dy);

    return metric;
  }

  SCHRO_ASSERT(0);

  return SCHRO_METRIC_INVALID;
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


