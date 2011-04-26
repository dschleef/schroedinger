
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <schroedinger/schroorc.h>
#include <orc/orc.h>


/* runtime orc code generation */

static SchroMotionFuncs motion_funcs[32];

static void
schro_motion_init_functions (SchroMotion * motion)
{
  if (motion_funcs[motion->xblen >> 1].block_accumulate == NULL) {
    OrcProgram *p;
    OrcCompileResult result;

    p = orc_program_new ();
    orc_program_set_constant_n (p, motion->xblen);
    orc_program_set_2d (p);
    orc_program_set_name (p, "block_acc_Xxn");

    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 1, "s2");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append (p, "convubw", ORC_VAR_T1, ORC_VAR_S2, ORC_VAR_D1);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_S1);
    orc_program_append (p, "addw", ORC_VAR_D1, ORC_VAR_D1, ORC_VAR_T1);

    result = orc_program_compile (p);
    if (!ORC_COMPILE_RESULT_IS_SUCCESSFUL (result)) {
      SCHRO_ERROR ("compile failed");
    }

    motion_funcs[motion->xblen / 2].block_accumulate = p;
  }

  if (motion_funcs[motion->xblen >> 1].block_accumulate_scaled == NULL) {
    OrcProgram *p;
    OrcCompileResult result;

    p = orc_program_new ();
    orc_program_set_constant_n (p, motion->xblen);
    orc_program_set_2d (p);
    orc_program_set_name (p, "block_acc_scaled_Xxn");

    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 1, "s2");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_constant (p, 2, 32, "c1");
    orc_program_add_constant (p, 2, 6, "c2");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append (p, "convubw", ORC_VAR_T1, ORC_VAR_S2, ORC_VAR_D1);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_P1);
    orc_program_append (p, "addw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_C1);
    orc_program_append (p, "shrsw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_C2);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_S1);
    orc_program_append (p, "addw", ORC_VAR_D1, ORC_VAR_D1, ORC_VAR_T1);

    result = orc_program_compile (p);
    if (!ORC_COMPILE_RESULT_IS_SUCCESSFUL (result)) {
      SCHRO_ERROR ("compile failed");
    }

    motion_funcs[motion->xblen / 2].block_accumulate_scaled = p;
  }

  if (motion_funcs[motion->xblen >> 1].block_accumulate_dc == NULL) {
    OrcProgram *p;
    OrcCompileResult result;

    p = orc_program_new ();
    orc_program_set_constant_n (p, motion->xblen);
    orc_program_set_2d (p);
    orc_program_set_name (p, "block_acc_dc_Xxn");

    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_S1, ORC_VAR_P1);
    orc_program_append (p, "addw", ORC_VAR_D1, ORC_VAR_D1, ORC_VAR_T1);

    result = orc_program_compile (p);
    if (!ORC_COMPILE_RESULT_IS_SUCCESSFUL (result)) {
      SCHRO_ERROR ("compile failed");
    }

    motion_funcs[motion->xblen / 2].block_accumulate_dc = p;
  }

  if (motion_funcs[motion->xblen >> 1].block_accumulate_avg == NULL) {
    OrcProgram *p;
    OrcCompileResult result;

    p = orc_program_new ();
    orc_program_set_constant_n (p, motion->xblen);
    orc_program_set_2d (p);
    orc_program_set_name (p, "block_acc_avg_Xxn");

    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 1, "s2");
    orc_program_add_source (p, 1, "s3");
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 1, "t2");

    orc_program_append (p, "avgub", ORC_VAR_T2, ORC_VAR_S2, ORC_VAR_S3);
    orc_program_append (p, "convubw", ORC_VAR_T1, ORC_VAR_T2, 0);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_S1);
    orc_program_append (p, "addw", ORC_VAR_D1, ORC_VAR_D1, ORC_VAR_T1);

    result = orc_program_compile (p);
    if (!ORC_COMPILE_RESULT_IS_SUCCESSFUL (result)) {
      SCHRO_ERROR ("compile failed");
    }

    motion_funcs[motion->xblen / 2].block_accumulate_avg = p;
  }

  if (motion_funcs[motion->xblen >> 1].block_accumulate_biref == NULL) {
    OrcProgram *p;
    OrcCompileResult result;

    p = orc_program_new ();
    orc_program_set_constant_n (p, motion->xblen);
    orc_program_set_2d (p);
    orc_program_set_name (p, "block_acc_biref_Xxn");

    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 1, "s2");
    orc_program_add_source (p, 1, "s3");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_parameter (p, 2, "p2");
    orc_program_add_constant (p, 2, 32, "c1");
    orc_program_add_constant (p, 2, 6, "c2");
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 2, "t2");

    orc_program_append (p, "convubw", ORC_VAR_T1, ORC_VAR_S2, 0);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_P1);
    orc_program_append (p, "convubw", ORC_VAR_T2, ORC_VAR_S3, 0);
    orc_program_append (p, "mullw", ORC_VAR_T2, ORC_VAR_T2, ORC_VAR_P2);
    orc_program_append (p, "addw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_T2);
    orc_program_append (p, "addw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_C1);
    orc_program_append (p, "shrsw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_C2);
    orc_program_append (p, "mullw", ORC_VAR_T1, ORC_VAR_T1, ORC_VAR_S1);
    orc_program_append (p, "addw", ORC_VAR_D1, ORC_VAR_D1, ORC_VAR_T1);

    result = orc_program_compile (p);
    if (!ORC_COMPILE_RESULT_IS_SUCCESSFUL (result)) {
      SCHRO_ERROR ("compile failed");
    }

    motion_funcs[motion->xblen / 2].block_accumulate_biref = p;
  }
}

static void
orc_multiply_and_acc_Xxn_s16_u8 (int16_t * d1, int d1_stride,
    const int16_t * s1, int s1_stride, const uint8_t * s2, int s2_stride,
    int n, int m)
{
  OrcExecutor _ex, *ex = &_ex;
  OrcProgram *p = 0;
  void (*func) (OrcExecutor *);

  p = motion_funcs[n >> 1].block_accumulate;
  ex->program = p;

  ex->n = n;
  ORC_EXECUTOR_M (ex) = m;
  ex->arrays[ORC_VAR_D1] = d1;
  ex->params[ORC_VAR_D1] = d1_stride;
  ex->arrays[ORC_VAR_S1] = (void *) s1;
  ex->params[ORC_VAR_S1] = s1_stride;
  ex->arrays[ORC_VAR_S2] = (void *) s2;
  ex->params[ORC_VAR_S2] = s2_stride;

  func = p->code_exec;
  func (ex);
}

static void
orc_multiply_and_acc_scaled_Xxn_s16_u8 (int16_t * d1, int d1_stride,
    const int16_t * s1, int s1_stride, const uint8_t * s2, int s2_stride,
    int p1, int n, int m)
{
  OrcExecutor _ex, *ex = &_ex;
  OrcProgram *p = 0;
  void (*func) (OrcExecutor *);

  p = motion_funcs[n >> 1].block_accumulate_scaled;
  ex->program = p;

  ex->n = n;
  ORC_EXECUTOR_M (ex) = m;
  ex->arrays[ORC_VAR_D1] = d1;
  ex->params[ORC_VAR_D1] = d1_stride;
  ex->arrays[ORC_VAR_S1] = (void *) s1;
  ex->params[ORC_VAR_S1] = s1_stride;
  ex->arrays[ORC_VAR_S2] = (void *) s2;
  ex->params[ORC_VAR_S2] = s2_stride;
  ex->params[ORC_VAR_P1] = p1;

  func = p->code_exec;
  func (ex);
}

static void
orc_multiply_and_acc_dc_Xxn_s16_u8 (int16_t * d1, int d1_stride,
    const int16_t * s1, int s1_stride, int p1, int n, int m)
{
  OrcExecutor _ex, *ex = &_ex;
  OrcProgram *p = 0;
  void (*func) (OrcExecutor *);

  p = motion_funcs[n >> 1].block_accumulate_dc;
  ex->program = p;

  ex->n = n;
  ORC_EXECUTOR_M (ex) = m;
  ex->arrays[ORC_VAR_D1] = d1;
  ex->params[ORC_VAR_D1] = d1_stride;
  ex->arrays[ORC_VAR_S1] = (void *) s1;
  ex->params[ORC_VAR_S1] = s1_stride;
  ex->params[ORC_VAR_P1] = p1;

  func = p->code_exec;
  func (ex);
}

static void
orc_multiply_and_acc_avg_Xxn_s16_u8 (int16_t * d1, int d1_stride,
    const int16_t * s1, int s1_stride, const uint8_t * s2, int s2_stride,
    const uint8_t * s3, int s3_stride, int n, int m)
{
  OrcExecutor _ex, *ex = &_ex;
  OrcProgram *p;
  void (*func) (OrcExecutor *);

  p = motion_funcs[n >> 1].block_accumulate_avg;

  ex->program = p;

  ex->n = n;
  ORC_EXECUTOR_M (ex) = m;
  ex->arrays[ORC_VAR_D1] = d1;
  ex->params[ORC_VAR_D1] = d1_stride;
  ex->arrays[ORC_VAR_S1] = (void *) s1;
  ex->params[ORC_VAR_S1] = s1_stride;
  ex->arrays[ORC_VAR_S2] = (void *) s2;
  ex->params[ORC_VAR_S2] = s2_stride;
  ex->arrays[ORC_VAR_S3] = (void *) s3;
  ex->params[ORC_VAR_S3] = s3_stride;

  func = p->code_exec;
  func (ex);
}

static void
orc_multiply_and_acc_biref_Xxn_s16_u8 (int16_t * d1, int d1_stride,
    const int16_t * s1, int s1_stride, const uint8_t * s2, int s2_stride,
    const uint8_t * s3, int s3_stride, int p1, int p2, int n, int m)
{
  OrcExecutor _ex, *ex = &_ex;
  OrcProgram *p;
  void (*func) (OrcExecutor *);

  p = motion_funcs[n >> 1].block_accumulate_biref;

  ex->program = p;

  ex->n = n;
  ORC_EXECUTOR_M (ex) = m;
  ex->arrays[ORC_VAR_D1] = d1;
  ex->params[ORC_VAR_D1] = d1_stride;
  ex->arrays[ORC_VAR_S1] = (void *) s1;
  ex->params[ORC_VAR_S1] = s1_stride;
  ex->arrays[ORC_VAR_S2] = (void *) s2;
  ex->params[ORC_VAR_S2] = s2_stride;
  ex->arrays[ORC_VAR_S3] = (void *) s3;
  ex->params[ORC_VAR_S3] = s3_stride;
  ex->params[ORC_VAR_P1] = p1;
  ex->params[ORC_VAR_P2] = p2;

  func = p->code_exec;
  func (ex);
}

/* motion render (faster) */

static void
get_block (SchroMotion * motion, int k, int ref, int i, int j, int dx, int dy)
{
  int px, py;
  int x, y;
  SchroUpsampledFrame *upframe;
  int exp;

  if (k > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->
        video_format->chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT (motion->params->
        video_format->chroma_format);
  }
  if (ref) {
    upframe = motion->src2;
  } else {
    upframe = motion->src1;
  }

  x = motion->xbsep * i - motion->xoffset;
  y = motion->ybsep * j - motion->yoffset;
  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;
  exp = 32 << motion->mv_precision;

  px = CLAMP (px, -exp, motion->max_fast_x + exp - 1);
  py = CLAMP (py, -exp, motion->max_fast_y + exp - 1);

  schro_upsampled_frame_get_block_fast_precN (upframe, k, px, py,
      motion->mv_precision, &motion->block_ref[ref],
      &motion->alloc_block_ref[ref]);
}

static void
get_dc_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int value;
  int ii, jj;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
  value = mv->u.dc.dc[k];
  for (jj = 0; jj < motion->yblen; jj++) {
    uint8_t *data = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
    /* FIXME splat */
    for (ii = 0; ii < motion->xblen; ii++) {
      data[ii] = value + 128;
    }
  }
}

static void
get_ref1_block_simple (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);
}

static void
get_ref1_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int ii, jj;
  int weight;
  int shift;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);

  weight = motion->ref1_weight + motion->ref2_weight;
  shift = motion->ref_weight_precision;

  if (motion->oneref_noscale) {
    memcpy (&motion->block, &motion->block_ref[0], sizeof (SchroFrameData));
  } else {
    memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
    for (jj = 0; jj < motion->yblen; jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[0], jj);
      for (ii = 0; ii < motion->xblen; ii++) {
        d[ii] = ROUND_SHIFT (s[ii] * weight, shift);
      }
    }
  }
}

static void
get_ref2_block_simple (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);
}

static void
get_ref2_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int ii, jj;
  int weight;
  int shift;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);

  weight = motion->ref1_weight + motion->ref2_weight;
  shift = motion->ref_weight_precision;

  if (motion->oneref_noscale) {
    memcpy (&motion->block, &motion->block_ref[1], sizeof (SchroFrameData));
  } else {
    memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
    for (jj = 0; jj < motion->yblen; jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block_ref[1], jj);
      for (ii = 0; ii < motion->xblen; ii++) {
        d[ii] = ROUND_SHIFT (s[ii] * weight, shift);
      }
    }
  }
}

static void
get_biref_block_simple (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);
  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);
}

static void
get_biref_block (SchroMotion * motion, int i, int j, int k, int x, int y)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;
  int weight0, weight1;
  int shift;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];
  SCHRO_ASSERT (mv->using_global == FALSE);

  get_block (motion, k, 0, i, j, mv->u.vec.dx[0], mv->u.vec.dy[0]);
  get_block (motion, k, 1, i, j, mv->u.vec.dx[1], mv->u.vec.dy[1]);

  weight0 = motion->ref1_weight;
  weight1 = motion->ref2_weight;
  shift = motion->ref_weight_precision;

  memcpy (&motion->block, &motion->alloc_block, sizeof (SchroFrameData));
  if (motion->simple_weight) {
    switch (motion->xblen) {
      case 8:
        orc_avg2_8xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 12:
        orc_avg2_12xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 16:
        orc_avg2_16xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      case 32:
        orc_avg2_32xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->yblen);
        break;
      default:
        orc_avg2_nxm_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            motion->xblen, motion->yblen);
        break;
    }
  } else {
    switch (motion->xblen) {
#if 0
      case 8:
        orc_combine2_8xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            weight0, weight1, (1 << shift) >> 1, shift, motion->yblen);
        break;
      case 12:
        orc_combine2_12xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            weight0, weight1, (1 << shift) >> 1, shift, motion->yblen);
        break;
      case 16:
        orc_combine2_16xn_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            weight0, weight1, (1 << shift) >> 1, shift, motion->yblen);
        break;
#endif
      default:
        orc_combine2_nxm_u8 (motion->block.data, motion->block.stride,
            motion->block_ref[0].data, motion->block_ref[0].stride,
            motion->block_ref[1].data, motion->block_ref[1].stride,
            weight0, weight1, (1 << shift) >> 1, shift,
            motion->xblen, motion->yblen);
        break;
    }
  }
}

static void
schro_motion_block_predict_block (SchroMotion * motion, int x, int y, int k,
    int i, int j)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  switch (mv->pred_mode) {
    case 0:
      get_dc_block (motion, i, j, k, x, y);
      break;
    case 1:
      get_ref1_block (motion, i, j, k, x, y);
      break;
    case 2:
      get_ref2_block (motion, i, j, k, x, y);
      break;
    case 3:
      get_biref_block (motion, i, j, k, x, y);
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}

static void
schro_motion_block_predict_and_acc (SchroMotion * motion, int x, int y, int k,
    int i, int j, SchroFrameData * comp)
{
  SchroParams *params = motion->params;
  SchroMotionVector *mv;

  mv = &motion->motion_vectors[j * params->x_num_blocks + i];

  if (motion->simple_weight) {
    switch (mv->pred_mode) {
      case 0:
        orc_multiply_and_acc_dc_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, mv->u.dc.dc[k] + 128, motion->xblen,
            motion->yblen);
        break;
      case 1:
        get_ref1_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp,
                x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[0].data,
            motion->block_ref[0].stride, motion->xblen, motion->yblen);
        break;
      case 2:
        get_ref2_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp,
                x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[1].data,
            motion->block_ref[1].stride, motion->xblen, motion->yblen);
        break;
      case 3:
        get_biref_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_avg_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[0].data,
            motion->block_ref[0].stride, motion->block_ref[1].data,
            motion->block_ref[1].stride, motion->xblen, motion->yblen);
        break;
      default:
        SCHRO_ASSERT (0);
        break;
    }
  } else {
    int weight0, weight1, shift;

    weight0 = motion->ref1_weight;
    weight1 = motion->ref2_weight;
    shift = motion->ref_weight_precision;

    switch (mv->pred_mode) {
      case 0:
        orc_multiply_and_acc_dc_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, mv->u.dc.dc[k] + 128, motion->xblen,
            motion->yblen);
        break;
      case 1:
        get_ref1_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_scaled_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[0].data,
            motion->block_ref[0].stride, (weight0 + weight1) << (6 - shift),
            motion->xblen, motion->yblen);
        break;
      case 2:
        get_ref2_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_scaled_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[1].data,
            motion->block_ref[1].stride, (weight0 + weight1) << (6 - shift),
            motion->xblen, motion->yblen);
        break;
      case 3:
        get_biref_block_simple (motion, i, j, k, x, y);
        orc_multiply_and_acc_biref_Xxn_s16_u8 (SCHRO_FRAME_DATA_GET_PIXEL_S16
            (comp, x, y), comp->stride, motion->obmc_weight.data,
            motion->obmc_weight.stride, motion->block_ref[0].data,
            motion->block_ref[0].stride, motion->block_ref[1].data,
            motion->block_ref[1].stride, weight0 << (6 - shift),
            weight1 << (6 - shift), motion->xblen, motion->yblen);
        break;
      default:
        SCHRO_ASSERT (0);
        break;
    }
  }
}

static void
schro_motion_block_accumulate_slow (SchroMotion * motion, SchroFrameData * comp,
    int x, int y)
{
  int i, j;
  int w_x, w_y;

  for (j = 0; j < motion->yblen; j++) {
    int16_t *d = SCHRO_FRAME_DATA_GET_PIXEL_S16 (comp, x, y + j);
    uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->block, j);

    if (y + j < 0 || y + j >= comp->height)
      continue;

    w_y = motion->weight_y[j];
    if (y + j < motion->yoffset) {
      w_y += motion->weight_y[2 * motion->yoffset - j - 1];
    }
    if (y + j >= motion->params->y_num_blocks * motion->ybsep - motion->yoffset) {
      w_y += motion->weight_y[2 * (motion->yblen - motion->yoffset) - j - 1];
    }


    for (i = 0; i < motion->xblen; i++) {
      if (x + i < 0 || x + i >= comp->width)
        continue;

      w_x = motion->weight_x[i];
      if (x + i < motion->xoffset) {
        w_x += motion->weight_x[2 * motion->xoffset - i - 1];
      }
      if (x + i >=
          motion->params->x_num_blocks * motion->xbsep - motion->xoffset) {
        w_x += motion->weight_x[2 * (motion->xblen - motion->xoffset) - i - 1];
      }

      d[i] += s[i] * w_x * w_y;
    }
  }
}

void
schro_motion_render_u8 (SchroMotion * motion, SchroFrame * dest,
    SchroFrame * addframe, int add, SchroFrame * output_frame)
{
  int i, j;
  int x, y;
  int k;
  SchroParams *params = motion->params;
  int max_x_blocks;
  int max_y_blocks;

  if (params->num_refs == 1) {
    SCHRO_ASSERT (params->picture_weight_2 == 1);
  }

  motion->ref_weight_precision = params->picture_weight_bits;
  motion->ref1_weight = params->picture_weight_1;
  motion->ref2_weight = params->picture_weight_2;

  motion->mv_precision = params->mv_precision;

  for (k = 0; k < 3; k++) {
    SchroFrameData *comp = dest->components + k;
    SchroFrameData *acomp = addframe->components + k;
    SchroFrameData *ocomp = NULL;

    if (output_frame) {
      ocomp = output_frame->components + k;
    }

    if (k == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
      motion->width = comp->width;
      motion->height = comp->height;
    } else {
      motion->xbsep = params->xbsep_luma >>
          SCHRO_CHROMA_FORMAT_H_SHIFT (motion->params->
          video_format->chroma_format);
      motion->ybsep =
          params->ybsep_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (motion->
          params->video_format->chroma_format);
      motion->xblen =
          params->xblen_luma >> SCHRO_CHROMA_FORMAT_H_SHIFT (motion->
          params->video_format->chroma_format);
      motion->yblen =
          params->yblen_luma >> SCHRO_CHROMA_FORMAT_V_SHIFT (motion->
          params->video_format->chroma_format);
      motion->width = comp->width;
      motion->height = comp->height;
    }
    motion->xoffset = (motion->xblen - motion->xbsep) / 2;
    motion->yoffset = (motion->yblen - motion->ybsep) / 2;
    motion->max_fast_x =
        (motion->width - motion->xblen) << motion->mv_precision;
    motion->max_fast_y =
        (motion->height - motion->yblen) << motion->mv_precision;

    motion->alloc_block.data =
        schro_malloc (motion->xblen * motion->yblen * sizeof (uint8_t));
    motion->alloc_block.stride = motion->xblen * sizeof (uint8_t);
    motion->alloc_block.width = motion->xblen;
    motion->alloc_block.height = motion->yblen;
    motion->obmc_weight.data =
        schro_malloc (motion->xblen * motion->yblen * sizeof (int16_t));
    motion->obmc_weight.stride = motion->xblen * sizeof (int16_t);
    motion->obmc_weight.width = motion->xblen;
    motion->obmc_weight.height = motion->yblen;
    motion->alloc_block_ref[0].data =
        schro_malloc (motion->xblen * motion->yblen * sizeof (uint8_t));
    motion->alloc_block_ref[0].stride = motion->xblen * sizeof (uint8_t);
    motion->alloc_block_ref[0].width = motion->xblen;
    motion->alloc_block_ref[0].height = motion->yblen;
    motion->alloc_block_ref[1].data =
        schro_malloc (motion->xblen * motion->yblen * sizeof (uint8_t));
    motion->alloc_block_ref[1].stride = motion->xblen * sizeof (uint8_t);
    motion->alloc_block_ref[1].width = motion->xblen;
    motion->alloc_block_ref[1].height = motion->yblen;

    if (motion->ref1_weight == 1 && motion->ref2_weight == 1 &&
        motion->ref_weight_precision == 1) {
      motion->simple_weight = TRUE;
    }
    if (motion->ref1_weight + motion->ref2_weight ==
        (1 << motion->ref_weight_precision)) {
      motion->oneref_noscale = TRUE;
    }

    schro_motion_init_obmc_weight (motion);
    schro_motion_init_functions (motion);
    //schro_motion_set_block_accumulate (motion);

    max_x_blocks = MIN (params->x_num_blocks - 1,
        (motion->width - motion->xoffset) / motion->xbsep);
    max_y_blocks = MIN (params->y_num_blocks - 1,
        (motion->height - motion->yoffset) / motion->ybsep);

    j = 0;
    orc_splat_s16_2d (SCHRO_FRAME_DATA_GET_LINE (comp, 0), comp->stride,
        0, comp->width, motion->ybsep + motion->yoffset);
    for (i = 0; i < params->x_num_blocks; i++) {
      x = motion->xbsep * i - motion->xoffset;
      y = motion->ybsep * j - motion->yoffset;

      schro_motion_block_predict_block (motion, x, y, k, i, j);
      schro_motion_block_accumulate_slow (motion, comp, x, y);
    }
    if (add) {
      if (SCHRO_FRAME_FORMAT_DEPTH (addframe->format) ==
          SCHRO_FRAME_FORMAT_DEPTH_S16) {
        orc_rrshift6_add_s16_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, 0),
            ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, 0), acomp->stride,
            SCHRO_FRAME_DATA_GET_LINE (comp, 0), comp->stride, motion->width,
            motion->ybsep - motion->yoffset);
      } else {
        orc_rrshift6_add_s32_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, 0),
            ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, 0), acomp->stride,
            SCHRO_FRAME_DATA_GET_LINE (comp, 0), comp->stride, motion->width,
            motion->ybsep - motion->yoffset);
      }
    } else {
      orc_rrshift6_sub_s16_2d (SCHRO_FRAME_DATA_GET_LINE (acomp, 0),
          acomp->stride, SCHRO_FRAME_DATA_GET_LINE (comp, 0), comp->stride,
          motion->width, motion->ybsep - motion->yoffset);
    }
    for (j = 1; j < max_y_blocks; j++) {
      y = motion->ybsep * j - motion->yoffset;
      orc_splat_s16_2d (SCHRO_FRAME_DATA_GET_LINE (comp,
              y + motion->yoffset * 2), comp->stride, 0, comp->width,
          motion->ybsep);

      i = 0;
      {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }

      for (i = 1; i < max_x_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_and_acc (motion, x, y, k, i, j, comp);
      }

      for (; i < params->x_num_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
      if (add) {
        if (SCHRO_FRAME_FORMAT_DEPTH (addframe->format) ==
            SCHRO_FRAME_FORMAT_DEPTH_S16) {
          orc_rrshift6_add_s16_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
              ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
              SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, motion->width,
              motion->ybsep);
        } else {
          orc_rrshift6_add_s32_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
              ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
              SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, motion->width,
              motion->ybsep);
        }
      } else {
        orc_rrshift6_sub_s16_2d (SCHRO_FRAME_DATA_GET_LINE (acomp, y),
            acomp->stride, SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride,
            motion->width, motion->ybsep);
      }
    }
    for (j = max_y_blocks; j < params->y_num_blocks; j++) {
      y = motion->ybsep * j - motion->yoffset;
      orc_splat_s16_2d (SCHRO_FRAME_DATA_GET_LINE (comp,
              y + motion->yoffset * 2), comp->stride, 0, comp->width,
          CLAMP (comp->height - (y + motion->yoffset * 2), 0, motion->ybsep));
      for (i = 0; i < params->x_num_blocks; i++) {
        x = motion->xbsep * i - motion->xoffset;

        schro_motion_block_predict_block (motion, x, y, k, i, j);
        schro_motion_block_accumulate_slow (motion, comp, x, y);
      }
      if (add) {
        if (SCHRO_FRAME_FORMAT_DEPTH (addframe->format) ==
            SCHRO_FRAME_FORMAT_DEPTH_S16) {
          orc_rrshift6_add_s16_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
              ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
              SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, comp->width,
              CLAMP (comp->height - y, 0, motion->ybsep));
        } else {
          orc_rrshift6_add_s32_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
              ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
              SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, comp->width,
              CLAMP (comp->height - y, 0, motion->ybsep));
        }
      } else {
        orc_rrshift6_sub_s16_2d (SCHRO_FRAME_DATA_GET_LINE (acomp, y),
            acomp->stride, SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride,
            comp->width, CLAMP (comp->height - y, 0, motion->ybsep));
      }
    }

    y = params->y_num_blocks * motion->ybsep - motion->yoffset;
    if (add) {
      if (SCHRO_FRAME_FORMAT_DEPTH (addframe->format) ==
          SCHRO_FRAME_FORMAT_DEPTH_S16) {
        orc_rrshift6_add_s16_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
            ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
            SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, comp->width,
            CLAMP (comp->height - y, 0, motion->ybsep));
      } else {
        orc_rrshift6_add_s32_2d (SCHRO_FRAME_DATA_GET_LINE (ocomp, y),
            ocomp->stride, SCHRO_FRAME_DATA_GET_LINE (acomp, y), acomp->stride,
            SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride, comp->width,
            CLAMP (comp->height - y, 0, motion->ybsep));
      }
    } else {
      orc_rrshift6_sub_s16_2d (SCHRO_FRAME_DATA_GET_LINE (acomp, y),
          acomp->stride, SCHRO_FRAME_DATA_GET_LINE (comp, y), comp->stride,
          comp->width, CLAMP (comp->height - y, 0, motion->ybsep));
    }

    schro_free (motion->alloc_block.data);
    schro_free (motion->obmc_weight.data);
    schro_free (motion->alloc_block_ref[0].data);
    schro_free (motion->alloc_block_ref[1].data);
  }

}

