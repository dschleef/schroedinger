
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <string.h>
#include <schroedinger/schroorc.h>
#include <orc/orc.h>

extern int _schro_motion_ref;



SchroMotion *
schro_motion_new (SchroParams * params, SchroUpsampledFrame * ref1,
    SchroUpsampledFrame * ref2)
{
  SchroMotion *motion;

  motion = schro_malloc0 (sizeof (SchroMotion));

  motion->params = params;
  motion->src1 = ref1;
  motion->src2 = ref2;

  motion->motion_vectors =
      schro_malloc0 (sizeof (SchroMotionVector) * params->x_num_blocks *
      params->y_num_blocks);

  motion->tmpdata = schro_malloc (64 * 64 * 3);

  return motion;
}

void
schro_motion_free (SchroMotion * motion)
{
  schro_free (motion->tmpdata);
  schro_free (motion->motion_vectors);
  schro_free (motion);
}

static int
get_ramp (int x, int offset)
{
  if (offset == 1) {
    if (x == 0)
      return 3;
    return 5;
  }
  return 1 + (6 * x + offset - 1) / (2 * offset - 1);
}


void
schro_motion_init_obmc_weight (SchroMotion * motion)
{
  int i;
  int j;
  int wx, wy;

  for (i = 0; i < motion->xblen; i++) {
    if (motion->xoffset == 0) {
      wx = 8;
    } else if (i < 2 * motion->xoffset) {
      wx = get_ramp (i, motion->xoffset);
    } else if (motion->xblen - 1 - i < 2 * motion->xoffset) {
      wx = get_ramp (motion->xblen - 1 - i, motion->xoffset);
    } else {
      wx = 8;
    }
    motion->weight_x[i] = wx;
  }

  for (j = 0; j < motion->yblen; j++) {
    if (motion->yoffset == 0) {
      wy = 8;
    } else if (j < 2 * motion->yoffset) {
      wy = get_ramp (j, motion->yoffset);
    } else if (motion->yblen - 1 - j < 2 * motion->yoffset) {
      wy = get_ramp (motion->yblen - 1 - j, motion->yoffset);
    } else {
      wy = 8;
    }
    motion->weight_y[j] = wy;
  }

  for (j = 0; j < motion->yblen; j++) {
    int16_t *w = SCHRO_FRAME_DATA_GET_LINE (&motion->obmc_weight, j);

    for (i = 0; i < motion->xblen; i++) {
      w[i] = motion->weight_x[i] * motion->weight_y[j];
    }
  }

}

void
schro_motion_render (SchroMotion * motion, SchroFrame * dest,
    SchroFrame * addframe, int add, SchroFrame * output_frame)
{
  SchroParams *params = motion->params;

#ifdef ENABLE_MOTION_REF
  if (_schro_motion_ref) {
    schro_motion_render_ref (motion, dest, addframe, add, output_frame);
    return;
  }
#endif

  if (0 && schro_motion_render_fast_allowed (motion)) {
    schro_motion_render_fast (motion, dest, addframe, add, output_frame);
    return;
  }

  if (params->have_global_motion) {
#ifdef ENABLE_MOTION_REF
    SCHRO_WARNING ("global motion enabled, using reference motion renderer");
    schro_motion_render_ref (motion, dest, addframe, add, output_frame);
    return;
#else
    SCHRO_ERROR ("global motion enabled, probably will crash");
#endif
  }

  {
    int min_extension;
    int i;

    min_extension = motion->src1->frames[0]->extension;
    for (i = 0; i < 4; i++) {
      if (motion->src1->frames[i]) {
        min_extension = MIN (min_extension, motion->src1->frames[i]->extension);
      }
      if (motion->src2 && motion->src2->frames[i]) {
        min_extension = MIN (min_extension, motion->src2->frames[i]->extension);
      }
    }

    if (MAX (params->xblen_luma, params->yblen_luma) > min_extension) {
#ifdef ENABLE_MOTION_REF
      SCHRO_WARNING
          ("block size (%dx%d) larger than minimum frame extension %d, using reference motion renderer",
          params->xblen_luma, params->yblen_luma, min_extension);
      schro_motion_render_ref (motion, dest, addframe, add, output_frame);
      return;
#else
      SCHRO_ERROR
          ("block size (%dx%d) larger than minimum frame extension %d, probably will crash",
          params->xblen_luma, params->yblen_luma, min_extension);
#endif
    }
  }

  schro_motion_render_u8 (motion, dest, addframe, add, output_frame);
}



/* original */


void
schro_motion_dc_prediction (SchroMotion * motion, int x, int y, int *pred)
{
  SchroMotionVector *mv;
  int i;

  for (i = 0; i < 3; i++) {
    int sum = 0;
    int n = 0;

    if (x > 0) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y);
      if (mv->pred_mode == 0) {
        sum += mv->u.dc.dc[i];
        n++;
      }
    }
    if (y > 0) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 1);
      if (mv->pred_mode == 0) {
        sum += mv->u.dc.dc[i];
        n++;
      }
    }
    if (x > 0 && y > 0) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y - 1);
      if (mv->pred_mode == 0) {
        sum += mv->u.dc.dc[i];
        n++;
      }
    }
    switch (n) {
      case 0:
        pred[i] = 0;
        break;
      case 1:
        pred[i] = (short) sum;
        break;
      case 2:
        pred[i] = (sum + 1) >> 1;
        break;
      case 3:
        pred[i] = schro_divide3 (sum + 1);
        break;
      default:
        SCHRO_ASSERT (0);
    }
  }
}

int
schro_motion_get_global_prediction (SchroMotion * motion, int x, int y)
{
  SchroMotionVector *mv;
  int sum;

  if (x == 0 && y == 0) {
    return 0;
  }
  if (y == 0) {
    mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, 0);
    return mv->using_global;
  }
  if (x == 0) {
    mv = SCHRO_MOTION_GET_BLOCK (motion, 0, y - 1);
    return mv->using_global;
  }

  sum = 0;
  mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y);
  sum += mv->using_global;
  mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 1);
  sum += mv->using_global;
  mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y - 1);
  sum += mv->using_global;

  return (sum >= 2);
}

static int
median3 (int a, int b, int c)
{
  if (a < b) {
    if (b < c)
      return b;
    if (c < a)
      return a;
    return c;
  } else {
    if (a < c)
      return a;
    if (c < b)
      return b;
    return c;
  }
}

void
schro_mf_vector_prediction (SchroMotionField * mf,
    int x, int y, int *pred_x, int *pred_y, int mode)
{
  int x_num_blocks;
  SchroMotionVector *mv;
  int vx[3], vy[3];
  int n = 0;
  int ref = mode - 1;

  SCHRO_ASSERT (mf && pred_x && pred_y);
  SCHRO_ASSERT (1 == mode || 2 == mode);

  x_num_blocks = mf->x_num_blocks;

  if (0 < x) {
    mv = &mf->motion_vectors[y * x_num_blocks + x - 1];
    vx[n] = mv->u.vec.dx[ref];
    vy[n] = mv->u.vec.dy[ref];
    ++n;
  }
  if (0 < y) {
    mv = &mf->motion_vectors[(y - 1) * x_num_blocks + x];
    vx[n] = mv->u.vec.dx[ref];
    vy[n] = mv->u.vec.dy[ref];
    ++n;
  }
  if (0 < x && 0 < y) {
    mv = &mf->motion_vectors[(y - 1) * x_num_blocks + x - 1];
    vx[n] = mv->u.vec.dx[ref];
    vy[n] = mv->u.vec.dy[ref];
    ++n;
  }
  switch (n) {
    case 0:
      *pred_x = 0;
      *pred_y = 0;
      break;
    case 1:
      *pred_x = vx[0];
      *pred_y = vy[0];
      break;
    case 2:
      *pred_x = (vx[0] + vx[1] + 1) >> 1;
      *pred_y = (vy[0] + vy[1] + 1) >> 1;
      break;
    case 3:
      *pred_x = median3 (vx[0], vx[1], vx[2]);
      *pred_y = median3 (vy[0], vy[1], vy[2]);
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

void
schro_motion_vector_prediction (SchroMotion * motion,
    int x, int y, int *pred_x, int *pred_y, int mode)
{
  SchroMotionVector *mv;
  int vx[3];
  int vy[3];
  int n = 0;

  SCHRO_ASSERT (mode == 1 || mode == 2);
  if (x > 0) {
    mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->u.vec.dx[mode - 1];
      vy[n] = mv->u.vec.dy[mode - 1];
      n++;
    }
  }
  if (y > 0) {
    mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->u.vec.dx[mode - 1];
      vy[n] = mv->u.vec.dy[mode - 1];
      n++;
    }
  }
  if (x > 0 && y > 0) {
    mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y - 1);
    if (mv->using_global == FALSE && (mv->pred_mode & mode)) {
      vx[n] = mv->u.vec.dx[mode - 1];
      vy[n] = mv->u.vec.dy[mode - 1];
      n++;
    }
  }
  switch (n) {
    case 0:
      *pred_x = 0;
      *pred_y = 0;
      break;
    case 1:
      *pred_x = vx[0];
      *pred_y = vy[0];
      break;
    case 2:
      *pred_x = (vx[0] + vx[1] + 1) >> 1;
      *pred_y = (vy[0] + vy[1] + 1) >> 1;
      break;
    case 3:
      *pred_x = median3 (vx[0], vx[1], vx[2]);
      *pred_y = median3 (vy[0], vy[1], vy[2]);
      break;
    default:
      SCHRO_ASSERT (0);
  }
}

int
schro_motion_split_prediction (SchroMotion * motion, int x, int y)
{
  SchroMotionVector *mv;

  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 4, 0);
      return mv->split;
    }
  } else {
    if (x == 0) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 4);
      return mv->split;
    } else {
      int sum;

      mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 4);
      sum = mv->split;
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 4, y);
      sum += mv->split;
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 4, y - 4);
      sum += mv->split;

      return (sum + 1) / 3;
    }
  }
}

int
schro_motion_get_mode_prediction (SchroMotion * motion, int x, int y)
{
  SchroMotionVector *mv;

  if (y == 0) {
    if (x == 0) {
      return 0;
    } else {
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, 0);
      return mv->pred_mode;
    }
  } else {
    if (x == 0) {
      mv = SCHRO_MOTION_GET_BLOCK (motion, 0, y - 1);
      return mv->pred_mode;
    } else {
      int a, b, c;

      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y);
      a = mv->pred_mode;
      mv = SCHRO_MOTION_GET_BLOCK (motion, x, y - 1);
      b = mv->pred_mode;
      mv = SCHRO_MOTION_GET_BLOCK (motion, x - 1, y - 1);
      c = mv->pred_mode;

      return (a & b) | (b & c) | (c & a);
    }
  }
}

int
schro_motion_vector_is_equal (SchroMotionVector * a, SchroMotionVector * b)
{
  if (a == b)
    return 1;
  return (memcmp (a, b, sizeof (SchroMotionVector)) == 0);
}

int
schro_motion_verify (SchroMotion * motion)
{
  int x, y;
  SchroMotionVector *mv, *sbmv, *bmv;
  SchroParams *params = motion->params;

  if (motion->src1 == NULL) {
    SCHRO_ERROR ("motion->src1 is NULL");
    return 0;
  }

  for (y = 0; y < params->y_num_blocks; y++) {
    for (x = 0; x < params->x_num_blocks; x++) {
      mv = &motion->motion_vectors[y * params->x_num_blocks + x];
      sbmv =
          &motion->motion_vectors[(y & ~3) * params->x_num_blocks + (x & ~3)];

      if (mv->split != sbmv->split) {
        SCHRO_ERROR ("mv(%d,%d) has the wrong split", x, y);
        return 0;
      }

      switch (sbmv->split) {
        case 0:
          if (!schro_motion_vector_is_equal (mv, sbmv)) {
            SCHRO_ERROR ("mv(%d,%d) not equal to superblock mv", x, y);
            return 0;
          }
          break;
        case 1:
          bmv =
              &motion->motion_vectors[(y & ~1) * params->x_num_blocks +
              (x & ~1)];
          if (!schro_motion_vector_is_equal (mv, bmv)) {
            SCHRO_ERROR ("mv(%d,%d) not equal to 2-block mv", x, y);
            return 0;
          }
          break;
        case 2:
          break;
        default:
          SCHRO_ERROR ("mv(%d,%d) had bad split %d", sbmv->split);
          break;
      }

      switch (mv->pred_mode) {
        case 0:
        {
          int i;

          for (i = 0; i < 3; i++) {
            /* FIXME 8bit */
            if (mv->u.dc.dc[i] < -128 || mv->u.dc.dc[i] > 127) {
              SCHRO_ERROR ("mv(%d,%d) has bad DC value [%d] %d", x, y,
                  i, mv->u.dc.dc[i]);
              return 0;
            }
          }
        }
          break;
        case 1:
          break;
        case 2:
        case 3:
          if (motion->params->num_refs < 2) {
            SCHRO_ERROR ("mv(%d,%d) uses non-existent src2", x, y);
            return 0;
          }
          break;
        default:
          SCHRO_ASSERT (0);
          break;
      }

      if (params->have_global_motion == FALSE) {
        if (mv->using_global) {
          SCHRO_ERROR ("mv(%d,%d) uses global motion (disabled)", x, y);
          return 0;
        }
      }
    }
  }

  return 1;
}
