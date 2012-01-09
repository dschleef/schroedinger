
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroorc.h>
#include <schroedinger/schrovirtframe.h>
#include <orc/orc.h>

static void schro_iwt_desl_9_3 (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_5_3 (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_13_5 (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_haar0 (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_haar1 (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_fidelity (SchroFrameData *fd, int16_t *tmp);
static void schro_iwt_daub_9_7 (SchroFrameData *fd, int16_t *tmp);

static void schro_iwt_desl_9_3_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_5_3_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_13_5_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_haar0_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_haar1_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_fidelity_s32 (SchroFrameData *fd, int32_t *tmp);
static void schro_iwt_daub_9_7_s32 (SchroFrameData *fd, int32_t *tmp);

static void schro_iiwt_desl_9_3 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_5_3 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_13_5 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_haar0 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_haar1 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_fidelity (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);
static void schro_iiwt_daub_9_7 (SchroFrameData *dest, SchroFrameData *src,
        int16_t *tmp);

static void schro_iiwt_desl_9_3_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_5_3_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_13_5_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_haar0_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_haar1_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_fidelity_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);
static void schro_iiwt_daub_9_7_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t *tmp);


/* Forward transform splitter function */

void
schro_wavelet_transform_2d (SchroFrameData * fd, int filter, int16_t * tmp)
{
  if ((SCHRO_FRAME_FORMAT_DEPTH (fd->format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S16)) {
    switch (filter) {
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
        schro_iwt_desl_9_3 (fd, tmp);
        break;
      case SCHRO_WAVELET_LE_GALL_5_3:
        schro_iwt_5_3 (fd, tmp);
        break;
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
        schro_iwt_13_5 (fd, tmp);
        break;
      case SCHRO_WAVELET_HAAR_0:
        schro_iwt_haar0 (fd, tmp);
        break;
      case SCHRO_WAVELET_HAAR_1:
        schro_iwt_haar1 (fd, tmp);
        break;
      case SCHRO_WAVELET_FIDELITY:
        schro_iwt_fidelity (fd, tmp);
        break;
      case SCHRO_WAVELET_DAUBECHIES_9_7:
        schro_iwt_daub_9_7 (fd, tmp);
        break;
      default:
        SCHRO_ASSERT (0);
    }
  } else {
    switch (filter) {
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
        schro_iwt_desl_9_3_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_LE_GALL_5_3:
        schro_iwt_5_3_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
        schro_iwt_13_5_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_HAAR_0:
        schro_iwt_haar0_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_HAAR_1:
        schro_iwt_haar1_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_FIDELITY:
        schro_iwt_fidelity_s32 (fd, (orc_int32 *)tmp);
        break;
      case SCHRO_WAVELET_DAUBECHIES_9_7:
        schro_iwt_daub_9_7_s32 (fd, (orc_int32 *)tmp);
        break;
      default:
        SCHRO_ASSERT (0);
    }
  }
}

/* Inverse transform splitter function */

void
schro_wavelet_inverse_transform_2d (SchroFrameData * fd_dest,
    SchroFrameData *fd_src, int filter, int16_t * tmp)
{

  if ((SCHRO_FRAME_FORMAT_DEPTH (fd_dest->format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S16)) {
    SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (fd_src->format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S16);

    switch (filter) {
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
        schro_iiwt_desl_9_3 (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_LE_GALL_5_3:
        schro_iiwt_5_3 (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
        schro_iiwt_13_5 (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_HAAR_0:
        schro_iiwt_haar0 (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_HAAR_1:
        schro_iiwt_haar1 (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_FIDELITY:
        schro_iiwt_fidelity (fd_dest, fd_src, tmp);
        break;
      case SCHRO_WAVELET_DAUBECHIES_9_7:
        schro_iiwt_daub_9_7 (fd_dest, fd_src, tmp);
        break;
      default:
        SCHRO_ASSERT (0);
    }
  } else if ((SCHRO_FRAME_FORMAT_DEPTH (fd_dest->format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S32)) {
    SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (fd_src->format) ==
        SCHRO_FRAME_FORMAT_DEPTH_S32);

    switch (filter) {
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
        schro_iiwt_desl_9_3_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_LE_GALL_5_3:
        schro_iiwt_5_3_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
        schro_iiwt_13_5_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_HAAR_0:
        schro_iiwt_haar0_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_HAAR_1:
        schro_iiwt_haar1_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_FIDELITY:
        schro_iiwt_fidelity_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      case SCHRO_WAVELET_DAUBECHIES_9_7:
        schro_iiwt_daub_9_7_s32 (fd_dest, fd_src, (int32_t *)tmp);
        break;
      default:
        SCHRO_ASSERT (0);
    }
  }

}

/* some utility functions */

static void
extend_1_2 (orc_int16 *data, int n)
{
  data[-1] = data[0];

  data[n] = data[n - 1];
  data[n + 1] = data[n - 1];
}

static void
extend_2_1 (orc_int16 *data, int n)
{
  data[-2] = data[0];
  data[-1] = data[0];

  data[n] = data[n - 1];
}

static void
extend_1_1 (orc_int16 *data, int n)
{
  data[-1] = data[0];

  data[n] = data[n - 1];
}

static void
extend_1_0 (orc_int16 *data, int n)
{
  data[-1] = data[0];
}

static void
extend_2_2 (orc_int16 *data, int n)
{
  data[-2] = data[0];
  data[-1] = data[0];

  data[n] = data[n - 1];
  data[n+1] = data[n - 1];
}

#define extend_4_3 extend_4_4
#define extend_3_4 extend_4_4
static void
extend_4_4 (orc_int16 *data, int n)
{
  data[-4] = data[0];
  data[-3] = data[0];
  data[-2] = data[0];
  data[-1] = data[0];

  data[n] = data[n - 1];
  data[n+1] = data[n - 1];
  data[n+2] = data[n - 1];
  data[n+3] = data[n - 1];
}

#define extend_1_0_s32 extend_4_4_s32
#define extend_1_1_s32 extend_4_4_s32
#define extend_2_2_s32 extend_4_4_s32
#define extend_1_2_s32 extend_4_4_s32
#define extend_2_1_s32 extend_4_4_s32
#define extend_4_3_s32 extend_4_4_s32
#define extend_3_4_s32 extend_4_4_s32
static void
extend_4_4_s32 (orc_int32 *data, int n)
{
  data[-4] = data[0];
  data[-3] = data[0];
  data[-2] = data[0];
  data[-1] = data[0];

  data[n] = data[n - 1];
  data[n+1] = data[n - 1];
  data[n+2] = data[n - 1];
  data[n+3] = data[n - 1];
}

static void
join (orc_int16 *dest, orc_int16 *src1, orc_int16 *src2, int width)
{
  orc_memcpy (dest, src1, width / 2 * sizeof (int16_t));
  orc_memcpy (dest + width / 2, src2, width / 2 * sizeof (int16_t));
}

static void
join_s32 (orc_int32 *dest, orc_int32 *src1, orc_int32 *src2, int width)
{
  orc_memcpy (dest, src1, width / 2 * sizeof (int32_t));
  orc_memcpy (dest + width / 2, src2, width / 2 * sizeof (int32_t));
}

/* Forward, 16-bit, Wavelet #0: Deslauriers-Dubuc 9,7 */

static void
wavelet_iwt_desl_9_3_horiz (SchroFrameData * fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s16 (hi, lo, line, width / 2);
  extend_1_2 (hi, width/2);
  orc_mas4_horiz_sub_s16_1991_ip (lo, hi - 1, 1 << 3, 4, width/2);
  extend_1_0 (lo, width/2);
  orc_add2_rshift_add_s16_22 (hi, lo - 1, width/2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_desl_9_3_vert_odd (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;

  if (i & 1) {
    if (i < 3 || i >= height - 3) {
      orc_mas4_vert_sub_s16_1991 (
          SCHRO_FRAME_DATA_GET_LINE (fd, i),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 3, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 1, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i + 1, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i + 3, 0, height - 2)), 1 << 3, 4, width);
    } else {
      orc_mas4_vert_sub_s16_1991 (
          SCHRO_FRAME_DATA_GET_LINE (fd, i),
          SCHRO_FRAME_DATA_GET_LINE (fd, i - 3),
          SCHRO_FRAME_DATA_GET_LINE (fd, i - 1),
          SCHRO_FRAME_DATA_GET_LINE (fd, i + 1),
          SCHRO_FRAME_DATA_GET_LINE (fd, i + 3), 1 << 3, 4, width);
    }
  }
}

static void
wavelet_iwt_desl_9_3_vert_even (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i+1);

    orc_add2_rshift_add_s16_22_vert (lo, hi1, hi2, width);
  }
}

void
schro_iwt_desl_9_3 (SchroFrameData *fd, int16_t * tmp)
{
  int i;
  int j;

  for(i=-6;i<fd->height;i++) {
    j = i+6;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_desl_9_3_horiz (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_desl_9_3_vert_odd (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_desl_9_3_vert_even (fd, j);
    }
  }

}

/* Forward, 16-bit, Wavelet #1: LeGall 5,3 */

static void
wavelet_iwt_5_3_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s16 (hi, lo, line, width / 2);
  extend_1_1 (hi, width/2);
  orc_add2_rshift_sub_s16_11 (lo, hi, width/2);
  extend_1_1 (lo, width/2);
  orc_add2_rshift_add_s16_22 (hi, lo - 1, width/2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_5_3_vert_odd (SchroFrameData * fd, int i)
{

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE (fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_add2_rshift_sub_s16_11_vert (hi, lo1, lo2, fd->width);
  }
}

static void
wavelet_iwt_5_3_vert_even (SchroFrameData *fd, int i)
{

  if ((i & 1) == 0) {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i - 1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_add2_rshift_add_s16_22_vert (lo, hi1, hi2, fd->width);
  }
}

void
schro_iwt_5_3 (SchroFrameData *fd, int16_t * tmp)
{
  int i;
  int j;

  for(i=-2;i<fd->height;i++) {
    j = i+2;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_5_3_horiz (fd, j, tmp);
    }

    j = i+1;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_5_3_vert_odd (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_5_3_vert_even (fd, j);
    }
  }

}

/* Forward, 16-bit, Wavelet #2: Deslauriers-Dubuc 13,7 */

static void
wavelet_iwt_13_5_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s16 (hi, lo, line, width / 2);
  extend_1_2 (hi, width/2);
  orc_mas4_horiz_sub_s16_1991_ip (lo, hi - 1, 1 << 3, 4, width/2);
  extend_2_1 (lo, width/2);
  orc_mas4_horiz_add_s16_1991_ip (hi, lo - 2, 1 << 4, 5, width/2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_13_5_vert_odd (SchroFrameData *fd, int i)
{
  int16_t *dest, *s1, *s2, *s3, *s4;

  if (i & 1) {
    if (i < 3 || i >= fd->height - 3) {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 3, 0, fd->height - 2));
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 1, 0, fd->height - 2));
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 1, 0, fd->height - 2));
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 3, 0, fd->height - 2));
    } else {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 3);
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 3);
    }
    dest = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    orc_mas4_vert_sub_s16_1991 (dest, s1, s2, s3, s4, 1 << 3, 4, fd->width);
  }
}

static void
wavelet_iwt_13_5_vert_even (SchroFrameData *fd, int i)
{
  int16_t *dest, *s1, *s2, *s3, *s4;

  if ((i & 1) == 0) {
    if (i < 3 || i >= fd->height - 3) {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 3, 1, fd->height - 1));
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 1, 1, fd->height - 1));
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 1, 1, fd->height - 1));
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 3, 1, fd->height - 1));
    } else {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 3);
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 3);
    }
    dest = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    orc_mas4_vert_add_s16_1991 (dest, s1, s2, s3, s4, 1 << 4, 5, fd->width);
  }
}

void
schro_iwt_13_5 (SchroFrameData *fd, int16_t * tmp)
{
  int i;
  int j;

  for(i=-6;i<fd->height;i++) {
    j = i+6;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_13_5_horiz (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_13_5_vert_odd (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_13_5_vert_even (fd, j);
    }
  }
}

/* Forward, 16-bit, Wavelet #3,4: Haar 0 and Haar 1 */

static void
wavelet_iwt_haar_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + width / 2;
  int16_t *lo = tmp;

  orc_haar_deint_split_s16 (lo, hi, line, width / 2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_haar_shift1_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + width / 2;
  int16_t *lo = tmp;

  orc_haar_deint_lshift1_split_s16 (lo, hi, line, width / 2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_haar_vert (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int16_t *hi = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *lo = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

  orc_haar_split_s16_op (hi, lo, tmp, tmp + fd->width, fd->width);
}

static void
schro_iwt_haar0 (SchroFrameData *fd, int16_t * tmp)
{
  int i;

  for(i=0;i<fd->height;i+=2) {
    wavelet_iwt_haar_horiz (fd, i, tmp);
    wavelet_iwt_haar_horiz (fd, i+1, tmp + fd->width);

    wavelet_iwt_haar_vert (fd, i, tmp);
  }
}

static void
schro_iwt_haar1 (SchroFrameData *fd, int16_t * tmp)
{
  int i;

  for(i=0;i<fd->height;i+=2) {
    wavelet_iwt_haar_shift1_horiz (fd, i, tmp);
    wavelet_iwt_haar_shift1_horiz (fd, i+1, tmp + fd->width);

    wavelet_iwt_haar_vert (fd, i, tmp);
  }
}

/* Forward, 16-bit, Wavelet #5: Fidelity */

static void
mas8_add_s16 (int16_t * dest, const int16_t * src, const int16_t * weights,
    int offset, int shift, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    int x = offset;
    x += src[i + 0] * weights[0];
    x += src[i + 1] * weights[1];
    x += src[i + 2] * weights[2];
    x += src[i + 3] * weights[3];
    x += src[i + 4] * weights[4];
    x += src[i + 5] * weights[5];
    x += src[i + 6] * weights[6];
    x += src[i + 7] * weights[7];
    dest[i] += x >> shift;
  }
}

static void
schro_split_ext_fidelity (int16_t * hi, int16_t * lo, int n)
{
  static const int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static const int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };

  extend_4_3 (lo, n);
  mas8_add_s16 (hi, lo - 4, stage1_weights, 128, 8, n);
  extend_3_4 (hi, n);
  mas8_add_s16 (lo, hi - 3, stage2_weights, 127, 8, n);
}

static void
wavelet_iwt_fidelity_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_s16 (hi, lo, line, width / 2);
  schro_split_ext_fidelity (hi, lo, width / 2);
  join (line, hi, lo, width);
}

static void
mas8_vert_add_s16_2 (int16_t * dest, const int16_t * src,
    int16_t ** s, const int *weights, int offset, int shift, int n)
{
  int i;
  int j;
  for (i = 0; i < n; i++) {
    int x = offset;
    for (j = 0; j < 8; j++) {
      x += s[j][i] * weights[j];
    }
    dest[i] = src[i] + (x >> shift);
  }
}

static void
wavelet_iwt_fidelity_vert_odd (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;
  int16_t *s[8];
  int j;

  if (i & 1) {
    static const int weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
    for (j = 0; j < 8; j++) {
      s[j] = SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 7 + j * 2, 0, height - 2));
    }
    mas8_vert_add_s16_2 (SCHRO_FRAME_DATA_GET_LINE (fd, i),
        SCHRO_FRAME_DATA_GET_LINE (fd, i), s,
        weights, 127, 8, width);
  }
}

static void
wavelet_iwt_fidelity_vert_even (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;
  int16_t *s[8];
  int j;

  if ((i & 1) == 0) {
    static const int weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
    for (j = 0; j < 8; j++) {
      s[j] = SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 7 + j * 2, 1, height - 1));
    }
    mas8_vert_add_s16_2 (SCHRO_FRAME_DATA_GET_LINE (fd, i),
        SCHRO_FRAME_DATA_GET_LINE (fd, i), s,
        weights, 128, 8, width);
  }
}

void
schro_iwt_fidelity (SchroFrameData *fd, int16_t * tmp)
{
  int i;
  int j;

  for(i=-14;i<fd->height;i++) {
    j = i+14;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_fidelity_horiz (fd, j, tmp);
    }

    j = i+7;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_fidelity_vert_even (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_fidelity_vert_odd (fd, j);
    }
  }
}

/* Forward, 16-bit, Wavelet #6: Daubechies 9,7 */

static void
schro_split_ext_daub97 (int16_t * hi, int16_t * lo, int n)
{
  extend_1_1 (hi, n);
  orc_mas2_sub_s16_ip (lo, hi, 6497, 2048, 12, n);
  extend_1_1 (lo, n);
  orc_mas2_sub_s16_ip (hi, lo - 1, 217, 2048, 12, n);
  extend_1_1 (hi, n);
  orc_mas2_add_s16_ip (lo, hi, 3616, 2048, 12, n);
  extend_1_1 (lo, n);
  orc_mas2_add_s16_ip (hi, lo - 1, 1817, 2048, 12, n);
}

static void
wavelet_iwt_daub97_horiz (SchroFrameData *fd, int i, orc_int16 *tmp)
{
  int width = fd->width;
  int16_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s16 (hi, lo, line, width / 2);
  schro_split_ext_daub97 (hi, lo, width / 2);
  join (line, hi, lo, width);
}

static void
wavelet_iwt_daub97_vert1_odd (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_sub_s16_op (hi, hi, lo1, lo2, 6497, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert1_even (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_mas2_sub_s16_op (lo, lo, hi1, hi2, 217, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert2_odd (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_add_s16_op (hi, hi, lo1, lo2, 3616, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert2_even (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_mas2_add_s16_op (lo, lo, hi1, hi2, 1817, 2048, 12, width);
  }
}

void
schro_iwt_daub_9_7 (SchroFrameData *fd, int16_t * tmp)
{
  int i;
  int j;

  for(i=-4;i<fd->height;i++) {
    j = i+4;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_daub97_horiz (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_daub97_vert1_odd (fd, j);
    }

    j = i+2;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_daub97_vert1_even (fd, j);
    }

    j = i+1;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_daub97_vert2_odd (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_daub97_vert2_even (fd, j);
    }
  }
}

/* Forward, 32-bit, Wavelet #0: Deslauriers-Dubuc 9,7 */

static void
wavelet_iwt_desl_9_3_horiz_s32 (SchroFrameData * fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s32 (hi, lo, line, width / 2);
  extend_1_2_s32 (hi, width/2);
  orc_mas4_horiz_sub_s32_1991_ip (lo, hi - 1, 1 << 3, 4, width/2);
  extend_1_0_s32 (lo, width/2);
  orc_add2_rshift_add_s32_22 (hi, lo - 1, width/2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_desl_9_3_vert_odd_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;

  if (i & 1) {
    if (i < 3 || i >= height - 3) {
      orc_mas4_vert_sub_s32_1991 (
          SCHRO_FRAME_DATA_GET_LINE (fd, i),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 3, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 1, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i + 1, 0, height - 2)),
          SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i + 3, 0, height - 2)), 1 << 3, 4, width);
    } else {
      orc_mas4_vert_sub_s32_1991 (
          SCHRO_FRAME_DATA_GET_LINE (fd, i),
          SCHRO_FRAME_DATA_GET_LINE (fd, i - 3),
          SCHRO_FRAME_DATA_GET_LINE (fd, i - 1),
          SCHRO_FRAME_DATA_GET_LINE (fd, i + 1),
          SCHRO_FRAME_DATA_GET_LINE (fd, i + 3), 1 << 3, 4, width);
    }
  }
}

static void
wavelet_iwt_desl_9_3_vert_even_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i+1);

    orc_add2_rshift_add_s32_22_vert (lo, hi1, hi2, width);
  }
}

void
schro_iwt_desl_9_3_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;
  int j;

  for(i=-6;i<fd->height;i++) {
    j = i+6;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_desl_9_3_horiz_s32 (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_desl_9_3_vert_odd_s32 (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_desl_9_3_vert_even_s32 (fd, j);
    }
  }

}

/* Forward, 32-bit, Wavelet #1: LeGall 5,3 */

static void
wavelet_iwt_5_3_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s32 (hi, lo, line, width / 2);
  extend_1_1_s32 (hi, width/2);
  orc_add2_rshift_sub_s32_11 (lo, hi, width/2);
  extend_1_1_s32 (lo, width/2);
  orc_add2_rshift_add_s32_22 (hi, lo - 1, width/2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_5_3_vert_odd_s32 (SchroFrameData * fd, int i)
{

  if (i & 1) {
    int32_t *hi;
    int32_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE (fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_add2_rshift_sub_s32_11_vert (hi, lo1, lo2, fd->width);
  }
}

static void
wavelet_iwt_5_3_vert_even_s32 (SchroFrameData *fd, int i)
{

  if ((i & 1) == 0) {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE (fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i - 1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_add2_rshift_add_s32_22_vert (lo, hi1, hi2, fd->width);
  }
}

void
schro_iwt_5_3_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;
  int j;

  for(i=-6;i<fd->height;i++) {
    j = i+6;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_5_3_horiz_s32 (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_5_3_vert_odd_s32 (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_5_3_vert_even_s32 (fd, j);
    }
  }

}

/* Forward, 32-bit, Wavelet #2: Deslauriers-Dubuc 13,7 */

static void
wavelet_iwt_13_5_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s32 (hi, lo, line, width / 2);
  extend_1_2_s32 (hi, width/2);
  orc_mas4_horiz_sub_s32_1991_ip (lo, hi - 1, 1 << 3, 4, width/2);
  extend_2_1_s32 (lo, width/2);
  orc_mas4_horiz_add_s32_1991_ip (hi, lo - 2, 1 << 4, 5, width/2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_13_5_vert_odd_s32 (SchroFrameData *fd, int i)
{
  int32_t *dest, *s1, *s2, *s3, *s4;

  if (i & 1) {
    if (i < 3 || i >= fd->height - 3) {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 3, 0, fd->height - 2));
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 1, 0, fd->height - 2));
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 1, 0, fd->height - 2));
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 3, 0, fd->height - 2));
    } else {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 3);
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 3);
    }
    dest = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    orc_mas4_vert_sub_s32_1991 (dest, s1, s2, s3, s4, 1 << 3, 4, fd->width);
  }
}

static void
wavelet_iwt_13_5_vert_even_s32 (SchroFrameData *fd, int i)
{
  int32_t *dest, *s1, *s2, *s3, *s4;

  if ((i & 1) == 0) {
    if (i < 3 || i >= fd->height - 3) {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 3, 1, fd->height - 1));
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i - 1, 1, fd->height - 1));
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 1, 1, fd->height - 1));
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, CLAMP (i + 3, 1, fd->height - 1));
    } else {
      s1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 3);
      s2 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
      s3 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
      s4 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 3);
    }
    dest = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    orc_mas4_vert_add_s32_1991 (dest, s1, s2, s3, s4, 1 << 4, 5, fd->width);
  }
}

void
schro_iwt_13_5_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;
  int j;

  for(i=-6;i<fd->height;i++) {
    j = i+6;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_13_5_horiz_s32 (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_13_5_vert_odd_s32 (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_13_5_vert_even_s32 (fd, j);
    }
  }
}

/* Forward, 32-bit, Wavelet #3,4: Haar 0 and Haar 1 */

static void
wavelet_iwt_haar_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + width / 2;
  int32_t *lo = tmp;

  orc_haar_deint_split_s32 (lo, hi, line, width / 2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_haar_shift1_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + width / 2;
  int32_t *lo = tmp;

  orc_haar_deint_lshift1_split_s32 (lo, hi, line, width / 2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_haar_vert_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int32_t *hi = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *lo = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

  orc_haar_split_s32_op (hi, lo, tmp, tmp + fd->width, fd->width);
}

static void
schro_iwt_haar0_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;

  for(i=0;i<fd->height;i+=2) {
    wavelet_iwt_haar_horiz_s32 (fd, i, tmp);
    wavelet_iwt_haar_horiz_s32 (fd, i+1, tmp + fd->width);

    wavelet_iwt_haar_vert_s32 (fd, i, tmp);
  }
}

static void
schro_iwt_haar1_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;

  for(i=0;i<fd->height;i+=2) {
    wavelet_iwt_haar_shift1_horiz_s32 (fd, i, tmp);
    wavelet_iwt_haar_shift1_horiz_s32 (fd, i+1, tmp + fd->width);

    wavelet_iwt_haar_vert_s32 (fd, i, tmp);
  }
}

/* Forward, 32-bit, Wavelet #5: Fidelity */

static void
mas8_add_s32 (int32_t * dest, const int32_t * src, const int32_t * weights,
    int offset, int shift, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    int x = offset;
    x += src[i + 0] * weights[0];
    x += src[i + 1] * weights[1];
    x += src[i + 2] * weights[2];
    x += src[i + 3] * weights[3];
    x += src[i + 4] * weights[4];
    x += src[i + 5] * weights[5];
    x += src[i + 6] * weights[6];
    x += src[i + 7] * weights[7];
    dest[i] += x >> shift;
  }
}

static void
schro_split_ext_fidelity_s32 (int32_t * hi, int32_t * lo, int n)
{
  static const int32_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static const int32_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };

  extend_4_3_s32 (lo, n);
  mas8_add_s32 (hi, lo - 4, stage1_weights, 128, 8, n);
  extend_3_4_s32 (hi, n);
  mas8_add_s32 (lo, hi - 3, stage2_weights, 127, 8, n);
}

static void
wavelet_iwt_fidelity_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_s32 (hi, lo, line, width / 2);
  schro_split_ext_fidelity_s32 (hi, lo, width / 2);
  join_s32 (line, hi, lo, width);
}

static void
mas8_vert_add_s32_2 (int32_t * dest, const int32_t * src,
    int32_t ** s, const int32_t *weights, int offset, int shift, int n)
{
  int i;
  int j;
  for (i = 0; i < n; i++) {
    int x = offset;
    for (j = 0; j < 8; j++) {
      x += s[j][i] * weights[j];
    }
    dest[i] = src[i] + (x >> shift);
  }
}

static void
wavelet_iwt_fidelity_vert_odd_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;
  int32_t *s[8];
  int j;

  if (i & 1) {
    static const int32_t weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
    for (j = 0; j < 8; j++) {
      s[j] = SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 7 + j * 2, 0, height - 2));
    }
    mas8_vert_add_s32_2 (SCHRO_FRAME_DATA_GET_LINE (fd, i),
        SCHRO_FRAME_DATA_GET_LINE (fd, i), s,
        weights, 127, 8, width);
  }
}

static void
wavelet_iwt_fidelity_vert_even_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;
  int height = fd->height;
  int32_t *s[8];
  int j;

  if ((i & 1) == 0) {
    static const int32_t weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
    for (j = 0; j < 8; j++) {
      s[j] = SCHRO_FRAME_DATA_GET_LINE (fd, CLAMP (i - 7 + j * 2, 1, height - 1));
    }
    mas8_vert_add_s32_2 (SCHRO_FRAME_DATA_GET_LINE (fd, i),
        SCHRO_FRAME_DATA_GET_LINE (fd, i), s,
        weights, 128, 8, width);
  }
}

void
schro_iwt_fidelity_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;
  int j;

  for(i=-14;i<fd->height;i++) {
    j = i+14;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_fidelity_horiz_s32 (fd, j, tmp);
    }

    j = i+7;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_fidelity_vert_even_s32 (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_fidelity_vert_odd_s32 (fd, j);
    }
  }
}

/* Forward, 32-bit, Wavelet #6: Daubechies 9,7 */

static void
schro_split_ext_daub97_s32 (int32_t * hi, int32_t * lo, int n)
{
  extend_1_1_s32 (hi, n);
  orc_mas2_sub_s32_ip (lo, hi, 6497, 2048, 12, n);
  extend_1_1_s32 (lo, n);
  orc_mas2_sub_s32_ip (hi, lo - 1, 217, 2048, 12, n);
  extend_1_1_s32 (hi, n);
  orc_mas2_add_s32_ip (lo, hi, 3616, 2048, 12, n);
  extend_1_1_s32 (lo, n);
  orc_mas2_add_s32_ip (hi, lo - 1, 1817, 2048, 12, n);
}

static void
wavelet_iwt_daub97_horiz_s32 (SchroFrameData *fd, int i, orc_int32 *tmp)
{
  int width = fd->width;
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_deinterleave2_lshift1_s32 (hi, lo, line, width / 2);
  schro_split_ext_daub97_s32 (hi, lo, width / 2);
  join_s32 (line, hi, lo, width);
}

static void
wavelet_iwt_daub97_vert1_odd_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if (i & 1) {
    int32_t *hi;
    int32_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_sub_s32_op (hi, hi, lo1, lo2, 6497, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert1_even_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_mas2_sub_s32_op (lo, lo, hi1, hi2, 217, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert2_odd_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if (i & 1) {
    int32_t *hi;
    int32_t *lo1, *lo2;

    hi = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    lo1 = SCHRO_FRAME_DATA_GET_LINE(fd, i - 1);
    if (i + 1 < fd->height) {
      lo2 = SCHRO_FRAME_DATA_GET_LINE(fd, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_add_s32_op (hi, hi, lo1, lo2, 3616, 2048, 12, width);
  }
}

static void
wavelet_iwt_daub97_vert2_even_s32 (SchroFrameData *fd, int i)
{
  int width = fd->width;

  if ((i & 1) == 0) {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = SCHRO_FRAME_DATA_GET_LINE(fd, i);
    if (i == 0) {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, 1);
    } else {
      hi1 = SCHRO_FRAME_DATA_GET_LINE (fd, i-1);
    }
    hi2 = SCHRO_FRAME_DATA_GET_LINE (fd, i + 1);

    orc_mas2_add_s32_op (lo, lo, hi1, hi2, 1817, 2048, 12, width);
  }
}

void
schro_iwt_daub_9_7_s32 (SchroFrameData *fd, int32_t * tmp)
{
  int i;
  int j;

  for(i=-4;i<fd->height;i++) {
    j = i+4;
    if (j >= 0 && j < fd->height) {
      wavelet_iwt_daub97_horiz_s32 (fd, j, tmp);
    }

    j = i+3;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_daub97_vert1_odd_s32 (fd, j);
    }

    j = i+2;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_daub97_vert1_even_s32 (fd, j);
    }

    j = i+1;
    if (j >= 0 && j < fd->height && (j&1) == 1) {
      wavelet_iwt_daub97_vert2_odd_s32 (fd, j);
    }

    j = i;
    if (j >= 0 && j < fd->height && (j&1) == 0) {
      wavelet_iwt_daub97_vert2_even_s32 (fd, j);
    }
  }
}


/* Reverse transforms */

/* Reverse, 16-bit, Wavelet #0: Deslauriers-Dubuc 9,7 */

static void
schro_synth_ext_desl93 (int16_t * hi, int16_t * lo, int n)
{
  extend_2_2 (lo, n);
  orc_add2_rshift_sub_s16_22 (hi, lo - 1, n);
  extend_2_2 (hi, n);
  orc_mas4_horiz_add_s16_1991_ip (lo, hi - 1, 1 << 3, 4, n);
}

void
schro_iiwt_desl_9_3 (SchroFrameData *dest, SchroFrameData *src, int16_t * tmp)
{
  int i;
  int j;

  for(i=-7;i<dest->height;i++){
    j = i + 7;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        int16_t *lo;
        int16_t *hi1, *hi2;

        lo = SCHRO_FRAME_DATA_GET_LINE (src, j);
        if (j == 0) {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, 1);
        } else {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        }
        hi2 = SCHRO_FRAME_DATA_GET_LINE (src, j+1);

        orc_add2_rshift_sub_s16_22_vert (lo, hi1, hi2, src->width);
      }
    }

    j = i + 3;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        if (j < 3 || j >= src->height - 3) {
          orc_mas4_vert_add_s16_1991 (
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j - 3, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j - 1, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j + 1, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j + 3, 0, src->height - 2)),
              1 << 3, 4, src->width);
        } else {
          orc_mas4_vert_add_s16_1991 (
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, j - 3),
              SCHRO_FRAME_DATA_GET_LINE (src, j - 1),
              SCHRO_FRAME_DATA_GET_LINE (src, j + 1),
              SCHRO_FRAME_DATA_GET_LINE (src, j + 3),
              1 << 3, 4, src->width);
        }
      }
    }

    j = i;
    /* horizontal wavelet */
    if (j == CLAMP(j,0,src->height-1)) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + src->width/2 + 12;

      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S16(src, 0, j),
          src->width / 2 * sizeof (int16_t));
      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_PIXEL_S16(src, src->width/2, j),
          src->width / 2 * sizeof (int16_t));
      schro_synth_ext_desl93 (hi, lo, src->width / 2);
      orc_interleave2_rrshift1_s16 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (dest, 0, j), hi, lo, src->width / 2);
    }
  }
}

/* Reverse, 16-bit, Wavelet #1: LeGall 5,3 */

static void
schro_synth_ext_53 (int16_t * hi, int16_t * lo, int n)
{
  extend_1_1 (lo, n);
  orc_add2_rshift_sub_s16_22 (hi, lo - 1, n);
  extend_1_1 (hi, n);
  orc_add2_rshift_add_s16_11 (lo, hi, n);
}

void
schro_iiwt_5_3 (SchroFrameData *dest, SchroFrameData *src, int16_t * tmp)
{
  int i;
  int j;

  for(i=-2;i<dest->height;i++){
    j = i + 2;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        int16_t *lo;
        int16_t *hi1, *hi2;

        lo = SCHRO_FRAME_DATA_GET_LINE (src, j);
        if (j == 0) {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, 1);
        } else {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        }
        hi2 = SCHRO_FRAME_DATA_GET_LINE (src, j+1);

        orc_add2_rshift_sub_s16_22_vert (lo, hi1, hi2, src->width);
      }
    }

    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        int16_t *hi;
        int16_t *lo1, *lo2;

        hi = SCHRO_FRAME_DATA_GET_LINE (src, j);
        lo1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        if (j + 1 < src->height) {
          lo2 = SCHRO_FRAME_DATA_GET_LINE (src, j + 1);
        } else {
          lo2 = lo1;
        }

        orc_add2_rshift_add_s16_11_op (SCHRO_FRAME_DATA_GET_LINE (src, j),
            hi, lo1, lo2, src->width);
      }
    }

    j = i;
    /* horizontal wavelet */
    if (j == CLAMP(j,0,src->height-1)) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + src->width/2 + 12;

      orc_memcpy (hi,
          SCHRO_FRAME_DATA_GET_PIXEL_S16(src, 0, j),
          src->width / 2 * sizeof (int16_t));
      orc_memcpy (lo,
          SCHRO_FRAME_DATA_GET_PIXEL_S16(src, src->width/2, j),
          src->width / 2 * sizeof (int16_t));
      schro_synth_ext_53 (hi, lo, src->width / 2);
      orc_interleave2_rrshift1_s16 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16 (dest, 0, j), hi, lo, src->width / 2);
    }
  }
}

/* Reverse, 16-bit, Wavelet #2: */

static void
schro_synth_ext_135 (int16_t * hi, int16_t * lo, int n)
{
  extend_2_1 (lo, n);
  orc_mas4_horiz_sub_s16_1991_ip (hi, lo - 2, 1 << 4, 5, n);
  extend_1_2 (hi, n);
  orc_mas4_horiz_add_s16_1991_ip (lo, hi - 1, 1 << 3, 4, n);
}

void
schro_iiwt_13_5 (SchroFrameData *dest, SchroFrameData *src,
    int16_t * tmp)
{
  int i;
  int j;
  int16_t *srcline, *s1, *s2, *s3, *s4;
  int height = src->height;
  int width = src->width;

#define ROW(x) SCHRO_FRAME_DATA_GET_LINE (src, (x))
  for(i=-8;i<dest->height;i++){
    j = i + 8;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        if (j < 3 || j >= height - 3) {
          s1 = ROW (CLAMP (j - 3, 1, height - 1));
          s2 = ROW (CLAMP (j - 1, 1, height - 1));
          srcline = ROW (j);
          s3 = ROW (CLAMP (j + 1, 1, height - 1));
          s4 = ROW (CLAMP (j + 3, 1, height - 1));
        } else {
          s1 = ROW (j - 3);
          s2 = ROW (j - 1);
          srcline = ROW (j);
          s3 = ROW (j + 1);
          s4 = ROW (j + 3);
        }
        orc_mas4_vert_sub_s16_1991 (srcline, s1, s2, s3, s4, 1 << 4, 5, width);
      }
    }

    j = i + 4;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        if (j < 3 || j >= height - 3) {
          s1 = ROW (CLAMP (j - 3, 0, height - 2));
          s2 = ROW (CLAMP (j - 1, 0, height - 2));
          srcline = ROW (j);
          s3 = ROW (CLAMP (j + 1, 0, height - 2));
          s4 = ROW (CLAMP (j + 3, 0, height - 2));
        } else {
          s1 = ROW (j - 3);
          s2 = ROW (j - 1);
          srcline = ROW (j);
          s3 = ROW (j + 1);
          s4 = ROW (j + 3);
        }
        orc_mas4_vert_add_s16_1991 (srcline, s1, s2, s3, s4, 1 << 3, 4, width);
      }
    }
#undef ROW

    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + width/2 + 12;

      srcline = SCHRO_FRAME_DATA_GET_LINE(dest, j);

      orc_memcpy (hi, srcline, width / 2 * sizeof (int16_t));
      orc_memcpy (lo, srcline + width / 2, width / 2 * sizeof (int16_t));
      schro_synth_ext_135 (hi, lo, width / 2);
      orc_interleave2_rrshift1_s16 (
          SCHRO_FRAME_DATA_GET_LINE(dest, j), hi, lo, width / 2);
    }
  }

}

/* Reverse, 16-bit, Wavelet #3,4: Haar 0 and Haar 1 */

void
schro_iiwt_haar0 (SchroFrameData *dest, SchroFrameData *src, int16_t * tmp)
{
  int i;
  int j;
  int width = src->width;

  for(i=-8;i<dest->height;i++){
    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        orc_haar_synth_s16 (
            SCHRO_FRAME_DATA_GET_LINE (src, j),
            SCHRO_FRAME_DATA_GET_LINE (src, j+1),
            width);
      }
    }
      
    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + width/2 + 12;

      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_LINE(dest,j),
          width / 2 * sizeof (int16_t));
      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S16(src, src->width/2, j),
          width / 2 * sizeof (int16_t));
      orc_haar_synth_int_s16 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16(dest, 0, j), lo, hi,
          width / 2);
    }
  }
}

void
schro_iiwt_haar1 (SchroFrameData *dest, SchroFrameData *src,
    int16_t * tmp)
{
  int i;
  int j;
  int width = src->width;

  for(i=-8;i<dest->height;i++){
    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        orc_haar_synth_s16 (
            SCHRO_FRAME_DATA_GET_LINE (src, j),
            SCHRO_FRAME_DATA_GET_LINE (src, j+1),
            width);
      }
    }
      
    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + width/2 + 12;

      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_LINE(dest,j),
          width / 2 * sizeof (int16_t));
      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S16(src, src->width/2, j),
          width / 2 * sizeof (int16_t));
      orc_haar_synth_rrshift1_int_s16 (
          SCHRO_FRAME_DATA_GET_PIXEL_S16(dest, 0, j), lo, hi,
          width / 2);
    }
  }
}

/* Reverse, 16-bit, Wavelet #5: Fidelity */

static void
schro_synth_ext_fidelity (int16_t * hi, int16_t * lo, int n)
{
  static const int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static const int16_t stage2_weights[] =
      { 8, -21, 46, -161, -161, 46, -21, 8 };

  extend_3_4 (hi, n);
  mas8_add_s16 (lo, hi - 3, stage1_weights, 128, 8, n);
  extend_4_3 (lo, n);
  mas8_add_s16 (hi, lo - 4, stage2_weights, 127, 8, n);
}

static void
wavelet_iiwt_fidelity_horiz (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *tmp = frame->virt_priv2;
  int16_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_memcpy (hi, src, width / 2 * sizeof (int16_t));
  orc_memcpy (lo, src + width / 2, width / 2 * sizeof (int16_t));
  schro_synth_ext_fidelity (hi, lo, width / 2);
  orc_interleave2_s16 (dest, hi, lo, width / 2);
}

static void
mas8_vert_sub_s16_2 (int16_t * dest, const int16_t * src,
    int16_t ** s, const int *weights, int offset, int shift, int n)
{
  int i;
  int j;
  for (i = 0; i < n; i++) {
    int x = offset;
    for (j = 0; j < 8; j++) {
      x += s[j][i] * weights[j];
    }
    dest[i] = src[i] - (x >> shift);
  }
}

static void
wavelet_iiwt_fidelity_vert (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int height = frame->components[component].height;
  int16_t *s[8];
  int j;

  if (i & 1) {
    static const int weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
#define ROW(x) \
  schro_virt_frame_get_line (frame->virt_frame1, component, (x))
#define ROW2(x) \
  schro_virt_frame_get_line (frame, component, (x))
    for (j = 0; j < 8; j++) {
      s[j] = ROW (CLAMP (i - 7 + j * 2, 0, height - 2));
    }
    mas8_vert_sub_s16_2 (dest, ROW (i), s, weights, 127, 8, width);
  } else {
    static const int weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
    for (j = 0; j < 8; j++) {
      s[j] = ROW2 (CLAMP (i - 7 + j * 2, 1, height - 1));
    }
    mas8_vert_sub_s16_2 (dest, ROW (i), s, weights, 128, 8, width);
  }
#undef ROW
#undef ROW2
}

void
schro_iiwt_fidelity (SchroFrameData *dest, SchroFrameData *src,
    int16_t * tmp)
{
  SchroFrame *frame;
  SchroFrame *frame2;
  SchroFrame *vf1;
  SchroFrame *vf2;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S16_444;
  frame->width = src->width;
  frame->height = src->height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame->components[0].width = src->width;
  frame->components[0].height = src->height;
  frame->components[0].stride = src->stride;
  frame->components[0].data = src->data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf1->virt_frame1 = frame;
  vf1->render_line = wavelet_iiwt_fidelity_vert;

  vf2 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iiwt_fidelity_horiz;

  frame2 = schro_frame_new ();

  frame2->format = SCHRO_FRAME_FORMAT_S16_444;
  frame2->width = dest->width;
  frame2->height = dest->height;

  frame2->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame2->components[0].width = dest->width;
  frame2->components[0].height = dest->height;
  frame2->components[0].stride = dest->stride;
  frame2->components[0].data = dest->data;

  schro_virt_frame_render (vf2, frame2);

  schro_frame_unref (vf2);
  schro_frame_unref (frame2);
}

/* Reverse, 16-bit, Wavelet #6: Daubechies 9,7 */

static void
schro_synth_ext_daub97 (int16_t * hi, int16_t * lo, int n)
{
  extend_1_1 (lo, n);
  orc_mas2_sub_s16_ip (hi, lo - 1, 1817, 2048, 12, n);
  extend_1_1 (hi, n);
  orc_mas2_sub_s16_ip (lo, hi, 3616, 2048, 12, n);
  extend_1_1 (lo, n);
  orc_mas2_add_s16_ip (hi, lo - 1, 217, 2048, 12, n);
  extend_1_1 (hi, n);
  orc_mas2_add_s16_ip (lo, hi, 6497, 2048, 12, n);
}

static void
wavelet_iiwt_daub97_horiz (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *tmp = frame->virt_priv2;
  int16_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  int16_t *hi = tmp + 4;
  int16_t *lo = tmp + width/2 + 12;

  orc_memcpy (hi, src, width / 2 * sizeof (int16_t));
  orc_memcpy (lo, src + width / 2, width / 2 * sizeof (int16_t));
  schro_synth_ext_daub97 (hi, lo, width / 2);
  orc_interleave2_rrshift1_s16 (dest, hi, lo, width / 2);
}

static void
wavelet_iiwt_daub97_vert1 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame, component, i - 1);
    if (i + 1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame, component, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_add_s16_op (dest, hi, lo1, lo2, 6497, 2048, 12, width);
  } else {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, i - 1);
    }
    hi2 = schro_virt_frame_get_line (frame->virt_frame1, component, i + 1);

    orc_mas2_add_s16_op (dest, lo, hi1, hi2, 217, 2048, 12, width);
  }
}

static void
wavelet_iiwt_daub97_vert2 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame, component, i - 1);
    if (i + 1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame, component, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_sub_s16_op (dest, hi, lo1, lo2, 3616, 2048, 12, width);
  } else {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, i - 1);
    }
    hi2 = schro_virt_frame_get_line (frame->virt_frame1, component, i + 1);

    orc_mas2_sub_s16_op (dest, lo, hi1, hi2, 1817, 2048, 12, width);
  }
}

void
schro_iiwt_daub_9_7 (SchroFrameData *dest, SchroFrameData *src,
    int16_t * tmp)
{
  SchroFrame *frame;
  SchroFrame *frame2;
  SchroFrame *vf1;
  SchroFrame *vf2;
  SchroFrame *vf3;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S16_444;
  frame->width = src->width;
  frame->height = src->height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame->components[0].width = src->width;
  frame->components[0].height = src->height;
  frame->components[0].stride = src->stride;
  frame->components[0].data = src->data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf1->virt_frame1 = frame;
  vf1->virt_priv2 = tmp;
  vf1->render_line = wavelet_iiwt_daub97_vert2;

  vf2 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iiwt_daub97_vert1;

  vf3 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf3->virt_frame1 = vf2;
  vf3->virt_priv2 = tmp;
  vf3->render_line = wavelet_iiwt_daub97_horiz;

  frame2 = schro_frame_new ();

  frame2->format = SCHRO_FRAME_FORMAT_S16_444;
  frame2->width = dest->width;
  frame2->height = dest->height;

  frame2->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame2->components[0].width = dest->width;
  frame2->components[0].height = dest->height;
  frame2->components[0].stride = dest->stride;
  frame2->components[0].data = dest->data;

  schro_virt_frame_render (vf3, frame2);

  schro_frame_unref (vf3);
  schro_frame_unref (frame2);
}

/* 32 bit versions */

/* Reverse, 32-bit, Wavelet #0: Deslauriers-Dubuc 9,7 */

static void
schro_synth_ext_desl93_s32 (int32_t * hi, int32_t * lo, int n)
{
  extend_2_2_s32 (lo, n);
  orc_add2_rshift_sub_s32_22 (hi, lo - 1, n);
  extend_2_2_s32 (hi, n);
  orc_mas4_horiz_add_s32_1991_ip (lo, hi - 1, 1 << 3, 4, n);
}

void
schro_iiwt_desl_9_3_s32 (SchroFrameData *dest, SchroFrameData *src, int32_t * tmp)
{
  int i;
  int j;

  for(i=-7;i<dest->height;i++){
    j = i + 7;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        int32_t *lo;
        int32_t *hi1, *hi2;

        lo = SCHRO_FRAME_DATA_GET_LINE (src, j);
        if (j == 0) {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, 1);
        } else {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        }
        hi2 = SCHRO_FRAME_DATA_GET_LINE (src, j+1);

        orc_add2_rshift_sub_s32_22_op (SCHRO_FRAME_DATA_GET_LINE (src, j),
            lo, hi1, hi2, src->width);
      }
    }

    j = i + 3;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        if (j < 3 || j >= src->height - 3) {
          orc_mas4_vert_add_s32_1991_op (
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j - 3, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j - 1, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j + 1, 0, src->height - 2)),
              SCHRO_FRAME_DATA_GET_LINE (src, CLAMP (j + 3, 0, src->height - 2)),
              1 << 3, 4, src->width);
        } else {
          orc_mas4_vert_add_s32_1991_op (
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, j),
              SCHRO_FRAME_DATA_GET_LINE (src, j - 3),
              SCHRO_FRAME_DATA_GET_LINE (src, j - 1),
              SCHRO_FRAME_DATA_GET_LINE (src, j + 1),
              SCHRO_FRAME_DATA_GET_LINE (src, j + 3),
              1 << 3, 4, src->width);
        }
      }
    }

    j = i;
    /* horizontal wavelet */
    if (j == CLAMP(j,0,src->height-1)) {
      int32_t *hi = tmp + 4;
      int32_t *lo = tmp + src->width/2 + 12;

      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S32(src, 0, j),
          src->width / 2 * sizeof (int32_t));
      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_PIXEL_S32(src, src->width/2, j),
          src->width / 2 * sizeof (int32_t));
      schro_synth_ext_desl93_s32 (hi, lo, src->width / 2);
      orc_interleave2_rrshift1_s32 (
          SCHRO_FRAME_DATA_GET_PIXEL_S32 (dest, 0, j), hi, lo, src->width / 2);
    }
  }
}

/* Reverse, 32-bit, Wavelet #1: LeGall 5,3 */

static void
schro_synth_ext_53_s32 (int32_t * hi, int32_t * lo, int n)
{
  extend_1_1_s32 (lo, n);
  orc_add2_rshift_sub_s32_22 (hi, lo - 1, n);
  extend_1_1_s32 (hi, n);
  orc_add2_rshift_add_s32_11 (lo, hi, n);
}

void
schro_iiwt_5_3_s32 (SchroFrameData *dest, SchroFrameData *src, int32_t * tmp)
{
  int i;
  int j;

  for(i=-2;i<dest->height;i++){
    j = i + 2;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        int32_t *lo;
        int32_t *hi1, *hi2;

        lo = SCHRO_FRAME_DATA_GET_LINE (src, j);
        if (j == 0) {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, 1);
        } else {
          hi1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        }
        hi2 = SCHRO_FRAME_DATA_GET_LINE (src, j+1);

        orc_add2_rshift_sub_s32_22_op (SCHRO_FRAME_DATA_GET_LINE (src, j),
            lo, hi1, hi2, src->width);
      }
    }

    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        int32_t *hi;
        int32_t *lo1, *lo2;

        hi = SCHRO_FRAME_DATA_GET_LINE (src, j);
        lo1 = SCHRO_FRAME_DATA_GET_LINE (src, j-1);
        if (j + 1 < src->height) {
          lo2 = SCHRO_FRAME_DATA_GET_LINE (src, j + 1);
        } else {
          lo2 = lo1;
        }

        orc_add2_rshift_add_s32_11_op (SCHRO_FRAME_DATA_GET_LINE (src, j),
            hi, lo1, lo2, src->width);
      }
    }

    j = i;
    /* horizontal wavelet */
    if (j == CLAMP(j,0,src->height-1)) {
      int32_t *hi = tmp + 4;
      int32_t *lo = tmp + src->width/2 + 12;

      orc_memcpy (hi,
          SCHRO_FRAME_DATA_GET_PIXEL_S32(src, 0, j),
          src->width / 2 * sizeof (int32_t));
      orc_memcpy (lo,
          SCHRO_FRAME_DATA_GET_PIXEL_S32(src, src->width/2, j),
          src->width / 2 * sizeof (int32_t));
      schro_synth_ext_53_s32 (hi, lo, src->width / 2);
      orc_interleave2_rrshift1_s32 (
          SCHRO_FRAME_DATA_GET_PIXEL_S32 (dest, 0, j), hi, lo, src->width / 2);
    }
  }
}

/* Reverse, 32-bit, Wavelet #2: */

static void
schro_synth_ext_135_s32 (int32_t * hi, int32_t * lo, int n)
{
  extend_2_1_s32 (lo, n);
  orc_mas4_horiz_sub_s32_1991_ip (hi, lo - 2, 1 << 4, 5, n);
  extend_1_2_s32 (hi, n);
  orc_mas4_horiz_add_s32_1991_ip (lo, hi - 1, 1 << 3, 4, n);
}

void
schro_iiwt_13_5_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t * tmp)
{
  int i;
  int j;
  int32_t *srcline, *s1, *s2, *s3, *s4;
  int height = src->height;
  int width = src->width;

#define ROW(x) SCHRO_FRAME_DATA_GET_LINE (src, (x))
  for(i=-8;i<dest->height;i++){
    j = i + 8;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        if (j < 3 || j >= height - 3) {
          s1 = ROW (CLAMP (j - 3, 1, height - 1));
          s2 = ROW (CLAMP (j - 1, 1, height - 1));
          srcline = ROW (j);
          s3 = ROW (CLAMP (j + 1, 1, height - 1));
          s4 = ROW (CLAMP (j + 3, 1, height - 1));
        } else {
          s1 = ROW (j - 3);
          s2 = ROW (j - 1);
          srcline = ROW (j);
          s3 = ROW (j + 1);
          s4 = ROW (j + 3);
        }
        orc_mas4_vert_sub_s32_1991_op (srcline,
            srcline, s1, s2, s3, s4, 1 << 4, 5, width);
      }
    }

    j = i + 4;
    if (j == CLAMP(j,0,src->height-1)) {
      if (j & 1) {
        if (j < 3 || j >= height - 3) {
          s1 = ROW (CLAMP (j - 3, 0, height - 2));
          s2 = ROW (CLAMP (j - 1, 0, height - 2));
          srcline = ROW (j);
          s3 = ROW (CLAMP (j + 1, 0, height - 2));
          s4 = ROW (CLAMP (j + 3, 0, height - 2));
        } else {
          s1 = ROW (j - 3);
          s2 = ROW (j - 1);
          srcline = ROW (j);
          s3 = ROW (j + 1);
          s4 = ROW (j + 3);
        }
        orc_mas4_vert_add_s32_1991_op (srcline,
            srcline, s1, s2, s3, s4, 1 << 3, 4, width);
      }
    }
#undef ROW

    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int32_t *hi = tmp + 4;
      int32_t *lo = tmp + width/2 + 12;

      srcline = SCHRO_FRAME_DATA_GET_LINE(dest, j);

      orc_memcpy (hi, srcline, width / 2 * sizeof (int32_t));
      orc_memcpy (lo, srcline + width / 2, width / 2 * sizeof (int32_t));
      schro_synth_ext_135_s32 (hi, lo, width / 2);
      orc_interleave2_rrshift1_s32 (
          SCHRO_FRAME_DATA_GET_LINE(dest, j), hi, lo, width / 2);
    }
  }

}

/* Reverse, 32-bit, Wavelet #3,4: Haar 0 and Haar 1 */

void
schro_iiwt_haar0_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t * tmp)
{
  int i;
  int j;
  int width = src->width;

  for(i=-8;i<dest->height;i++){
    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        orc_haar_synth_s32 (
            SCHRO_FRAME_DATA_GET_LINE (src, j),
            SCHRO_FRAME_DATA_GET_LINE (src, j+1),
            width);
      }
    }
      
    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int32_t *hi = tmp + 4;
      int32_t *lo = tmp + width/2 + 12;

      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_LINE(dest,j),
          width / 2 * sizeof (int32_t));
      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S32(src, src->width/2, j),
          width / 2 * sizeof (int32_t));
      orc_haar_synth_int_s32 (
          SCHRO_FRAME_DATA_GET_PIXEL_S32(dest, 0, j), lo, hi,
          width / 2);
    }
  }
}

void
schro_iiwt_haar1_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t * tmp)
{
  int i;
  int j;
  int width = src->width;

  for(i=-8;i<dest->height;i++){
    j = i + 1;
    if (j == CLAMP(j,0,src->height-1)) {
      if (!(j & 1)) {
        orc_haar_synth_s32 (
            SCHRO_FRAME_DATA_GET_LINE (src, j),
            SCHRO_FRAME_DATA_GET_LINE (src, j+1),
            width);
      }
    }
      
    j = i;
    if (j == CLAMP(j,0,src->height-1)) {
      int32_t *hi = tmp + 4;
      int32_t *lo = tmp + width/2 + 12;

      orc_memcpy (lo, SCHRO_FRAME_DATA_GET_LINE(dest,j),
          width / 2 * sizeof (int32_t));
      orc_memcpy (hi, SCHRO_FRAME_DATA_GET_PIXEL_S32(src, src->width/2, j),
          width / 2 * sizeof (int32_t));
      orc_haar_synth_rrshift1_int_s32 (
          SCHRO_FRAME_DATA_GET_PIXEL_S32(dest, 0, j), lo, hi,
          width / 2);
    }
  }
}

/* Reverse, 32-bit, Wavelet #5: Fidelity */

#if 0
/* already defined as part of forward transform */
static void
mas8_add_s32 (int32_t * dest, const int32_t * src, const int32_t * weights,
    int offset, int shift, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    int x = offset;
    x += src[i + 0] * weights[0];
    x += src[i + 1] * weights[1];
    x += src[i + 2] * weights[2];
    x += src[i + 3] * weights[3];
    x += src[i + 4] * weights[4];
    x += src[i + 5] * weights[5];
    x += src[i + 6] * weights[6];
    x += src[i + 7] * weights[7];
    dest[i] += x >> shift;
  }
}
#endif

static void
schro_synth_ext_fidelity_s32 (int32_t * hi, int32_t * lo, int n)
{
  static const int32_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static const int32_t stage2_weights[] =
      { 8, -21, 46, -161, -161, 46, -21, 8 };

  extend_3_4_s32 (hi, n);
  mas8_add_s32 (lo, hi - 3, stage1_weights, 128, 8, n);
  extend_4_3_s32 (lo, n);
  mas8_add_s32 (hi, lo - 4, stage2_weights, 127, 8, n);
}

static void
wavelet_iiwt_fidelity_horiz_s32 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int32_t *dest = _dest;
  int width = frame->components[component].width;
  int32_t *tmp = frame->virt_priv2;
  int32_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_memcpy (hi, src, width / 2 * sizeof (int32_t));
  orc_memcpy (lo, src + width / 2, width / 2 * sizeof (int32_t));
  schro_synth_ext_fidelity_s32 (hi, lo, width / 2);
  orc_interleave2_s32 (dest, hi, lo, width / 2);
}

static void
mas8_vert_sub_s32_2 (int32_t * dest, const int32_t * src,
    int32_t ** s, const int *weights, int offset, int shift, int n)
{
  int i;
  int j;
  for (i = 0; i < n; i++) {
    int x = offset;
    for (j = 0; j < 8; j++) {
      x += s[j][i] * weights[j];
    }
    dest[i] = src[i] - (x >> shift);
  }
}

static void
wavelet_iiwt_fidelity_vert_s32 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int32_t *dest = _dest;
  int width = frame->components[component].width;
  int height = frame->components[component].height;
  int32_t *s[8];
  int j;

  if (i & 1) {
    static const int weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
#define ROW(x) \
  schro_virt_frame_get_line (frame->virt_frame1, component, (x))
#define ROW2(x) \
  schro_virt_frame_get_line (frame, component, (x))
    for (j = 0; j < 8; j++) {
      s[j] = ROW (CLAMP (i - 7 + j * 2, 0, height - 2));
    }
    mas8_vert_sub_s32_2 (dest, ROW (i), s, weights, 127, 8, width);
  } else {
    static const int weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
    for (j = 0; j < 8; j++) {
      s[j] = ROW2 (CLAMP (i - 7 + j * 2, 1, height - 1));
    }
    mas8_vert_sub_s32_2 (dest, ROW (i), s, weights, 128, 8, width);
  }
#undef ROW
#undef ROW2
}

void
schro_iiwt_fidelity_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t * tmp)
{
  SchroFrame *frame;
  SchroFrame *frame2;
  SchroFrame *vf1;
  SchroFrame *vf2;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S32_444;
  frame->width = src->width;
  frame->height = src->height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S32_444;
  frame->components[0].width = src->width;
  frame->components[0].height = src->height;
  frame->components[0].stride = src->stride;
  frame->components[0].data = src->data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf1->virt_frame1 = frame;
  vf1->render_line = wavelet_iiwt_fidelity_vert_s32;

  vf2 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iiwt_fidelity_horiz_s32;

  frame2 = schro_frame_new ();

  frame2->format = SCHRO_FRAME_FORMAT_S32_444;
  frame2->width = dest->width;
  frame2->height = dest->height;

  frame2->components[0].format = SCHRO_FRAME_FORMAT_S32_444;
  frame2->components[0].width = dest->width;
  frame2->components[0].height = dest->height;
  frame2->components[0].stride = dest->stride;
  frame2->components[0].data = dest->data;

  schro_virt_frame_render (vf2, frame2);

  schro_frame_unref (vf2);
  schro_frame_unref (frame2);
}

/* Reverse, 32-bit, Wavelet #6: Daubechies 9,7 */

static void
schro_synth_ext_daub97_s32 (int32_t * hi, int32_t * lo, int n)
{
  extend_1_1_s32 (lo, n);
  orc_mas2_sub_s32_ip (hi, lo - 1, 1817, 2048, 12, n);
  extend_1_1_s32 (hi, n);
  orc_mas2_sub_s32_ip (lo, hi, 3616, 2048, 12, n);
  extend_1_1_s32 (lo, n);
  orc_mas2_add_s32_ip (hi, lo - 1, 217, 2048, 12, n);
  extend_1_1_s32 (hi, n);
  orc_mas2_add_s32_ip (lo, hi, 6497, 2048, 12, n);
}

static void
wavelet_iiwt_daub97_horiz_s32 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int32_t *dest = _dest;
  int width = frame->components[component].width;
  int32_t *tmp = frame->virt_priv2;
  int32_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  int32_t *hi = tmp + 4;
  int32_t *lo = tmp + width/2 + 12;

  orc_memcpy (hi, src, width / 2 * sizeof (int32_t));
  orc_memcpy (lo, src + width / 2, width / 2 * sizeof (int32_t));
  schro_synth_ext_daub97_s32 (hi, lo, width / 2);
  orc_interleave2_rrshift1_s32 (dest, hi, lo, width / 2);
}

static void
wavelet_iiwt_daub97_vert1_s32 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int32_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int32_t *hi;
    int32_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame, component, i - 1);
    if (i + 1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame, component, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_add_s32_op (dest, hi, lo1, lo2, 6497, 2048, 12, width);
  } else {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, i - 1);
    }
    hi2 = schro_virt_frame_get_line (frame->virt_frame1, component, i + 1);

    orc_mas2_add_s32_op (dest, lo, hi1, hi2, 217, 2048, 12, width);
  }
}

static void
wavelet_iiwt_daub97_vert2_s32 (SchroFrame * frame, void *_dest, int component,
    int i)
{
  int32_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int32_t *hi;
    int32_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame, component, i - 1);
    if (i + 1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame, component, i + 1);
    } else {
      lo2 = lo1;
    }

    orc_mas2_sub_s32_op (dest, hi, lo1, lo2, 3616, 2048, 12, width);
  } else {
    int32_t *lo;
    int32_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, i - 1);
    }
    hi2 = schro_virt_frame_get_line (frame->virt_frame1, component, i + 1);

    orc_mas2_sub_s32_op (dest, lo, hi1, hi2, 1817, 2048, 12, width);
  }
}

void
schro_iiwt_daub_9_7_s32 (SchroFrameData *dest, SchroFrameData *src,
    int32_t * tmp)
{
  SchroFrame *frame;
  SchroFrame *frame2;
  SchroFrame *vf1;
  SchroFrame *vf2;
  SchroFrame *vf3;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S32_444;
  frame->width = src->width;
  frame->height = src->height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S32_444;
  frame->components[0].width = src->width;
  frame->components[0].height = src->height;
  frame->components[0].stride = src->stride;
  frame->components[0].data = src->data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf1->virt_frame1 = frame;
  vf1->virt_priv2 = tmp;
  vf1->render_line = wavelet_iiwt_daub97_vert2_s32;

  vf2 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iiwt_daub97_vert1_s32;

  vf3 = schro_frame_new_virtual (NULL, frame->format, src->width, src->height);
  vf3->virt_frame1 = vf2;
  vf3->virt_priv2 = tmp;
  vf3->render_line = wavelet_iiwt_daub97_horiz_s32;

  frame2 = schro_frame_new ();

  frame2->format = SCHRO_FRAME_FORMAT_S32_444;
  frame2->width = dest->width;
  frame2->height = dest->height;

  frame2->components[0].format = SCHRO_FRAME_FORMAT_S32_444;
  frame2->components[0].width = dest->width;
  frame2->components[0].height = dest->height;
  frame2->components[0].stride = dest->stride;
  frame2->components[0].data = dest->data;

  schro_virt_frame_render (vf3, frame2);

  schro_frame_unref (vf3);
  schro_frame_unref (frame2);
}

