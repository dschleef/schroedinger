
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroorc.h>
#include <schroedinger/schrovirtframe.h>


/**
 * schro_split_ext_desl93:
 * @hi: array of even components / high subband components
 * @lo: array of odd components / low subband components
 * @n: number of
 *
 * Performs an in-place Deslauries 9,3 wavelet transformation on
 * the arrays @hi and @lo containing the even and odd components
 * of the array to be transformed.  This function reads and writes
 * to the memory locations immediately adjacent to the array, in
 * order to handle edge extension.
 */
void
schro_split_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  orc_mas4_across_sub_s16_1991_ip (lo, hi - 1, hi, hi + 1, hi + 2, 1<<3, 4, n);

  lo[-1] = lo[0];

  orc_add2_rshift_add_s16_22 (hi, lo - 1, lo, n);
}

void
schro_split_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_add2_rshift_sub_s16_11 (lo, hi, hi+1, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_add2_rshift_add_s16_22 (hi, lo - 1, lo, n);
}

void
schro_split_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  orc_mas4_across_sub_s16_1991_ip (lo, hi - 1, hi, hi + 1, hi + 2, 1<<3, 4, n);

  lo[-1] = lo[0];
  lo[-2] = lo[0];
  lo[n] = lo[n-1];

  orc_mas4_across_add_s16_1991_ip (hi, lo - 2, lo - 1, lo, lo + 1, 1<<4, 5, n);
}

void
schro_split_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  orc_haar_split_s16 (hi, lo, n);
}

static void
mas8_add_s16 (int16_t *dest, const int16_t *src, const int16_t *weights, int offset,
    int shift, int n)
{
  int i;
  for(i=0;i<n;i++){
    int x = offset;
    x += src[i+0] * weights[0];
    x += src[i+1] * weights[1];
    x += src[i+2] * weights[2];
    x += src[i+3] * weights[3];
    x += src[i+4] * weights[4];
    x += src[i+5] * weights[5];
    x += src[i+6] * weights[6];
    x += src[i+7] * weights[7];
    dest[i] += x >> shift;
  }
}

void
schro_split_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static const int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };

  lo[-4] = lo[0];
  lo[-3] = lo[0];
  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];
  lo[n+2] = lo[n-1];

  mas8_add_s16 (hi, lo - 4, stage1_weights, 128, 8, n);

  hi[-3] = hi[0];
  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  hi[n+2] = hi[n-1];
  hi[n+3] = hi[n-1];

  mas8_add_s16 (lo, hi - 3, stage2_weights, 127, 8, n);
}

void
schro_split_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_mas2_sub_s16_ip (lo, hi, hi + 1, 6497, 2048, 12, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_mas2_sub_s16_ip (hi, lo - 1, lo, 217, 2048, 12, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_mas2_add_s16_ip (lo, hi, hi + 1, 3616, 2048, 12, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_mas2_add_s16_ip (hi, lo - 1, lo, 1817, 2048, 12, n);

}

void
schro_synth_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];

  orc_add2_rshift_sub_s16_22 (hi, lo - 1, lo, n);

  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];

  orc_mas4_across_add_s16_1991_ip (lo, hi - 1, hi, hi + 1, hi + 2, 1<<3, 4, n);
}

void
schro_synth_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_add2_rshift_sub_s16_22 (hi, lo - 1, lo, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_add2_rshift_add_s16_11 (lo, hi, hi+1, n);
}

void
schro_synth_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  lo[-1] = lo[0];
  lo[-2] = lo[0];
  lo[n] = lo[n-1];
  orc_mas4_across_sub_s16_1991_ip (hi, lo - 2, lo - 1, lo, lo + 1, 1<<4, 5, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  orc_mas4_across_add_s16_1991_ip (lo, hi-1, hi, hi + 1, hi + 2, 1<<3, 4, n);
}

void
schro_synth_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  orc_haar_synth_s16 (hi, lo, n);
}

void
schro_synth_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static const int16_t stage2_weights[] = { 8, -21, 46, -161, -161, 46, -21, 8 };

  hi[-3] = hi[0];
  hi[-2] = hi[0];
  hi[-1] = hi[0];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-1];
  hi[n+2] = hi[n-1];
  hi[n+3] = hi[n-1];

  mas8_add_s16 (lo, hi - 3, stage1_weights, 128, 8, n);

  lo[-4] = lo[0];
  lo[-3] = lo[0];
  lo[-2] = lo[0];
  lo[-1] = lo[0];
  lo[n] = lo[n-1];
  lo[n+1] = lo[n-1];
  lo[n+2] = lo[n-1];

  mas8_add_s16 (hi, lo - 4, stage2_weights, 127, 8, n);
}

void
schro_synth_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_mas2_sub_s16_ip (hi, lo - 1, lo, 1817, 2048, 12, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_mas2_sub_s16_ip (lo, hi, hi + 1, 3616, 2048, 12, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-1];

  orc_mas2_add_s16_ip (hi, lo - 1, lo, 217, 2048, 12, n);

  hi[-1] = hi[0];
  hi[n] = hi[n-1];

  orc_mas2_add_s16_ip (lo, hi, hi + 1, 6497, 2048, 12, n);
}


void schro_iwt_desl_9_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))

  /* FIXME */
  SCHRO_ASSERT(height>=6);

  for(i=0;i<height + 6;i++){
    int i1 = i-4;
    int i2 = i-6;
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_deinterleave2_lshift1_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_desl93 (hi, lo, width/2);
      orc_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      orc_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      if (i1 < 2 || i1 >= height-4) {
        orc_mas4_across_sub_s16_1991_ip (ROW(i1+1),
            ROW(CLAMP(i1-2,0,height-2)),
            ROW(CLAMP(i1+0,0,height-2)),
            ROW(CLAMP(i1+2,0,height-2)),
            ROW(CLAMP(i1+4,0,height-2)),
            1<<3, 4, width);
      } else {
        orc_mas4_across_sub_s16_1991_ip (ROW(i1+1),
            ROW(i1-2),
            ROW(i1+0),
            ROW(i1+2),
            ROW(i1+4),
            1<<3, 4, width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      if (i2 == 0) {
        orc_add2_rshift_add_s16_22 (ROW(i2),
            ROW(CLAMP(i2-1,1,height-1)),
            ROW(CLAMP(i2+1,1,height-1)),
            width);
      } else {
        orc_add2_rshift_add_s16_22 (ROW(i2),
            ROW(i2-1), ROW(i2+1),
            width);
      }
    }
  }
}

#if 0
void schro_iwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 2;i++){
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_deinterleave2_lshift1_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_53 (hi, lo, width/2);
      orc_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      orc_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }

    if ((i&1) == 0 && i >= 2) {
      int16_t *d;
      if (i<height) {
        d = OFFSET(data,i*stride);
      } else {
        d = OFFSET(data,(height-2)*stride);
      }
      orc_add2_rshift_sub_s16_11 (
          OFFSET(data, (i-1)*stride),
          OFFSET(data, (i-2)*stride),
          d, width);

      if (i-3>=0) {
        d = OFFSET(data, (i-3)*stride);
      } else {
        d = OFFSET(data, 1*stride);
      }
      orc_add2_rshift_add_s16_22 (
          OFFSET(data, (i-2)*stride),
          d,
          OFFSET(data, (i-1)*stride),
          width);
    }
  }
#undef ROW
}
#endif

void schro_iwt_13_5 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

  /* FIXME */
  SCHRO_ASSERT(height>=6);

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 8;i++){
    int i1 = i-4;
    int i2 = i-6;
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_deinterleave2_lshift1_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_135 (hi, lo, width/2);
      orc_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      orc_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      if (i1 < 2 || i1 >= height-4) {
        orc_mas4_across_sub_s16_1991_ip (ROW(i1+1),
            ROW(CLAMP(i1-2,0,height-2)),
            ROW(CLAMP(i1+0,0,height-2)),
            ROW(CLAMP(i1+2,0,height-2)),
            ROW(CLAMP(i1+4,0,height-2)),
            1<<3, 4, width);
      } else {
        orc_mas4_across_sub_s16_1991_ip (ROW(i1+1),
            ROW(i1-2), ROW(i1+0), ROW(i1+2), ROW(i1+4),
            1<<3, 4, width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      if (i2 < 4 || i2 >= height-2) {
        orc_mas4_across_add_s16_1991_ip (ROW(i2),
            ROW(CLAMP(i2-3,1,height-1)),
            ROW(CLAMP(i2-1,1,height-1)),
            ROW(CLAMP(i2+1,1,height-1)),
            ROW(CLAMP(i2+3,1,height-1)),
            1<<4, 5, width);
      } else {
        orc_mas4_across_add_s16_1991_ip (
            ROW(i2),
            ROW(i2-3), ROW(i2-1), ROW(i2+1), ROW(i2+3),
            1<<4, 5, width);
      }
    }
#undef ROW
  }
}

#if 0
static void
schro_iwt_haar (int16_t *data, int stride, int width, int height,
    int16_t *tmp, int16_t shift)
{
  int16_t *data1;
  int16_t *data2;
  int i;

  for(i=0;i<height;i+=2){
    data1 = OFFSET(data,i*stride);
    if (shift == 1) {
      orc_lshift1_s16(tmp, data1, width);
    } else {
      orc_memcpy (tmp, data1, width*sizeof(int16_t));
    }
    orc_haar_deint_split_s16 (data1, data1 + width/2, tmp, width/2);

    data2 = OFFSET(data,(i+1)*stride);
    if (shift == 1) {
      orc_lshift1_s16(tmp, data2, width);
    } else {
      orc_memcpy (tmp, data2, width*sizeof(int16_t));
    }
    orc_haar_deint_split_s16 (data2, data2 + width/2, tmp, width/2);

    orc_haar_split_s16 (data1, data2, width);
  }
}

void schro_iwt_haar0 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 0);
}

void schro_iwt_haar1 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 1);
}
#endif

static void
mas8_across_add_s16 (int16_t *dest, const int16_t *src, int stride,
    const int16_t *weights, int offset, int shift, int n)
{
  int i;
  for(i=0;i<n;i++){
    int x = offset;
    x += ((int16_t *)SCHRO_OFFSET(src,stride*0))[i] * weights[0];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*1))[i] * weights[1];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*2))[i] * weights[2];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*3))[i] * weights[3];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*4))[i] * weights[4];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*5))[i] * weights[5];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*6))[i] * weights[6];
    x += ((int16_t *)SCHRO_OFFSET(src,stride*7))[i] * weights[7];
    dest[i] += x >> shift;
  }
}

void schro_iwt_fidelity (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

  /* FIXME */
  SCHRO_ASSERT(height>=16);

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 16;i++){
    int i1 = i-8;
    int i2 = i-16;
    if (i < height) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + 12 + width/2;
      orc_deinterleave2_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_fidelity (hi, lo, width/2);
      orc_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      orc_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_weights[][8] = {
        { 161 + 161 - 46 + 21 - 8, -46, 21, -8, 0, 0, 0, 0 },
        { 161 - 46 + 21 - 8, 161, -46, 21, -8, 0, 0, 0 },
        { -46 + 21 - 8, 161, 161, -46, 21, -8, 0, 0 },
        { 21 - 8, -46, 161, 161, -46, 21, -8, 0 },
        { -8, 21, -46, 161, 161, -46, 21, -8 },
        { 0, -8, 21, -46, 161, 161, -46, 21 -8 },
        { 0, 0, -8, 21, -46, 161, 161, -46 + 21 - 8 },
        { 0, 0, 0, -8, 21, -46, 161, 161 - 46 + 21 - 8 },
      };
      const int16_t *weights;
      int offset;
      if (i1 < 8) {
        weights = stage1_weights[i1/2];
        offset = 1;
      } else if (i1 >= height - 6) {
        weights = stage1_weights[8 - (height - i1)/2];
        offset = height + 1 - 16;
      } else {
        weights = stage1_weights[4];
        offset = i1 - 7;
      }
      mas8_across_add_s16 (ROW(i1), ROW(offset), stride * 2,
          weights, 128, 8, width);
    }
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_weights[][8] = {
        { 2 - 10 + 25 - 81, -81, 25, -10, 2, 0, 0, 0 },
        { 2 - 10 + 25, -81, -81, 25, -10, 2, 0, 0 },
        { 2 -10, 25, -81, -81, 25, -10, 2, 0 },
        { 2, -10, 25, -81, -81, 25, -10, 2 },
        { 0, 2, -10, 25, -81, -81, 25, -10 + 2 },
        { 0, 0, 2, -10, 25, -81, -81, 25 - 10 + 2 },
        { 0, 0, 0, 2, -10, 25, -81, -81 + 25 - 10 + 2 },
        { 0, 0, 0, 0, 2, -10, 25, -81 - 81 + 25 - 10 + 2 }
      };
      const int16_t *weights;
      int offset;
      if (i2 < 6) {
        weights = stage2_weights[i2/2];
        offset = 0;
      } else if (i2 >= height - 8) {
        weights = stage2_weights[8 - (height - i2)/2];
        offset = height - 16;
      } else {
        weights = stage2_weights[3];
        offset = i2 - 6;
      }
      mas8_across_add_s16 (ROW(i2+1), ROW(offset), stride * 2,
          weights, 127, 8, width);
    }
  }
#undef ROW
}

void schro_iwt_daub_9_7 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;
  int i1;
  int i2;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=0;i<height + 4;i++){
    i1 = i - 2;
    i2 = i - 4;
    if (i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_deinterleave2_lshift1_s16 (hi, lo, ROW(i), width/2);
      schro_split_ext_daub97 (hi, lo, width/2);
      orc_memcpy (ROW(i), hi, width/2*sizeof(int16_t));
      orc_memcpy (ROW(i) + width/2, lo, width/2*sizeof(int16_t));
    }

    if ((i1&1) == 0 && i1 >=0 && i1 < height) {
      orc_mas2_sub_s16_ip (ROW(i1+1), ROW(i1), ROW(CLAMP(i1+2,0,height-2)),
          6497, 2048, 12, width);
      orc_mas2_sub_s16_ip (ROW(i1), ROW(CLAMP(i1-1,1,height-1)), ROW(i1+1),
          217, 2048, 12, width);
    }
    if ((i2&1) == 0 && i2 >=0 && i2 < height) {
      orc_mas2_add_s16_ip (ROW(i2+1), ROW(i2), ROW(CLAMP(i2+2,0,height-2)),
          3616, 2048, 12, width);
      orc_mas2_add_s16_ip (ROW(i2), ROW(CLAMP(i2-1,1,height-1)), ROW(i2+1),
          1817, 2048, 12, width);
    }
  }
#undef ROW
}

void
schro_wavelet_transform_2d (SchroFrameData *fd, int filter, int16_t *tmp)
{
  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(fd->format) ==
      SCHRO_FRAME_FORMAT_DEPTH_S16);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_iwt_desl_9_3 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iwt_5_3 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_iwt_13_5 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iwt_haar0 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iwt_haar1 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iwt_fidelity (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_iwt_daub_9_7(fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
  }
}




void schro_iiwt_desl_9_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))

  /* FIXME */
  SCHRO_ASSERT(height>=6);

  for(i=-6;i<height;i++){
    int i1 = i+2;
    int i2 = i+6;
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      if (i2 == 0) {
        orc_add2_rshift_sub_s16_22 (
            OFFSET(data,i2*stride),
            OFFSET(data, (i2+1)*stride), OFFSET(data, (i2+1)*stride),
            width);
      } else {
        orc_add2_rshift_sub_s16_22 (
            OFFSET(data,i2*stride),
            OFFSET(data, (i2-1)*stride), OFFSET(data, (i2+1)*stride),
            width);
      }
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      if (i1 < 2 || i1 >= height-4) {
        orc_mas4_across_add_s16_1991_ip (ROW(i1+1),
            ROW(CLAMP(i1-2,0,height-2)),
            ROW(CLAMP(i1+0,0,height-2)),
            ROW(CLAMP(i1+2,0,height-2)),
            ROW(CLAMP(i1+4,0,height-2)),
            1<<3, 4, width);
      } else {
        orc_mas4_across_add_s16_1991_ip (ROW(i1+1),
            ROW(i1-2),
            ROW(i1+0),
            ROW(i1+2),
            ROW(i1+4),
            1<<3, 4, width);
      }
    }
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      orc_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_desl93 (hi, lo, width/2);
      orc_interleave2_rrshift1_s16 (ROW(i), hi, lo, width/2);
    }
  }
}

#if 0
void schro_iiwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-4;i<height + 2;i++){
    int i1 = i + 2;
    int i2 = i + 4;

    if ((i2&1) == 0 && i2 >= 0 && i2 < height) {
      int16_t *d;
      if (i2-1>=0) {
        d = OFFSET(data, (i2-1)*stride);
      } else {
        d = OFFSET(data, 1*stride);
      }
      orc_add2_rshift_sub_s16_22 (
          OFFSET(data, i2*stride),
          d,
          OFFSET(data, (i2+1)*stride),
          width);
    }
    if ((i1&1) == 0 && i1 >= 0 && i1 < height) {
      int16_t *d;
      if (i1+2<height) {
        d = OFFSET(data,(i1+2)*stride);
      } else {
        d = OFFSET(data,(height-2)*stride);
      }
      orc_add2_rshift_add_s16_11 (
          OFFSET(data, (i1+1)*stride),
          OFFSET(data, i1*stride),
          d,
          width);
    }
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      orc_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_53 (hi, lo, width/2);
      orc_interleave2_rrshift1_s16 (ROW(i), hi, lo, width/2);
    }
  }
#undef ROW
}
#endif

void schro_iiwt_13_5 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

  /* FIXME */
  SCHRO_ASSERT(height>=6);

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-8;i<height;i++){
    int i1 = i+4;
    int i2 = i+8;
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      if (i2 < 4 || i2 >= height-2) {
        orc_mas4_across_sub_s16_1991_ip (ROW(i2),
            ROW(CLAMP(i2-3,1,height-1)),
            ROW(CLAMP(i2-1,1,height-1)),
            ROW(CLAMP(i2+1,1,height-1)),
            ROW(CLAMP(i2+3,1,height-1)),
            1<<4, 5, width);
      } else {
        orc_mas4_across_sub_s16_1991_ip (
            ROW(i2),
            ROW(i2-3), ROW(i2-1), ROW(i2+1), ROW(i2+3),
            1<<4, 5, width);
      }
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      if (i1 < 2 || i1 >= height-4) {
        orc_mas4_across_add_s16_1991_ip (ROW(i1+1),
            ROW(CLAMP(i1-2,0,height-2)),
            ROW(CLAMP(i1+0,0,height-2)),
            ROW(CLAMP(i1+2,0,height-2)),
            ROW(CLAMP(i1+4,0,height-2)),
            1<<3, 4, width);
      } else {
        orc_mas4_across_add_s16_1991_ip (ROW(i1+1),
            ROW(i1-2), ROW(i1+0), ROW(i1+2), ROW(i1+4),
            1<<3, 4, width);
      }
    }
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      orc_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_135 (hi, lo, width/2);
      orc_interleave2_rrshift1_s16 (ROW(i), hi, lo, width/2);
    }
#undef ROW
  }
}

static void
schro_iiwt_haar (int16_t *data, int stride, int width, int height,
    int16_t *tmp, int16_t shift)
{
  int16_t *data1;
  int16_t *data2;
  int i;

  for(i=0;i<height;i+=2){
    data1 = OFFSET(data,i*stride);
    data2 = OFFSET(data,(i+1)*stride);

    orc_haar_synth_s16 (data1, data2, width);

    orc_haar_synth_s16 (data1, data1 + width/2, width/2);
    if (shift) {
      orc_add_const_rshift_s16_11 (tmp, data1, width);
    } else {
      orc_memcpy (tmp, data1, width*sizeof(int16_t));
    }
    orc_interleave2_s16 (data1, tmp, tmp + width/2, width/2);

    orc_haar_synth_s16 (data2, data2 + width/2, width/2);
    if (shift) {
      orc_add_const_rshift_s16_11 (tmp, data2, width);
    } else {
      orc_memcpy (tmp, data2, width*sizeof(int16_t));
    }
    orc_interleave2_s16 (data2, tmp, tmp + width/2, width/2);
  }
}

void schro_iiwt_haar0 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iiwt_haar (data, stride, width, height, tmp, 0);
}

void schro_iiwt_haar1 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iiwt_haar (data, stride, width, height, tmp, 1);
}

void schro_iiwt_fidelity (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;

  /* FIXME */
  SCHRO_ASSERT(height>=16);

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-16;i<height;i++){
    int i1 = i+8;
    int i2 = i+16;
    if ((i2&1) == 0 && i2>=0 && i2 < height) {
      static const int16_t stage2_weights[][8] = {
        { -2 + 10 - 25 + 81, 81, -25, 10, -2, 0, 0, 0 },
        { -2 + 10 - 25, 81, 81, -25, 10, -2, 0, 0 },
        { -2 + 10, -25, 81, 81, -25, 10, -2, 0 },
        { -2, 10, -25, 81, 81, -25, 10, -2 },
        { 0, -2, 10, -25, 81, 81, -25, 10 - 2 },
        { 0, 0, -2, 10, -25, 81, 81, -25 + 10 - 2 },
        { 0, 0, 0, -2, 10, -25, 81, 81 - 25 + 10 - 2 },
        { 0, 0, 0, 0, -2, 10, -25, 81 + 81 - 25 + 10 - 2 }
      };
      const int16_t *weights;
      int offset;
      if (i2 < 6) {
        weights = stage2_weights[i2/2];
        offset = 0;
      } else if (i2 >= height - 8) {
        weights = stage2_weights[8 - (height - i2)/2];
        offset = height - 16;
      } else {
        weights = stage2_weights[3];
        offset = i2 - 6;
      }
      mas8_across_add_s16 (ROW(i2+1), ROW(offset), stride * 2,
          weights, 128, 8, width);
    }
    if ((i1&1) == 0 && i1>=0 && i1 < height) {
      static const int16_t stage1_weights[][8] = {
        { 8 - 21 + 46 - 161 - 161, 46, -21, 8, 0, 0, 0, 0 },
        { 8 - 21 + 46 - 161, -161, 46, -21, 8, 0, 0, 0 },
        { 8 - 21 + 46, -161, -161, 46, -21, 8, 0, 0 },
        { 8 -21, 46, -161, -161, 46, -21, 8, 0 },
        { 8, -21, 46, -161, -161, 46, -21, 8 },
        { 0, 8, -21, 46, -161, -161, 46, -21 + 8 },
        { 0, 0, 8, -21, 46, -161, -161, 46 - 21 + 8 },
        { 0, 0, 0, 8, -21, 46, -161, -161 + 46 - 21 + 8 },
      };
      const int16_t *weights;
      int offset;
      if (i1 < 8) {
        weights = stage1_weights[i1/2];
        offset = 1;
      } else if (i1 >= height - 6) {
        weights = stage1_weights[8 - (height - i1)/2];
        offset = height + 1 - 16;
      } else {
        weights = stage1_weights[4];
        offset = i1 - 7;
      }
      mas8_across_add_s16 (ROW(i1), ROW(offset), stride * 2,
          weights, 127, 8, width);
    }
    if (i >=0 && i < height) {
      int16_t *hi = tmp + 4;
      int16_t *lo = tmp + 12 + width/2;
      orc_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      orc_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_fidelity (hi, lo, width/2);
      orc_interleave2_s16 (ROW(i), hi, lo, width/2);
    }
  }
#undef ROW
}

void schro_iiwt_daub_9_7 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  int i;
  int i1;
  int i2;
  int i3;
  int i4;

#define ROW(row) ((int16_t *)OFFSET(data, (row)*stride))
  for(i=-6;i<height;i++){
    i1 = i + 0;
    i2 = i + 2;
    i3 = i + 2;
    i4 = i + 6;

    if ((i4&1) == 0 && i4 >=0 && i4 < height) {
      orc_mas2_sub_s16_ip (ROW(i4), ROW(CLAMP(i4-1,1,height-1)), ROW(i4+1),
          1817, 2048, 12, width);
    }

    if ((i3&1) == 0 && i3 >=0 && i3 < height) {
      orc_mas2_sub_s16_ip (ROW(i3+1), ROW(i3), ROW(CLAMP(i3+2,0,height-2)),
          3616, 2048, 12, width);
    }

    if ((i2&1) == 0 && i2 >=0 && i2 < height) {
      orc_mas2_add_s16_ip (ROW(i2), ROW(CLAMP(i2-1,1,height-1)), ROW(i2+1),
          217, 2048, 12, width);
    }

    if ((i1&1) == 0 && i1 >=0 && i1 < height) {
      orc_mas2_add_s16_ip (ROW(i1+1), ROW(i1), ROW(CLAMP(i1+2,0,height-2)),
          6497, 2048, 12, width);
    }

    if (i >=0 && i < height) {
      int16_t *hi = tmp + 2;
      int16_t *lo = tmp + 6 + width/2;
      orc_memcpy (hi, ROW(i), width/2*sizeof(int16_t));
      orc_memcpy (lo, ROW(i) + width/2, width/2*sizeof(int16_t));
      schro_synth_ext_daub97 (hi, lo, width/2);
      orc_interleave2_s16 (ROW(i), hi, lo, width/2);
      orc_add_const_rshift_s16_11 (ROW(i), ROW(i), width);
    }
  }
#undef ROW
}

void
schro_wavelet_inverse_transform_2d (SchroFrameData *fd, int filter,
    int16_t *tmp)
{
  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(fd->format) ==
      SCHRO_FRAME_FORMAT_DEPTH_S16);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_iiwt_desl_9_3 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iiwt_5_3 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_iiwt_13_5 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iiwt_haar0 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iiwt_haar1 (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iiwt_fidelity (fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_iiwt_daub_9_7(fd->data, fd->stride, fd->width, fd->height, tmp);
      break;
  }
}

static void
wavelet_iiwt_5_3_vert (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame, component, i-1);
    if (i+1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame, component, i+1);
    } else {
      lo2 = lo1;
    }

    orc_add2_rshift_add_s16_11_op (dest, hi, lo1, lo2, width);
  } else {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame->virt_frame1, component, i-1);
    }
    hi2 = schro_virt_frame_get_line (frame->virt_frame1, component, i+1);

    orc_add2_rshift_sub_s16_22_op (dest, lo, hi1, hi2, width);
  }
}

static void
wavelet_iiwt_5_3_horiz (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *tmp = frame->virt_priv2;
  int16_t *hi = tmp + 2;
  int16_t *lo = tmp + 6 + width/2;
  int16_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  orc_memcpy (hi, src, width/2*sizeof(int16_t));
  orc_memcpy (lo, src + width/2, width/2*sizeof(int16_t));
  schro_synth_ext_53 (hi, lo, width/2);
  orc_interleave2_rrshift1_s16 (dest, hi, lo, width/2);
}

static void
wavelet_iwt_5_3_horiz (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *tmp = frame->virt_priv2;
  int16_t *src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  int16_t *hi = tmp + 2;
  int16_t *lo = tmp + 6 + width/2;

  orc_deinterleave2_lshift1_s16 (hi, lo, src, width/2);
  schro_split_ext_53 (hi, lo, width/2);
  orc_memcpy (dest, hi, width/2*sizeof(int16_t));
  orc_memcpy (dest + width/2, lo, width/2*sizeof(int16_t));
}

static void
wavelet_iwt_5_3_vert (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo1, *lo2;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo1 = schro_virt_frame_get_line (frame->virt_frame1, component, i-1);
    if (i+1 < frame->height) {
      lo2 = schro_virt_frame_get_line (frame->virt_frame1, component, i+1);
    } else {
      lo2 = lo1;
    }

    orc_add2_rshift_sub_s16_11_op (dest, hi, lo1, lo2, width);
  } else {
    int16_t *lo;
    int16_t *hi1, *hi2;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    if (i == 0) {
      hi1 = schro_virt_frame_get_line (frame, component, 1);
    } else {
      hi1 = schro_virt_frame_get_line (frame, component, i-1);
    }
    hi2 = schro_virt_frame_get_line (frame, component, i+1);

    orc_add2_rshift_add_s16_22_op (dest, lo, hi1, hi2, width);
  }
}

void schro_iiwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  SchroFrame *frame;
  SchroFrame *vf1;
  SchroFrame *vf2;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S16_444;
  frame->width = width;
  frame->height = height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = stride;
  frame->components[0].data = data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf1->virt_frame1 = frame;
  vf1->virt_priv2 = tmp;
  vf1->render_line = wavelet_iiwt_5_3_vert;

  vf2 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iiwt_5_3_horiz;

  schro_virt_frame_render (vf2, frame);

  schro_frame_unref (vf2);
}

void schro_iwt_5_3 (int16_t *data, int stride, int width, int height,
    int16_t *tmp)
{
  SchroFrame *frame;
  SchroFrame *vf1;
  SchroFrame *vf2;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S16_444;
  frame->width = width;
  frame->height = height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = stride;
  frame->components[0].data = data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf1->virt_frame1 = frame;
  vf1->virt_priv2 = tmp;
  vf1->render_line = wavelet_iwt_5_3_horiz;

  vf2 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iwt_5_3_vert;

  schro_virt_frame_render (vf2, frame);

  schro_frame_unref (vf2);
}

static void
wavelet_iwt_haar_horiz (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *src;
  int16_t *hi = dest + width/2;
  int16_t *lo = dest;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  orc_haar_deint_split_s16 (lo, hi, src, width/2);
}

static void
wavelet_iwt_haar_shift1_horiz (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;
  int16_t *src;
  int16_t *hi = dest + width/2;
  int16_t *lo = dest;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  orc_haar_deint_lshift1_split_s16 (lo, hi, src, width/2);
}

static void
wavelet_iwt_haar_vert (SchroFrame *frame, void *_dest, int component,
    int i)
{
  int16_t *dest = _dest;
  int width = frame->components[component].width;

  if (i & 1) {
    int16_t *hi;
    int16_t *lo;

    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i-1);

    orc_haar_split_s16_hi (dest, lo, hi, width);
  } else {
    int16_t *lo;
    int16_t *hi;

    lo = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    hi = schro_virt_frame_get_line (frame->virt_frame1, component, i+1);

    orc_haar_split_s16_lo (dest, lo, hi, width);
  }
}

static void
schro_iwt_haar (int16_t *data, int stride, int width, int height,
    int16_t *tmp, int16_t shift)
{
  SchroFrame *frame;
  SchroFrame *vf1;
  SchroFrame *vf2;

  frame = schro_frame_new ();

  frame->format = SCHRO_FRAME_FORMAT_S16_444;
  frame->width = width;
  frame->height = height;

  frame->components[0].format = SCHRO_FRAME_FORMAT_S16_444;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = stride;
  frame->components[0].data = data;

  vf1 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf1->virt_frame1 = frame;
  vf1->virt_priv2 = tmp;
  if (shift) {
    vf1->render_line = wavelet_iwt_haar_shift1_horiz;
  } else {
    vf1->render_line = wavelet_iwt_haar_horiz;
  }

  vf2 = schro_frame_new_virtual (NULL, frame->format, width, height);
  vf2->virt_frame1 = vf1;
  vf2->virt_priv2 = tmp;
  vf2->render_line = wavelet_iwt_haar_vert;

  schro_virt_frame_render (vf2, frame);

  schro_frame_unref (vf2);
}

void schro_iwt_haar0 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 0);
}

void schro_iwt_haar1 (int16_t *data, int stride, int width, int height, int16_t *tmp)
{
  schro_iwt_haar (data, stride, width, height, tmp, 1);
}

