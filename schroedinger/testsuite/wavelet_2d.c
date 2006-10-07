
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <liboil/liboil.h>

int filtershift[] = { 1, 1, 1, 0, 1, 2, 0, 1 };

typedef struct _Picture Picture;
struct _Picture {
  int16_t *data;
  int stride;
  int width;
  int height;
};

#define OFFSET(ptr,x) (void *)(((uint8_t *)ptr) + (x))

void dump(Picture *p);
void iwt_ref(Picture *p, int filter);
void iiwt_ref(Picture *p, int filter);

void schro_split_ext (int16_t *hi, int16_t *lo, int n, int filter);
void schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter);

void
gen_const (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = 100;
    }
  }
}

void
gen_vramp (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (100*j+p->height/2)/p->height;
    }
  }
}

void
gen_hramp (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (100*i+p->width/2)/p->width;
    }
  }
}

void
gen_valt (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (j&1)*100;
    }
  }
}

void
gen_halt (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (i&1)*100;
    }
  }
}

void
gen_checkerboard (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = ((i+j)&1)*100;
    }
  }
}


typedef struct _Generator Generator;
struct _Generator {
  char *name;
  void (*create)(Picture *p);
};

Generator generators[] = {
  { "constant", gen_const },
  { "hramp", gen_hramp },
  { "vramp", gen_vramp },
  { "halt", gen_halt },
  { "valt", gen_valt },
  { "checkerboard", gen_checkerboard }
};


int16_t tmp[20*20];

void
local_test (int filter)
{
  int $;
  Picture pict;
  Picture *p = &pict;

  p->data = tmp;
  p->stride = 40;
  p->width = 20;
  p->height = 20;

  for($=0;$<sizeof(generators)/sizeof(generators[0]);$++){
    printf("  test \"%s\":\n", generators[$].name);
    generators[$].create(p);
    dump(p);
    iwt_ref(p,filter);
    dump(p);
    iiwt_ref(p,filter);
    dump(p);
  }
}

int
main (int argc, char *argv[])
{
  int filter;

  schro_init();
    
  for(filter=0;filter<=SCHRO_WAVELET_DAUB_9_7;filter++){
    printf("Filter %d:\n", filter);
    local_test(filter);
    //random_test(filter);
  }

  return 0;
}

void dump(Picture *p)
{
  int i;
  int j;
  int16_t *data;

  printf("-----\n");
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      printf("%3d ", data[i]);
    }
    printf("\n");
  }
  printf("-----\n");
}

void
copy (int16_t *d, int ds, int16_t *s, int ss, int n)
{
  int i;
  int16_t *xd, *xs;
  for(i=0;i<n;i++){
    xd = OFFSET(d,ds * i);
    xs = OFFSET(s,ss * i);
    *xd = *xs;
  }
}

void
rshift (Picture *p, int n)
{
  int i;
  int j;
  int16_t *data;

  if (n==0) return;
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] >>= n;
    }
  }
}

void
lshift (Picture *p, int n)
{
  int i;
  int j;
  int16_t *data;

  if (n==0) return;
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] <<= n;
    }
  }
}

void iwt_ref(Picture *p, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t tmp3[100];
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  lshift(p, filtershift[filter]);

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    oil_deinterleave2_s16 (hi, lo, data, p->width/2);
    schro_split_ext (hi, lo, p->width/2, filter);
    copy(data, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
    copy(data + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
  }

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(tmp3, sizeof(int16_t), data, p->stride, p->height);
    oil_deinterleave2_s16 (hi, lo, tmp3, p->height/2);
    schro_split_ext (hi, lo, p->height/2, filter);
    oil_interleave2_s16 (tmp3, hi, lo, p->height/2);
    copy(data, p->stride, tmp3, sizeof(int16_t), p->height);
  }

}

void iiwt_ref(Picture *p, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t tmp3[100];
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(tmp3, sizeof(int16_t), data, p->stride, p->height);
    oil_deinterleave2_s16 (hi, lo, tmp3, p->height/2);
    schro_synth_ext (hi, lo, p->height/2, filter);
    oil_interleave2_s16 (tmp3, hi, lo, p->height/2);
    copy(data, p->stride, tmp3, sizeof(int16_t), p->height);
  }

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    copy(hi, sizeof(int16_t), data, sizeof(int16_t), p->width/2);
    copy(lo, sizeof(int16_t), data + p->width/2, sizeof(int16_t), p->width/2);
    schro_synth_ext (hi, lo, p->width/2, filter);
    oil_interleave2_s16 (data, hi, lo, p->width/2);
  }

  rshift(p, filtershift[filter]);
}

void
schro_split_ext (int16_t *hi, int16_t *lo, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_split_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_split_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_split_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_split_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_split_ext_daub97(hi, lo, n);
      break;
  }
}

void
schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_synth_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_synth_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_synth_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_synth_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_synth_ext_daub97(hi, lo, n);
      break;
  }
}

void notoil_lshift_s16 (int16_t *dest, int16_t *src, int32_t *shift, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i]<<(*shift);
  }
}

void notoil_rshift_s16 (int16_t *dest, int16_t *src, int32_t *shift, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i]>>(*shift);
  }
}

void
notoil_cross_mas2_add_s16 (int16_t *d, int16_t *s1,
    int16_t *s2, int16_t *s3, const int16_t *s4, const int16_t *s5, int n)
{
  int i;
  int x;
  for(i=0;i<n;i++){
    x = s5[0];
    x += s2[i]*s4[0] + s3[i]*s4[1];
    x >>= s5[1];
    d[i] = s1[i] + x;
  }
}

#define OIL_GET(ptr, offset, type) (*(type *)((uint8_t *)(ptr) + (offset)) )
void
notoil_cross_mas4_add_s16 (int16_t *d, int16_t *s1,
    int16_t *s2, int sstr2, const int16_t *s3, const int16_t *s4, int n)
{
  int i;
  int j;
  int x;
  for(i=0;i<n;i++){
    x = s4[0];
    for(j=0;j<4;j++){
      x += OIL_GET(s2, i*sizeof(int16_t) + j*sstr2, int16_t)*s3[j];
    }
    x >>= s4[1];
    d[i] = s1[i] + x;
  }
}

void
notoil_cross_mas8_add_s16 (int16_t *d, int16_t *s1,
    int16_t *s2, int sstr2, const int16_t *s3, const int16_t *s4, int n)
{
  int i;
  int j;
  int x;
  for(i=0;i<n;i++){
    x = s4[0];
    for(j=0;j<8;j++){
      x += OIL_GET(s2, i*sizeof(int16_t) + j*sstr2, int16_t)*s3[j];
    }
    x >>= s4[1];
    d[i] = s1[i] + x;
  }
}

void x_schro_iwt_desl_9_3 (Picture *p)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;
  int one = 1;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

#define ROW(p,row) ((int16_t *)OFFSET((p)->data, (row)*(p)->stride))
  for(i=0;i<p->height + 6;i++){
    int i1 = i-4;
    int i2 = i-6;
    if (i < p->height) {
      notoil_lshift_s16(ROW(p,i), ROW(p,i), &one, p->width);
      oil_deinterleave2_s16 (hi, lo, ROW(p,i), p->width/2);
      schro_split_ext_desl93 (hi, lo, p->width/2);
      copy(ROW(p,i), sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
      copy(ROW(p,i) + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    }
    if ((i1&1) == 0 && i1>=0 && i1 < p->height) {
      static const int16_t stage1_offset_shift[] = { 7, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { -9, -8, 1, 0 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      } else if (i1 == p->height-4) {
        static const int16_t stage1_weights[] = { 0, 1, -9, -8 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-4), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      } else if (i1 == p->height-2) {
        static const int16_t stage1_weights[] = { 2, -18 };
        notoil_cross_mas2_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-2), ROW(p, i1),
            stage1_weights, stage1_offset_shift, p->width);
      } else {
        static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-2), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < p->height) {
      static const int16_t stage2_offset_shift[] = { 2, 2 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { 1, 1 };
        notoil_cross_mas2_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2+1), ROW(p, i2+1),
            stage2_weights, stage2_offset_shift, p->width);
      } else {
        static const int16_t stage2_weights[] = { 1, 1 };
        notoil_cross_mas2_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2-1), ROW(p, i2+1),
            stage2_weights, stage2_offset_shift, p->width);
      }
    }
  }
}

void x_schro_iwt_5_3 (Picture *p)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 0, 1 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t *data;
  int i;
  int32_t one = 1;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  for(i=0;i<p->height + 2;i++){
    if (i < p->height) {
      data = OFFSET(p->data,i*p->stride);

      notoil_lshift_s16(data, data, &one, p->width);
      oil_deinterleave2_s16 (hi, lo, data, p->width/2);
      schro_split_ext_53 (hi, lo, p->width/2);
      copy(data, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
      copy(data + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    }

    if ((i&1) == 0 && i >= 2) {
      if (i<p->height) {
        data = OFFSET(p->data,i*p->stride);
      } else {
        data = OFFSET(p->data,(p->height-2)*p->stride);
      }
      notoil_cross_mas2_add_s16 (
          OFFSET(p->data, (i-1)*p->stride),
          OFFSET(p->data, (i-1)*p->stride),
          OFFSET(p->data, (i-2)*p->stride),
          data,
          stage1_weights, stage1_offset_shift, p->width);

      if (i-3>=0) {
        data = OFFSET(p->data, (i-3)*p->stride);
      } else {
        data = OFFSET(p->data, 1*p->stride);
      }
      notoil_cross_mas2_add_s16 (
          OFFSET(p->data, (i-2)*p->stride),
          OFFSET(p->data, (i-2)*p->stride),
          data,
          OFFSET(p->data, (i-1)*p->stride),
          stage2_weights, stage2_offset_shift, p->width);
    }
  }
}

void x_schro_iwt_13_5 (Picture *p)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;
  int one = 1;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

#define ROW(p,row) ((int16_t *)OFFSET((p)->data, (row)*(p)->stride))
  for(i=0;i<p->height + 8;i++){
    int i1 = i-4;
    int i2 = i-6;
    if (i < p->height) {
      notoil_lshift_s16(ROW(p,i), ROW(p,i), &one, p->width);
      oil_deinterleave2_s16 (hi, lo, ROW(p,i), p->width/2);
      schro_split_ext_135 (hi, lo, p->width/2);
      copy(ROW(p,i), sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
      copy(ROW(p,i) + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    }
    if ((i1&1) == 0 && i1>=0 && i1 < p->height) {
      static const int16_t stage1_offset_shift[] = { 7, 4 };
      if (i1 == 0) {
        static const int16_t stage1_weights[] = { -9, -8, 1, 0 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      } else if (i1 == p->height-4) {
        static const int16_t stage1_weights[] = { 0, 1, -9, -8 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-4), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      } else if (i1 == p->height-2) {
        static const int16_t stage1_weights[] = { 2, -18 };
        notoil_cross_mas2_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-2), ROW(p, i1),
            stage1_weights, stage1_offset_shift, p->width);
      } else {
        static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i1+1), ROW(p, i1+1), ROW(p, i1-2), p->stride * 2,
            stage1_weights, stage1_offset_shift, p->width);
      }
    }
    if ((i2&1) == 0 && i2>=0 && i2 < p->height) {
      static const int16_t stage2_offset_shift[] = { 16, 5 };
      if (i2 == 0) {
        static const int16_t stage2_weights[] = { 18, -2 };
        notoil_cross_mas2_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2+1), ROW(p, i2+3),
            stage2_weights, stage2_offset_shift, p->width);
      } else if (i2 == 2) {
        static const int16_t stage2_weights[] = { 8, 9, -1, 0 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2-1), p->stride * 2,
            stage2_weights, stage2_offset_shift, p->width);
      } else if (i2 == p->height-2) {
        static const int16_t stage2_weights[] = { 0, -1, 8, 9 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2-5), p->stride * 2,
            stage2_weights, stage2_offset_shift, p->width);
      } else {
        static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
        notoil_cross_mas4_add_s16 (
            ROW(p,i2), ROW(p, i2), ROW(p, i2-3), p->stride * 2,
            stage2_weights, stage2_offset_shift, p->width);
      }
    }
  }
}

void x_schro_iwt_haar (Picture *p, int shift)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t *data1;
  int16_t *data2;
  int i;
  int j;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  for(i=0;i<p->height;i+=2){
    data1 = OFFSET(p->data,i*p->stride);
    if (shift) {
      notoil_lshift_s16(data1, data1, &shift, p->width);
    }
    oil_deinterleave2_s16 (hi, lo, data1, p->width/2);
    schro_split_ext_haar (hi, lo, p->width/2);
    copy(data1, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
    copy(data1 + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);

    data2 = OFFSET(p->data,(i+1)*p->stride);
    if (shift) {
      notoil_lshift_s16(data2, data2, &shift, p->width);
    }
    oil_deinterleave2_s16 (hi, lo, data2, p->width/2);
    schro_split_ext_haar (hi, lo, p->width/2);
    copy(data2, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
    copy(data2 + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);

    for(j=0;j<p->width;j++){
      data2[j] -= data1[j];
      data1[j] += (data2[j] + 1)>>1;
    }
  }
}

void x_schro_iwt_haar0 (Picture *p)
{
  x_schro_iwt_haar (p, 0);
}

void x_schro_iwt_haar1 (Picture *p)
{
  x_schro_iwt_haar (p, 1);
}

void x_schro_iwt_haar2 (Picture *p)
{
  x_schro_iwt_haar (p, 2);
}

void x_schro_iwt_fidelity (Picture *p)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

#define ROW(p,row) ((int16_t *)OFFSET((p)->data, (row)*(p)->stride))
  for(i=0;i<p->height + 16;i++){
    int i1 = i-8;
    int i2 = i-16;
    if (i < p->height) {
      oil_deinterleave2_s16 (hi, lo, ROW(p,i), p->width/2);
      schro_split_ext_fidelity (hi, lo, p->width/2);
      copy(ROW(p,i), sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
      copy(ROW(p,i) + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    }
    if ((i1&1) == 0 && i1>=0 && i1 < p->height) {
      static const int16_t stage1_offset_shift[] = { 128, 8 };
      static const int16_t stage1_weights[][8] = {
        { 161 + 161, -46 - 46, 21 + 21, -8 - 8, 0, 0, 0, 0 },
        { 161 - 46, 161 + 21, -46 - 8, 21, -8, 0, 0, 0 },
        { -46 + 21, 161 - 8, 161, -46, 21, -8, 0, 0 },
        { 21 - 8, -46, 161, 161, -46, 21, -8, 0 },
        { -8, 21, -46, 161, 161, -46, 21, -8 },
        { 0, -8, 21, -46, 161, 161, -46 - 8, 21 },
        { 0, 0, -8, 21, -46, 161 - 8, 161 + 21, -46 },
        { 0, 0, 0, -8, 21 - 8, -46 + 21, 161 - 46, 161 },
      };
      const int16_t *weights;
      int offset;
      if (i1 < 8) {
        weights = stage1_weights[i1/2];
        offset = 1;
      } else if (i1 >= p->height - 6) {
        weights = stage1_weights[8 - (p->height - i1)/2];
        offset = p->height + 1 - 16;
      } else {
        weights = stage1_weights[4];
        offset = i1 - 7;
      }
      notoil_cross_mas8_add_s16 (
          ROW(p,i1), ROW(p, i1), ROW(p, offset), p->stride * 2,
          weights, stage1_offset_shift, p->width);
    }
    if ((i2&1) == 0 && i2>=0 && i2 < p->height) {
      static const int16_t stage2_offset_shift[] = { 127, 8 };
      static const int16_t stage2_weights[][8] = {
        { -81, -81 + 25, 25 - 10, -10 + 2, 2, 0, 0, 0 },
        { 25, -81 - 10, -81 + 2, 25, -10, 2, 0, 0 },
        { -10, 25 + 2, -81, -81, 25, -10, 2, 0 },
        { 2, -10, 25, -81, -81, 25, -10, 2 },
        { 0, 2, -10, 25, -81, -81, 25, -10 + 2 },
        { 0, 0, 2, -10, 25, -81, -81 + 2, 25 - 10 },
        { 0, 0, 0, 2, -10, 25 + 2, -81 - 10, -81 + 25 },
        { 0, 0, 0, 0, 2 + 2, -10 - 10, 25 + 25, -81 - 81 }
      };
      const int16_t *weights;
      int offset;
      if (i2 < 6) {
        weights = stage2_weights[i2/2];
        offset = 0;
      } else if (i2 >= p->height - 8) {
        weights = stage2_weights[8 - (p->height - i2)/2];
        offset = p->height - 16;
      } else {
        weights = stage2_weights[3];
        offset = i2 - 6;
      }
      notoil_cross_mas8_add_s16 (
          ROW(p,i2+1), ROW(p, i2+1), ROW(p, offset), p->stride * 2,
          weights, stage2_offset_shift, p->width);
    }
  }
}

void x_schro_iwt_daub_9_7 (Picture *p)
{
  static const int16_t stage1_weights[] = { -6497, -6497 };
  static const int16_t stage2_weights[] = { -217, -217 };
  static const int16_t stage3_weights[] = { 3616, 3616 };
  static const int16_t stage4_weights[] = { 1817, 1817 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t *data;
  int i;
  int one = 1;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  for(i=0;i<p->height + 4;i++){
    if (i < p->height) {
      data = OFFSET(p->data,i*p->stride);
      notoil_lshift_s16(data, data, &one, p->width);
      oil_deinterleave2_s16 (hi, lo, data, p->width/2);
      schro_split_ext_daub97 (hi, lo, p->width/2);
      copy(data, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
      copy(data + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    }

    if ((i&1) == 0) {
      if (i >= 2 && i < p->height + 2) {
        if (i<p->height) {
          data = OFFSET(p->data,i*p->stride);
        } else {
          data = OFFSET(p->data,(p->height-2)*p->stride);
        }
        notoil_cross_mas2_add_s16 (
            OFFSET(p->data, (i-1)*p->stride),
            OFFSET(p->data, (i-1)*p->stride),
            OFFSET(p->data, (i-2)*p->stride),
            data,
            stage1_weights, stage12_offset_shift, p->width);

        if (i-3>=0) {
          data = OFFSET(p->data, (i-3)*p->stride);
        } else {
          data = OFFSET(p->data, 1*p->stride);
        }
        notoil_cross_mas2_add_s16 (
            OFFSET(p->data, (i-2)*p->stride),
            OFFSET(p->data, (i-2)*p->stride),
            data,
            OFFSET(p->data, (i-1)*p->stride),
            stage2_weights, stage12_offset_shift, p->width);
      }
      if (i >= 4 && i < p->height + 4) {
        if (i-2<p->height) {
          data = OFFSET(p->data,(i-2)*p->stride);
        } else {
          data = OFFSET(p->data,(p->height-2)*p->stride);
        }
        notoil_cross_mas2_add_s16 (
            OFFSET(p->data, (i-3)*p->stride),
            OFFSET(p->data, (i-3)*p->stride),
            OFFSET(p->data, (i-4)*p->stride),
            data,
            stage3_weights, stage34_offset_shift, p->width);

        if (i-5>=0) {
          data = OFFSET(p->data, (i-5)*p->stride);
        } else {
          data = OFFSET(p->data, 1*p->stride);
        }
        notoil_cross_mas2_add_s16 (
            OFFSET(p->data, (i-4)*p->stride),
            OFFSET(p->data, (i-4)*p->stride),
            data,
            OFFSET(p->data, (i-3)*p->stride),
            stage4_weights, stage34_offset_shift, p->width);
      }


    }
  }
}

void iwt_test(Picture *p, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      x_schro_iwt_desl_9_3 (p);
      break;
    case SCHRO_WAVELET_5_3:
      x_schro_iwt_5_3 (p);
      break;
    case SCHRO_WAVELET_13_5:
      x_schro_iwt_13_5 (p);
      break;
    case SCHRO_WAVELET_HAAR_0:
      x_schro_iwt_haar0 (p);
      break;
    case SCHRO_WAVELET_HAAR_1:
      x_schro_iwt_haar1 (p);
      break;
    case SCHRO_WAVELET_HAAR_2:
      x_schro_iwt_haar2 (p);
      break;
    case SCHRO_WAVELET_FIDELITY:
      x_schro_iwt_fidelity (p);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      x_schro_iwt_daub_9_7(p);
      break;
  }
}



