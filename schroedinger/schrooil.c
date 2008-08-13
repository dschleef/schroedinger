
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>
#include <math.h>


void
oil_splat_s16_ns (int16_t *dest, const int16_t *src, int n)
{
  oil_splat_u16_ns ((uint16_t *)dest, (const uint16_t *)src, n);
}

#if 0
void
oil_lift_haar_split (int16_t *i1, int16_t *i2, int n)
{
  int i;
  for(i=0;i<n;i++){
    i2[i] -= i1[i];
    i1[i] += (i2[i] + 1)>>1;
  }
}

void
oil_lift_haar_synth (int16_t *i1, int16_t *i2, int n)
{
  int i;
  for(i=0;i<n;i++){
    i1[i] -= (i2[i] + 1)>>1;
    i2[i] += i1[i];
  }
}

void
oil_synth_haar (int16_t *d, const int16_t *s, int n)
{
  int i;

  for(i=0;i<n;i++){
    d[2*i] = s[2*i] - ((s[2*i+1] + 1)>>1);
    d[2*i + 1] = s[2*i+1] + d[2*i];
  }
}

void
oil_split_haar (int16_t *d, const int16_t *s, int n)
{
  int i;

  for(i=0;i<n;i++){
    d[2*i+1] = s[2*i+1] - s[2*i];
    d[2*i] = s[2*i] + ((d[2*i + 1] + 1)>>1);
  }
}

void
oil_multsumshift8_str_s16 (int16_t *d, const int16_t *s, int sstr,
    const int16_t *s2_8, const int16_t *s3_1, const int16_t *s4_1, int n)
{
  int i;
  int x;
  int j;

  for(i=0;i<n;i++){
    x = s3_1[0];
    for(j=0;j<8;j++){
      x += s[j] * s2_8[j];
    }
    d[i] = x >> s4_1[0];
    s = OFFSET(s,sstr);
  }
}
#endif

void
oil_sum_s32_u8 (int32_t *d_1, uint8_t *src, int n)
{
  int i;
  int x = 0;

  for(i=0;i<n;i++){
    x += src[i];
  }

  d_1[0] = x;
}

void
oil_sum_s32_s16 (int32_t *d_1, int16_t *src, int n)
{
  int i;
  int x = 0;

  for(i=0;i<n;i++){
    x += src[i];
  }

  d_1[0] = x;
}

void
oil_sum_square_diff_u8 (int32_t *d_1, uint8_t *s1, uint8_t *s2, int n)
{
  int sum = 0;
  int i;
  int x;

  for(i=0;i<n;i++){
    x = s1[i] - s2[i];
    sum += x*x;
  }
  d_1[0] = sum;
}

#if 0
void
oil_mas4_u8 (uint8_t *d, const uint8_t *s1_np3, const int16_t *s2_4,
    const int16_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<4;j++){
      x += s1_np3[i+j] * s2_4[j];
    }
    d[i] = (x + s3_2[0])>>s3_2[1];
  }
}

void
oil_mas4_s16 (int16_t *d, const int16_t *s1_np3, const int32_t *s2_4,
    const int32_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<4;j++){
      x += s1_np3[i+j] * s2_4[j];
    }
    d[i] = (x + s3_2[0])>>s3_2[1];
  }
}

void
oil_mas8_s16 (int16_t *d, const int16_t *s1_np3, const int32_t *s2_4,
    const int32_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<8;j++){
      x += s1_np3[i+j] * s2_4[j];
    }
    d[i] = (x + s3_2[0])>>s3_2[1];
  }
}
#endif

void
oil_mas10_s16 (int16_t *d, const int16_t *s1_np3, const int32_t *s2_4,
    const int32_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<10;j++){
      x += s1_np3[i+j] * s2_4[j];
    }
    d[i] = (x + s3_2[0])>>s3_2[1];
  }
}

#if 0
void
oil_mas8_across_u8 (uint8_t *d, uint8_t **s1_a8,
    const int16_t *s2_8, const int16_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<8;j++){
      x += s1_a8[j][i] * s2_8[j];
    }
    d[i] = CLAMP((x + s3_2[0])>>s3_2[1],0,255);
  }
}

void
oil_mas10_across_u8 (uint8_t *d, uint8_t **s1_a10,
    const int16_t *s2_10, const int16_t *s3_2, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<10;j++){
      x += s1_a10[j][i] * s2_10[j];
    }
    d[i] = CLAMP((x + s3_2[0])>>s3_2[1],0,255);
  }
}

void
oil_addc_rshift_clipconv_u8_s16 (uint8_t *d1, const int16_t *s1,
    const int16_t *s2_2, int n)
{
  int i;

  for(i=0;i<n;i++){
    d1[i] = CLAMP((s1[i] + s2_2[0])>>s2_2[1],0,255);
  }
}
#endif

void
oil_convert_f64_u8 (double *dest, uint8_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i];
  }
}

void
oil_iir3_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n)
{
  int i;

  for(i=0;i<n;i++){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i_3[0] + s2_4[2]*i_3[1] + s2_4[3]*i_3[2];
    i_3[2] = i_3[1];
    i_3[1] = i_3[0];
    i_3[0] = x;
    d[i] = rint(x);
  }
}

void
oil_iir3_rev_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n)
{
  int i;

  for(i=n-1;i>=0;i--){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i_3[0] + s2_4[2]*i_3[1] + s2_4[3]*i_3[2];
    i_3[2] = i_3[1];
    i_3[1] = i_3[0];
    i_3[0] = x;
    d[i] = rint(x);
  }
}

void
oil_iir3_across_u8_f64 (uint8_t *d, uint8_t *s, double *i1, double *i2, double *i3,
    double *s2_4, int n)
{
  int i;

  for(i=0;i<n;i++){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i1[i] + s2_4[2]*i2[i] + s2_4[3]*i3[i];
    i3[i] = i2[i];
    i2[i] = i1[i];
    i1[i] = x;
    d[i] = rint(x);
  }
}

void
oil_iir3_across_s16_f64 (int16_t *d, int16_t *s, double *i1, double *i2, double *i3,
    double *s2_4, int n)
{
  int i;

  for(i=0;i<n;i++){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i1[i] + s2_4[2]*i2[i] + s2_4[3]*i3[i];
    i3[i] = i2[i];
    i2[i] = i1[i];
    i1[i] = x;
    d[i] = rint(x);
  }
}

void
oil_convert_f64_s16 (double *dest, int16_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i];
  }
}

void
oil_iir3_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n)
{
  int i;

  for(i=0;i<n;i++){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i_3[0] + s2_4[2]*i_3[1] + s2_4[3]*i_3[2];
    i_3[2] = i_3[1];
    i_3[1] = i_3[0];
    i_3[0] = x;
    d[i] = rint(x);
  }
}

void
oil_iir3_rev_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n)
{
  int i;

  for(i=n-1;i>=0;i--){
    double x;

    x = s2_4[0]*s[i] + s2_4[1]*i_3[0] + s2_4[2]*i_3[1] + s2_4[3]*i_3[2];
    i_3[2] = i_3[1];
    i_3[1] = i_3[0];
    i_3[0] = x;
    d[i] = rint(x);
  }
}

#if 0
void
oil_mas12across_addc_rshift_u8 (uint8_t *dest, uint8_t **src,
    const int16_t *taps, const int16_t *offsetshift, int n)
{
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<12;j++){
      x += taps[j]*src[j][i];
    }
    dest[i] = CLAMP((x + offsetshift[0]) >> offsetshift[1],0,255);
  }
}
#endif


