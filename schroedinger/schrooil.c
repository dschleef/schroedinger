
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

#ifdef unused
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


