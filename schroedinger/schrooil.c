
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrooil.h>
#include <math.h>


#ifdef unused
void
schro_mas10_s16 (int16_t *d, const int16_t *s1_np3, const int32_t *s2_4,
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
schro_convert_f64_u8 (double *dest, uint8_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i];
  }
}

void
schro_iir3_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n)
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
schro_iir3_rev_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n)
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
schro_iir3_across_u8_f64 (uint8_t *d, uint8_t *s, double *i1, double *i2, double *i3,
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
schro_iir3_across_s16_f64 (int16_t *d, int16_t *s, double *i1, double *i2, double *i3,
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
schro_convert_f64_s16 (double *dest, int16_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[i];
  }
}

void
schro_iir3_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n)
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
schro_iir3_rev_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n)
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


