
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>


void
oil_splat_s16_ns (int16_t *dest, const int16_t *src, int n)
{
  oil_splat_u16_ns ((uint16_t *)dest, (const uint16_t *)src, n);
}

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


void
oil_mas10_across_u8 (uint8_t *d, const uint8_t *s1, const int16_t *s2_10,
    const int16_t *s3_2, int n)
{
  int i,j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<10;j++){
      x += s1[i+j] * s2_10[j];
    }
    x += s3_2[0];
    x >>= s3_2[1];
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    d[i] = x;
  }
}

void
oil_mas10_u8 (uint8_t *d, const uint8_t *s1, int sstr, const int16_t *s2_10,
    const int16_t *s3_2, int n)
{
  int i,j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<10;j++){
      x += s1[i+j*sstr] * s2_10[j];
    }
    x += s3_2[0];
    x >>= s3_2[1];
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    d[i] = x;
  }
}

void oil_add_const_rshift_u16 (uint16_t *d1, const uint16_t *s1,
    const int16_t *s2_2, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = (s1[i] + s2_2[0])>>s2_2[1];
  }
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

