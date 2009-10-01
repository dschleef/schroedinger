
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"


void
discrete_fourier_transform (double *d1, double *d2, double *s1, double *s2,
    double *s3, int n)
{
  int mask = n-1;
  double x;
  double y;
  int i;
  int j;

  for(i=0;i<n;i++){
    x = 0;
    y = 0;
    for(j=0;j<n;j++){
      x += s1[j] * s2[(i*j)&mask];
      y += s1[j] * s3[(i*j)&mask];
    }
    d1[i] = x;
    d2[i] = y;
  }
}

void
complex_discrete_fourier_transform (double *d1, double *d2,
    double *s1, double *s2, double *s3, double *s4, int n)
{
  int mask = n-1;
  double x;
  double y;
  int i;
  int j;

  for(i=0;i<n;i++){
    x = 0;
    y = 0;
    for(j=0;j<n;j++){
      x += s1[j] * s3[(i*j)&mask] - s2[j] * s4[(j*i)&mask];
      y += s1[j] * s4[(i*j)&mask] + s2[j] * s3[(j*i)&mask];
    }
    d1[i] = x;
    d2[i] = y;
  }
}


void sincos_array (double *d1, double *d2, double inc, int n)
{
  int i;

  for(i=1;i<n;i++){
    d1[i] = cos(inc*i);
    d2[i] = sin(inc*i);
  }
}

void complex_mult (double *d1, double *d2, double *s1, double *s2,
    double *s3, double *s4, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = s1[i] * s3[i] - s2[i] * s4[i];
    d2[i] = s1[i] * s4[i] + s2[i] * s3[i];
  }
}

void complex_normalize (double *i1, double *i2, int n)
{
  int i;
  double x;
  for(i=0;i<n;i++){
    x = sqrt(i1[i]*i1[i] + i2[i]*i2[i]);
    if (x > 0) x = 1/x;
    i1[i] *= x;
    i2[i] *= x;
  }
}

#define N 256

int main (int argc, char *argv[])
{
  double s[N], c[N];
  double image1[N];
  double image2[N];
  double ft1r[N];
  double ft1i[N];
  double ft2r[N];
  double ft2i[N];
  double conv_r[N], conv_i[N];
  double resr[N], resi[N];
  int i;

  sincos_array (c, s, 2*M_PI/N, N);

  for(i=0;i<N;i++){
    image1[i] = rand_f64();
    image2[i] = 0;
  }

  for(i=0;i<N/2;i++){
    image2[i] = image1[i+N/4+N/16];
  }

  discrete_fourier_transform (ft1r, ft1i, image1, c, s, N);
  discrete_fourier_transform (ft2r, ft2i, image2, c, s, N);

  for(i=0;i<N;i++){
    ft2i[i] = -ft2i[i];
  }
  complex_mult (conv_r, conv_i, ft1r, ft1i, ft2r, ft2i, N);

  complex_normalize (conv_r, conv_i, N);

  complex_discrete_fourier_transform (resi, resr, conv_i, conv_r, c, s, N);

  for(i=0;i<N;i++){
    printf("%d %g %g %g %g\n", i, resr[i], resi[i], conv_r[i], conv_i[i]);
  }

  return 0;
}



