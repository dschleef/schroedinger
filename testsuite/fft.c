
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define COMPLEX_MULT_R(a,b,c,d) ((a)*(c) - (b)*(d))
#define COMPLEX_MULT_I(a,b,c,d) ((a)*(d) + (b)*(c))

/* reference */

void
DFT_ref (double *d1, double *d2, double *s1, double *costable,
    double *sintable, int shift)
{
  int n = 1<<shift;
  int mask = n-1;
  double x;
  double y;
  int i;
  int j;

  for(i=0;i<n;i++){
    x = 0;
    y = 0;
    for(j=0;j<n;j++){
      x += s1[j] * costable[(i*j)&mask];
      y += s1[j] * sintable[(i*j)&mask];
    }
    d1[i] = x;
    d2[i] = y;
  }
}

/* attempt 1 */

void
fft_stage_try1 (double *d1, double *d2, int dstr, double *s1, double *s2,
    int sstr, double *costable, double *sintable, int tstr, int shift)
{
  int n = 1<<shift;
  int half_n = n>>1;
  int i;
  double x,y;

  for(i=0;i<half_n;i++){
    x = COMPLEX_MULT_R(s1[(2*i+1)*sstr], s2[(2*i+1)*sstr],
        costable[i*tstr], sintable[i*tstr]);
    y = COMPLEX_MULT_I(s1[(2*i+1)*sstr], s2[(2*i+1)*sstr],
        costable[i*tstr], sintable[i*tstr]);

    d1[(i)*dstr] = s1[(2*i)*sstr] + x;
    d2[(i)*dstr] = s2[(2*i)*sstr] + y;
    d1[(i + half_n)*dstr] = s1[(2*i)*sstr] - x;
    d2[(i + half_n)*dstr] = s2[(2*i)*sstr] - y;
  }
}

void
DFT_try1 (double *d1, double *d2, double *s1, double *costable,
    double *sintable, int shift)
{
  int i;
  int j;
  double *s2;
  int n = 1<<shift;
  int skip;

  s2 = malloc (sizeof(double)*n);
  memset (s2, 0, sizeof(double)*n);

  for(i = 0; i<shift; i++){
    skip = 1<<(shift - i - 1);
    for(j=0;j<skip; j++) {
      fft_stage_try1 (d1 + j, d2 + j, skip, s1 + j, s2 + j, skip,
          costable, sintable, skip, i+1);
    }
    memcpy(s1, d1, sizeof(double)*n);
    memcpy(s2, d2, sizeof(double)*n);
  }

  free(s2);
}

/* attempt 2 */

void
fft_stage_try2 (double *d1, double *d2, double *s1, double *s2,
    double *costable, double *sintable, int i,
    int shift)
{
  int j;
  int k;
  double x,y;
  int skip;
  int half_n;
  int offset;

  half_n = 1<<i;
  skip = 1<<(shift - i - 1);
  for(j=0;j<skip; j++) {
    for(k=0;k<half_n;k++){
      offset = 2*k*skip;
      x = COMPLEX_MULT_R(s1[offset + skip + j], s2[offset + skip + j],
          costable[k*skip], sintable[k*skip]);
      y = COMPLEX_MULT_I(s1[offset + skip + j], s2[offset + skip + j],
          costable[k*skip], sintable[k*skip]);

      d1[k*skip + j] = s1[offset + j] + x;
      d2[k*skip + j] = s2[offset + j] + y;
      d1[k*skip + half_n*skip + j] = s1[offset + j] - x;
      d2[k*skip + half_n*skip + j] = s2[offset + j] - y;
    }
  }
}

void
DFT_try2 (double *d1, double *d2, double *s1, double *costable,
    double *sintable, int shift)
{
  int i;
  double *s2;
  int n = 1<<shift;
  double *tmp;
  double *tmp1_1, *tmp1_2, *tmp2_1, *tmp2_2;

  s2 = malloc (sizeof(double)*n);
  memset (s2, 0, sizeof(double)*n);

  tmp = malloc (4*sizeof(double)*n);
  tmp1_1 = tmp;
  tmp1_2 = tmp + n;
  tmp2_1 = tmp + 2*n;
  tmp2_2 = tmp + 3*n;

  i = 0;
  fft_stage_try2 (tmp1_1, tmp1_2, s1, s2, costable, sintable, i, shift);
  for(i = 1; i < shift-2; i+=2){
    fft_stage_try2 (tmp2_1, tmp2_2, tmp1_1, tmp1_2, costable, sintable,
        i, shift);
    fft_stage_try2 (tmp1_1, tmp1_2, tmp2_1, tmp2_2, costable, sintable,
        i+1, shift);
  }
  if (i < shift - 1) {
    fft_stage_try2 (tmp2_1, tmp2_2, tmp1_1, tmp1_2, costable, sintable,
        i, shift);
    fft_stage_try2 (d1, d2, tmp2_1, tmp2_2, costable, sintable,
        i+1, shift);
  } else {
    fft_stage_try2 (d1, d2, tmp1_1, tmp1_2, costable, sintable,
        i, shift);
  }

  free(tmp);
  free(s2);
}

/* attempt 3 */

void
fft_stage_step0_try3 (double *d1, double *d2, double *s1, double *s2,
    double *costable, double *sintable,
    int shift)
{
  int j;
  double x,y;
  int skip;
  int half_n;

  half_n = 1<<(shift - 1);
  skip = 1<<(shift - 1);
  for(j=0; j<skip; j++) {
    x = COMPLEX_MULT_R(s1[skip + j], s2[skip + j], 1, 0);
    y = COMPLEX_MULT_I(s1[skip + j], s2[skip + j], 1, 0);

    d1[j] = s1[j] + x;
    d2[j] = s2[j] + y;
    d1[half_n + j] = s1[j] - x;
    d2[half_n + j] = s2[j] - y;
  }
}

void
fft_stage_step1_try3 (double *d1, double *d2, double *s1, double *s2,
    double *costable, double *sintable, int shift)
{
  int j;
  double x,y;
  int skip;
  int half_n;

  half_n = 1<<(shift - 1);
  skip = 1<<(shift - 2);
  for(j=0; j<skip; j++) {
    x = COMPLEX_MULT_R(s1[skip + j], s2[skip + j], 1, 0);
    y = COMPLEX_MULT_I(s1[skip + j], s2[skip + j], 1, 0);

    d1[j] = s1[j] + x;
    d2[j] = s2[j] + y;
    d1[half_n + j] = s1[j] - x;
    d2[half_n + j] = s2[j] - y;
  }

  for(j=skip; j<skip*2; j++) {
    x = COMPLEX_MULT_R(s1[half_n + j], s2[half_n + j], 0, 1);
    y = COMPLEX_MULT_I(s1[half_n + j], s2[half_n + j], 0, 1);

    d1[j] = s1[half_n + j - skip] + x;
    d2[j] = s2[half_n + j - skip] + y;
    d1[half_n + j] = s1[half_n + j - skip] - x;
    d2[half_n + j] = s2[half_n + j - skip] - y;
  }
}

void
fft_stage_try3 (double *d1, double *d2, double *s1, double *s2,
    double *costable, double *sintable, int i,
    int shift)
{
  int j;
  int k;
  double x,y;
  int skip;
  int half_n;
  int offset;

  half_n = 1<<(shift - 1);
  skip = 1<<(shift - 1 - i);
  for(j=0;j<skip; j++) {
    for(k=0;k<(1<<i);k++){
      offset = 2*k*skip;
      x = COMPLEX_MULT_R(s1[offset + skip + j], s2[offset + skip + j],
          costable[k*skip], sintable[k*skip]);
      y = COMPLEX_MULT_I(s1[offset + skip + j], s2[offset + skip + j],
          costable[k*skip], sintable[k*skip]);

      d1[k*skip + j] = s1[offset + j] + x;
      d2[k*skip + j] = s2[offset + j] + y;
      d1[k*skip + half_n + j] = s1[offset + j] - x;
      d2[k*skip + half_n + j] = s2[offset + j] - y;
    }
  }
}

void
DFT_try3 (double *d1, double *d2, double *s1, double *costable,
    double *sintable, int shift)
{
  int i;
  double *s2;
  int n = 1<<shift;

  s2 = malloc (sizeof(double)*n);
  memset (s2, 0, sizeof(double)*n);

  fft_stage_step0_try3 (d1, d2, s1, s2, costable, sintable, shift);
  memcpy(s1, d1, sizeof(double)*n);
  memcpy(s2, d2, sizeof(double)*n);
  fft_stage_step1_try3 (d1, d2, s1, s2, costable, sintable, shift);
  memcpy(s1, d1, sizeof(double)*n);
  memcpy(s2, d2, sizeof(double)*n);
  for(i = 2; i<shift; i++){
    fft_stage_try3 (d1, d2, s1, s2, costable, sintable, i, shift);
    memcpy(s1, d1, sizeof(double)*n);
    memcpy(s2, d2, sizeof(double)*n);
  }

  free(s2);
}




void sincos_array (double *d1, double *d2, double inc, int n)
{
  int i; 

  for(i=0;i<n;i++){
    d1[i] = cos(inc*i);
    d2[i] = sin(inc*i);
  }
}

double costable[65536];
double sintable[65536];

double s1[65536];
double s2[65536];
double d1_test[65536];
double d2_test[65536];
double d1_ref[65536];
double d2_ref[65536];


int
main (int argc, char *argv[])
{
  int n;
  int shift = 10;
  int i;
  double sum;

  n = 1<<shift;

  sincos_array (costable, sintable, 2*M_PI/n, n);

  for(i=0;i<n;i++){
    s1[i] = (rand() & 0xff);
    s2[i] = 0;
  }

  DFT_ref (d1_ref, d2_ref, s1, costable, sintable, shift);
  DFT_try2 (d1_test, d2_test, s1, costable, sintable, shift);

  sum = 0;
  for(i=0;i<n;i++){
    printf("%d %g %g %g %g\n", i, d1_ref[i], d2_ref[i], d1_test[i], d2_test[i]);
    sum += fabs(d1_ref[i] - d1_test[i]);
    sum += fabs(d2_ref[i] - d2_test[i]);
  }
  printf("%g\n", sum);

  return 0;
}


