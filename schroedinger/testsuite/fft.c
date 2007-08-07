
#include <stdio.h>
#include <math.h>
#include <string.h>


#define COMPLEX_MULT_R(a,b,c,d) ((a)*(c) - (b)*(d))
#define COMPLEX_MULT_I(a,b,c,d) ((a)*(d) + (b)*(c))


void
fft_stage (double *d1, double *d2, int dstr, double *s1, double *s2,
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
discrete_fourier_transform_ref (double *d1, double *d2, double *s1, double *s2,
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
double d1[65536];
double d2[65536];


int
main (int argc, char *argv[])
{
  int n;
  int shift = 5;
  int i;
  int skip;
  int j;

  n = 1<<shift;

  sincos_array (costable, sintable, 2*M_PI/n, n);

  for(i=0;i<n;i++){
    s1[i] = sin(2*M_PI/n * i) + cos(2*M_PI/n * 3*i);
    //s1[i] = (i<(n/2)) - 0.5;
    s2[i] = 0;
  }

  for(i = 0; i<shift; i++){
    skip = 1<<(shift - i - 1);
    for(j=0;j<skip; j++) {
      fft_stage (d1 + j, d2 + j, skip, s1 + j, s2 + j, skip,
          costable, sintable, skip, i+1);
    }
    memcpy(s1, d1, sizeof(double)*n);
    memcpy(s2, d2, sizeof(double)*n);
  }

#if 0
  for(i=0;i<8;i++){
    fft_stage (d1 + i, d2 + i, 8, s1 + i, s2 + i, 8, costable, sintable, 8, 1);
  }

  for(i=0;i<4;i++){
    fft_stage (s1 + i, s2 + i, 4, d1 + i, d2 + i, 4, costable, sintable, 4, 2);
  }

  fft_stage (d1 + 0, d2 + 0, 2, s1 + 0, s2 + 0, 2, costable, sintable, 2, 3);
  fft_stage (d1 + 1, d2 + 1, 2, s1 + 1, s2 + 1, 2, costable, sintable, 2, 3);

  fft_stage (s1, s2, 1, d1, d2, 1, costable, sintable, 1, 4);
#endif

  for(i=0;i<n;i++){
    printf("%d %g %g\n", i, d1[i], d2[i]);
  }

  return 0;
}


