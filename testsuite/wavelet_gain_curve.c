
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrofft.h>
#include <schroedinger/schroorc.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"

int16_t tmp[2000];
int16_t tmp2[2000];

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

void dump (int16_t *a, int n);
void dump_cmp (int16_t *a, int16_t *b, int n);
void solve (double *matrix, double *col, int n);

#define SHIFT 8
#define N (1<<SHIFT)


void
random_test(double *dest, int filter, double *weights)
{
  int16_t *a = tmp+10;
  float sintable[N];
  float costable[N];
  float dr[N];
  float di[N];
  float sr[N];
  float si[N];
  double power[N];
  int i;
  int j;
  int n_cycles = 10000;
  int amplitude = 100;
  double p0 = 0;

  for(i=0;i<N;i++){
    power[i] = 0;
  }

  for(j=0;j<n_cycles;j++){
    for(i=0;i<N;i++){
      a[i] = 0;
    }
    for(i=0;i<(N>>4);i++){
      a[i]=floor(0.5 + amplitude*random_std()*weights[0]);
    }
    for(i=(N>>4);i<(N>>3);i++){
      a[i]=floor(0.5 + amplitude*random_std()*weights[1]);
    }
    for(i=(N>>3);i<(N>>2);i++){
      a[i]=floor(0.5 + amplitude*random_std()*weights[2]);
    }
    for(i=(N>>2);i<(N>>1);i++){
      a[i]=floor(0.5 + amplitude*random_std()*weights[3]);
    }
    for(i=(N>>1);i<(N>>0);i++){
      a[i]=floor(0.5 + amplitude*random_std()*weights[4]);
    }

    synth(a,N>>3,filter);
    synth(a,N>>2,filter);
    synth(a,N>>1,filter);
    synth(a,N,filter);

    for(i=0;i<N;i++){
      si[i] = 0;
      sr[i] = a[i];
      p0 += a[i]*a[i];
    }

    schro_fft_generate_tables_f32 (costable, sintable, SHIFT);
    schro_fft_fwd_f32 (dr, di, sr, si, costable, sintable, SHIFT);

    for(i=0;i<N;i++){
      power[i] += dr[i]*dr[i]+di[i]*di[i];
    }

  }

  for(i=0;i<(N>>1);i++){
    double x;

    x = power[i]/n_cycles;
    x /= (amplitude*amplitude);
    /* divide by FFT normalization */
    x /= (1<<SHIFT);

    dest[i] = x;
  }
}

void
print_subband_quants (double *a, int filter, int n_levels)
{
  double min;
  double c[10];
  double b[20];
  int i;

  c[0] = 0;
  for(i=0;i<n_levels+1;i++){
    c[i] = 1/sqrt(a[i]);
    //printf("%d %g\n", i, c[i]);
  }

  b[0] = c[0] * c[0] / (1<<(n_levels*filtershift[filter])); 
  for(i=0;i<n_levels;i++){
    b[i*2+1] = c[i+0] * c[i+1] / (1<<((n_levels-i)*filtershift[filter]));
    b[i*2+2] = c[i+1] * c[i+1] / (1<<((n_levels-i)*filtershift[filter]));
  }

  min = b[0];
  for(i=0;i<n_levels*2+1;i++){
    if (b[i] < min) min = b[i];
  }

  for(i=0;i<n_levels*2+1;i++){
    printf("%d %5.3f %5.3f %d\n", i, b[i], b[i]/min, gain_to_quant_index(b[i]/min));
  }
}

int
main (int argc, char *argv[])
{
  int filter;
  double curves[5][N];
  double matrix[5*5];
  double column[5];
  double target[N];
  int i;
  int j;

  schro_init();

  filter = 0;
  if (argc > 1) {
    filter = strtol(argv[1], NULL, 0);
  }

  for(j=0;j<5;j++){
    double weights[5];

    memset(weights,0,sizeof(weights));
    weights[j] = 1;
    random_test(curves[j], filter, weights);
  }

  for(i=0;i<N/2;i++){
    //double x = ((double)i)/(N/2);
    //target[i] = 1.0 + 4*x*x;
    target[i] = 1.0;
  }
  for(i=0;i<N/2;i++){
    for(j=0;j<5;j++){
      curves[j][i] /= target[i];
    }
  }
  for(i=0;i<5;i++){
    column[i] = sum_f64 (curves[i], N/2);
    matrix[i*5+i] = multsum_f64 (curves[i], curves[i], N/2);
    for(j=i+1;j<5;j++){
      matrix[i*5+j] = multsum_f64 (curves[i], curves[j], N/2);
      matrix[j*5+i] = matrix[i*5+j];
    }
  }
  for(i=0;i<N/2;i++){
    for(j=0;j<5;j++){
      curves[j][i] *= target[i];
    }
  }

  solve (matrix, column, 5);

#if 0
  for(i=0;i<N/2;i++){
    double x = 0;
    for(j=0;j<5;j++){
      x += column[j] * curves[j][i];
    }
#if 1
    printf("%d %g %g %g %g %g %g\n", i, x,
      column[0] * curves[0][i],
      column[1] * curves[1][i],
      column[2] * curves[2][i],
      column[3] * curves[3][i],
      column[4] * curves[4][i]);
#else
    printf("%d %g %g %g %g %g %g\n", i, x,
      curves[0][i], curves[1][i], curves[2][i], curves[3][i], curves[4][i]);
#endif
  }
#else
  print_subband_quants (column, filter, 4);
#endif

  return 0;
}

void
solve (double *matrix, double *column, int n)
{
  int i;
  int j;
  int k;
  double x;

  for(i=0;i<n;i++){
    x = 1/matrix[i*n+i];
    for(k=i;k<n;k++) {
      matrix[i*n+k] *= x;
    }
    column[i] *= x;

    for(j=i+1;j<n;j++){
      x = matrix[j*n+i];
      for(k=i;k<n;k++) {
        matrix[j*n+k] -= matrix[i*n+k] * x;
      }
      column[j] -= column[i] * x;
    }
  }

  for(i=n-1;i>0;i--) {
    for(j=i-1;j>=0;j--) {
      column[j] -= matrix[j*n+i] * column[i];
      matrix[j*n+i] = 0;
    }
  }
}


