
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

void synth(int16_t *a, int filter, int n);
void split (int16_t *a, int filter, int n);
void synth_schro_ext (int16_t *a, int filter, int n);
void split_schro_ext (int16_t *a, int n, int filter);
void deinterleave (int16_t *a, int n);
void interleave (int16_t *a, int n);
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
      a[i]=rint(amplitude*random_std()*weights[0]);
    }
    for(i=(N>>4);i<(N>>3);i++){
      a[i]=rint(amplitude*random_std()*weights[1]);
    }
    for(i=(N>>3);i<(N>>2);i++){
      a[i]=rint(amplitude*random_std()*weights[2]);
    }
    for(i=(N>>2);i<(N>>1);i++){
      a[i]=rint(amplitude*random_std()*weights[3]);
    }
    for(i=(N>>1);i<(N>>0);i++){
      a[i]=rint(amplitude*random_std()*weights[4]);
    }

    interleave(a,N>>3);
    synth_schro_ext(a,N>>3,filter);
    interleave(a,N>>2);
    synth_schro_ext(a,N>>2,filter);
    interleave(a,N>>1);
    synth_schro_ext(a,N>>1,filter);
    interleave(a,N);
    synth_schro_ext(a,N,filter);

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


void
interleave (int16_t *a, int n)
{
  int i;
  for(i=0;i<n/2;i++){
    tmp2[i*2] = a[i];
    tmp2[i*2 + 1] = a[n/2 + i];
  }
  for(i=0;i<n;i++){
    a[i] = tmp2[i];
  }
}

void
deinterleave (int16_t *a, int n)
{
  int i;
  for(i=0;i<n/2;i++){
    tmp2[i] = a[i*2];
    tmp2[n/2 + i] = a[i*2+1];
  }
  for(i=0;i<n;i++){
    a[i] = tmp2[i];
  }
}

void
extend(int16_t *a, int n)
{
  a[-8] = a[0];
  a[-7] = a[1];
  a[-6] = a[0];
  a[-5] = a[1];
  a[-4] = a[0];
  a[-3] = a[1];
  a[-2] = a[0];
  a[-1] = a[1];
  a[n+0] = a[n-2];
  a[n+1] = a[n-1];
  a[n+2] = a[n-2];
  a[n+3] = a[n-1];
  a[n+4] = a[n-2];
  a[n+5] = a[n-1];
  a[n+6] = a[n-2];
  a[n+7] = a[n-1];
}

void
synth(int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (a[i] + a[i+2] + 1)>>1;
      }
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      for(i=0;i<n;i+=2){
        a[i] -= (a[i+1] + 1)>>1;
      }
      for(i=0;i<n;i+=2){
        a[i+1] += a[i];
      }
      break;
    case SCHRO_WAVELET_FIDELITY:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
      break;
  }
}

void
split (int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (a[i] + a[i+2] + 1)>>1;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      for(i=0;i<n;i+=2){
        a[i+1] -= a[i];
      }
      for(i=0;i<n;i+=2){
        a[i] += (a[i+1] + 1)>>1;
      }
      break;
    case SCHRO_WAVELET_FIDELITY:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
      break;
  }
}

void
split_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[2000], *hi;
  int16_t tmp2[2000], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  orc_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_split_ext_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_split_ext_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_split_ext_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_split_ext_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_ext_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_split_ext_daub97(hi, lo, n/2);
      break;
  }
  orc_interleave2_s16 (a, hi, lo, n/2);

}

void
synth_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[2000], *hi;
  int16_t tmp2[2000], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  orc_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_synth_ext_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_synth_ext_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_synth_ext_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_synth_ext_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_ext_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_synth_ext_daub97(hi, lo, n/2);
      break;
  }

  orc_interleave2_s16 (a, hi, lo, n/2);
}




