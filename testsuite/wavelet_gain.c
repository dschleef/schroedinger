
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

int filtershift[] = { 1, 1, 1, 0, 1, 2, 0, 1 };

void synth(int16_t *a, int filter, int n);
void split (int16_t *a, int filter, int n);
void synth_schro_ext (int16_t *a, int filter, int n);
void split_schro_ext (int16_t *a, int n, int filter);
void deinterleave (int16_t *a, int n);
void interleave (int16_t *a, int n);
void dump (int16_t *a, int n);
void dump_cmp (int16_t *a, int16_t *b, int n);

#define SHIFT 8
#define N (1<<SHIFT)


void
random_test(int filter, int level)
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
  double total;
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
    if (level == 0) {
      for(i=0;i<(N>>4);i++){
        a[i]=rint(amplitude*random_std());
      }
    } else if (level<5) {
      for(i=((N>>5)<<level);i<((N>>4)<<level);i++){
        a[i]=rint(amplitude*random_std());
      }
    } else if (level==5) {
      for(i=0;i<(N>>4);i++){
        a[i]=rint(amplitude*random_std()/3.608);
      }
      for(i=(N>>4);i<(N>>3);i++){
        a[i]=rint(amplitude*random_std()/1.972);
      }
      for(i=(N>>3);i<(N>>2);i++){
        a[i]=rint(amplitude*random_std()/1.382);
      }
      for(i=(N>>2);i<(N>>1);i++){
        a[i]=rint(amplitude*random_std()/0.991);
      }
      for(i=(N>>1);i<(N>>0);i++){
        a[i]=rint(amplitude*random_std()/0.821);
      }
    } else if (level==6) {
      double alpha = 1.282;
      double beta = 0.821;

      for(i=0;i<(N>>4);i++){
        a[i]=rint(amplitude*random_std()/(alpha*alpha*alpha*alpha));
      }
      for(i=(N>>4);i<(N>>3);i++){
        a[i]=rint(amplitude*random_std()/(alpha*alpha*alpha*beta));
      }
      for(i=(N>>3);i<(N>>2);i++){
        a[i]=rint(amplitude*random_std()/(alpha*alpha*beta));
      }
      for(i=(N>>2);i<(N>>1);i++){
        a[i]=rint(amplitude*random_std()/(alpha*beta));
      }
      for(i=(N>>1);i<(N>>0);i++){
        a[i]=rint(amplitude*random_std()/(beta));
      }
    }

    interleave(a,N>>3);
    synth(a,N>>3,filter);
    interleave(a,N>>2);
    synth(a,N>>2,filter);
    interleave(a,N>>1);
    synth(a,N>>1,filter);
    interleave(a,N);
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

  total = 0;
  for(i=0;i<(N>>1);i++){
    double x;

    x = power[i]/n_cycles;
    x /= (amplitude*amplitude);
#if 0
    if (level == 0) {
      x /= (1<<(filtershift[filter]*4));
    } else {
      x /= (1<<(filtershift[filter]*(5-level)));
    }
#endif
    /* divide by size of subband */
#if 0
    if (level == 0) {
      x /= (1<<(SHIFT-4+level));
    } else {
      x /= (1<<(SHIFT-5+level));
    }
#endif
    /* divide by FFT normalization */
    x /= (1<<SHIFT);
    /* unknown */
    x *= 8;

    total += x;
    printf("%d %g %g\n", i, x, total);
  }
  fprintf(stderr, "%g\n", p0/n_cycles/(1<<(SHIFT-5+level)));
  printf("\n");
}


void
response_test(int filter)
{
  int16_t *a = tmp+10;
  double p0, p1, p2, p3, p4;
  int i;
  int j;
  int k;

  for(j=0;j<128;j++){
    p0 = 0;
    p1 = 0;
    p2 = 0;
    p3 = 0;
    p4 = 0;

    for(k=0;k<100;k++){
      for(i=0;i<256;i++){
        a[i] = 128*sin(i*2*M_PI*j/256.0 + k*(M_PI/100.0));
      }

      split(a,256,filter);
      deinterleave(a,256);
      split(a,128,filter);
      deinterleave(a,128);
      split(a,64,filter);
      deinterleave(a,64);
      split(a,32,filter);
      deinterleave(a,32);

      for(i=0;i<16;i++){
        p0 += a[i]*a[i]*(1<<(filtershift[filter]*4));
      }
      for(i=16;i<32;i++){
        p1 += a[i]*a[i]*(1<<(filtershift[filter]*3));
      }
      for(i=32;i<64;i++){
        p2 += a[i]*a[i]*(1<<(filtershift[filter]*2));
      }
      for(i=64;i<128;i++){
        p3 += a[i]*a[i]*(1<<(filtershift[filter]*1));
      }
      for(i=128;i<256;i++){
        p4 += a[i]*a[i];
      }
    }
    printf("%d %g %g %g %g %g\n", j, p0, p1, p2, p3, p4);
  }

  printf("\n");
}

int
main (int argc, char *argv[])
{
  int filter;
  int level;

  schro_init();

  filter = 0;
  if (argc > 1) {
    filter = strtol(argv[1], NULL, 0);
  }

  if (1) {
    for(level=0;level<=6;level++){
      random_test(filter, level);
    }
  } else {
    response_test (filter);
  }

  return 0;
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

