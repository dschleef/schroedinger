
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

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

#define SHIFT 8
#define N (1<<SHIFT)

double
gain_test(int filter, int level, int n_levels)
{
  int16_t *a = tmp+10;
  double gain;
  int i;
  int j;
  double p1,p2;

  gain = 0;

  for(j=0;j<10000;j++){
    p1 = 0;
    p2 = 0;
    for(i=0;i<N;i++){
      a[i] = 0;
    }
    if (level == 0) {
      for(i=0;i<N>>n_levels;i++){
        a[i]=rint(100 * random_std());
      }
    } else {
      for(i=N>>(n_levels-level+1);i<N>>(n_levels-level);i++){
        a[i]=rint(100 * random_std());
      }
    }
    for(i=0;i<N;i++){
      p1 += a[i]*a[i];
    }

    for(i=0;i<n_levels;i++){
      interleave(a,N>>(n_levels-1-i));
      synth_schro_ext(a,N>>(n_levels-1-i),filter);
    }

    for(i=0;i<256;i++){
      p2 += a[i]*a[i];
    }

    gain += p2/p1;
  }

  //printf("%d %g %g\n", level, gain/10000, sqrt(gain/10000));

  return sqrt(gain/10000);
}

int
main (int argc, char *argv[])
{
  int filter;
  int n_levels;

  schro_init();

  filter = 0;
  if (argc > 1) {
    filter = strtol(argv[1], NULL, 0);
  }

  {
    double a[5];
    double b[9];
    int i;
    double min;

    for(n_levels=1;n_levels<=4;n_levels++){
      printf("n_levels=%d:\n", n_levels);
      for(i=0;i<=n_levels;i++){
        a[i] = gain_test (filter, i, n_levels);
      }
      for(i=0;i<=n_levels;i++){
        printf("%d %5.3f\n", i, a[i]);
      }

      b[0] = a[0] * a[0] / (1<<(n_levels*filtershift[filter]));
      for(i=0;i<n_levels;i++){
        b[i*2+1] = a[i+0] * a[i+1] / (1<<((n_levels-i)*filtershift[filter]));
        b[i*2+2] = a[i+1] * a[i+1] / (1<<((n_levels-i)*filtershift[filter]));
      }

      min = b[0];
      for(i=0;i<n_levels*2+1;i++){
        if (b[i] < min) min = b[i];
      }

      for(i=0;i<n_levels*2+1;i++){
        printf("%d %5.3f %5.3f %d\n", i, b[i], b[i]/min, gain_to_quant_index(b[i]/min));
      }
    }
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
split_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[2000], *hi;
  int16_t tmp2[2000], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

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
  oil_interleave2_s16 (a, hi, lo, n/2);

}

void
synth_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[2000], *hi;
  int16_t tmp2[2000], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

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

  oil_interleave2_s16 (a, hi, lo, n/2);
}




