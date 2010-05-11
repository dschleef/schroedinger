
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
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
        a[i]=floor(0.5 + 100 * random_std());
      }
    } else {
      for(i=N>>(n_levels-level+1);i<N>>(n_levels-level);i++){
        a[i]=floor(0.5 + 100 * random_std());
      }
    }
    for(i=0;i<N;i++){
      p1 += a[i]*a[i];
    }

    for(i=0;i<n_levels;i++){
      interleave(a,N>>(n_levels-1-i));
      synth(a,N>>(n_levels-1-i),filter);
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
quant_index (double x)
{
  int i = 0;

  x *= x;
  x *= x;
  while (x*x > 2) {
    x *= 0.5;
    i++;
  }

  return i;
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
        printf("%d %5.3f %5.3f %d\n", i, b[i], b[i]/min, quant_index(b[i]/min));
      }
    }
  }

  return 0;
}


