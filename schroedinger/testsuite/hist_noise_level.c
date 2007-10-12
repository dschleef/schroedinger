
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>
#include <schroedinger/schrohistogram.h>
#include <liboil/liboilprofile.h>
#include <math.h>

#include "common.h"


#define N 10000


int16_t testdata[N];
SchroHistogram h;


void
test (double s)
{
  SchroHistogram *hist = &h;
  int i;
  double sigma;

  for(i=0;i<N;i++){
    testdata[i] = rint(s*random_std ());
  }

  schro_histogram_init (hist);
  schro_histogram_add_array_s16(hist, testdata, N);

  sigma = schro_histogram_estimate_noise_level (hist);

  printf("%g %g\n", s, sigma);

}

static int
iexpx (int x)
{
  if (x < (1<<SCHRO_HISTOGRAM_SHIFT)) return x;

  return ((1<<SCHRO_HISTOGRAM_SHIFT)|(x&((1<<SCHRO_HISTOGRAM_SHIFT)-1))) << ((x>>SCHRO_HISTOGRAM_SHIFT)-1);
}

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  for(i=1;i<90;i++){
    test(iexpx(i));
  }

  return 0;
}


