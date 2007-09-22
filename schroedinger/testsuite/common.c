
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrofft.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


double sgn(double x)
{
  if (x<0) return -1;
  if (x>0) return 1;
  return 0;
}

double
random_std (void)
{
  double x;
  double y;

  while (1) {
    x = -5.0 + random () * (1.0/RAND_MAX) * 10;
    y = random () * (1.0/RAND_MAX);

    if (y < exp(-x*x*0.5)) return x;
  }
}

double
random_triangle (void)
{
  return random () * (1.0/RAND_MAX) - random () * (1.0/RAND_MAX);
}

int
gain_to_quant_index (double x)
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

double
sum_f64 (double *a, int n)
{
  double sum = 0;
  int i;
  for(i=0;i<n;i++){
    sum += a[i];
  }
  return sum;
}

double
multsum_f64 (double *a, double *b, int n)
{
  double sum = 0;
  int i;
  for(i=0;i<n;i++){
    sum += a[i]*b[i];
  }
  return sum;
}


