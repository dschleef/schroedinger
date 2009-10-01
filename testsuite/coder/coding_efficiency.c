
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <math.h>

#include "arith.h"

#define N 100000


int efficiency_arith_dirac (int x, unsigned char *data, int n);
int efficiency_arith_exp (int x, unsigned char *data, int n);
int efficiency_arith_dirac_byte (int x, unsigned char *data, int n);
int efficiency_arith_dirac_stats (int x, unsigned char *data, int n);
int efficiency_arith_dirac_both (int x, unsigned char *data, int n);
int efficiency_arith_qm (int x, unsigned char *data, int n);


double
efficiency_entropy (double x)
{
  if (x == 0 || x == 1) return 0;

  return (x*log(x) + (1-x)*log(1-x))/log(0.5);
}


unsigned char data[N];

int
main (int argc, char *argv[])
{
  int x;
  double a, b, c, d, e, f;
  double y;

  schro_init();

  for(x = 0; x <= 256; x += 1) {
    y = efficiency_entropy (x/256.0);
    a = efficiency_arith_dirac (x, data, N) / (double)N / y;
    b = efficiency_arith_qm (x, data, N) / (double)N / y;
    c = efficiency_arith_dirac_byte (x, data, N) / (double)N / y;
    d = efficiency_arith_dirac_stats (x, data, N) / (double)N / y;
    e = efficiency_arith_dirac_both (x, data, N) / (double)N / y;
    f = efficiency_arith_exp (x, data, N) / (double)N / y;

    printf("%g %g %g %g %g %g %g\n", x/256.0, a, b, c, d, e, f);
  }

  return 0;
}


