
#include "config.h"

#include <stdio.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <math.h>

#include "arith.h"

#define N 1000


double speed_arith_dirac (int x, unsigned char *data, int n);
double speed_arith_qm (int x, unsigned char *data, int n);
double speed_arith_dirac_byte (int x, unsigned char *data, int n);
double speed_arith_dirac_stats (int x, unsigned char *data, int n);
double speed_arith_dirac_both (int x, unsigned char *data, int n);
double speed_arith_exp (int x, unsigned char *data, int n);

unsigned char data[N];

int
main (int argc, char *argv[])
{
  int x;

  schro_init();

  printf("Number of cycles for 5%% 0's, 95%% 1's:\n");
  x = 256*0.05;
  printf("  Dirac (original)               %g\n",
      speed_arith_dirac (x, data, N));
  printf("  Dirac (bytewise shift)         %g\n",
    speed_arith_dirac_byte (x, data, N));
  printf("  Dirac (periodic stats update)  %g\n",
    speed_arith_dirac_stats (x, data, N));
  printf("  Dirac (both)                   %g\n",
    speed_arith_dirac_both (x, data, N));
  printf("  Dirac (experimental)           %g\n",
    speed_arith_exp (x, data, N));
  printf("  QM coder                       %g\n",
    speed_arith_qm (x, data, N));

  printf("Number of cycles for 40%% 0's, 60%% 1's:\n");
  x = 256*0.40;
  printf("  Dirac (original)               %g\n",
      speed_arith_dirac (x, data, N));
  printf("  Dirac (bytewise shift)         %g\n",
    speed_arith_dirac_byte (x, data, N));
  printf("  Dirac (periodic stats update)  %g\n",
    speed_arith_dirac_stats (x, data, N));
  printf("  Dirac (both)                   %g\n",
    speed_arith_dirac_both (x, data, N));
  printf("  Dirac (experimental)           %g\n",
    speed_arith_exp (x, data, N));
  printf("  QM coder                       %g\n",
    speed_arith_qm (x, data, N));

  return 0;
}


