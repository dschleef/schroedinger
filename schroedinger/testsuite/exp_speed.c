
#include <stdio.h>
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>
#include <liboil/liboilprofile.h>

#include "arith_exp.h"

#define N 10000

unsigned char in[N];
unsigned char data[N];

int
main (int argc, char *argv[])
{
  int i;
  int x;
  ArithExp coder = { 0 };
  OilProfile prof;
  double ave, std;
  int j;

  oil_init();

  for(x = 0; x <= 256; x += 1) {
    for(i=0;i<N;i++) {
      in[i] = (oil_rand_u8() < x);
    }

    oil_profile_init (&prof);
    for(j=0;j<10;j++){
      arith_exp_init (&coder);
      coder.dataptr = data;

      oil_profile_start (&prof);
      for (i=0;i<N;i++){
        arith_exp_encode (&coder, 0, in[i]);
      }
      oil_profile_stop (&prof);
    }

    oil_profile_get_ave_std (&prof, &ave, &std);
    printf("%g %g (%g)\n", x/256.0, ave/N, std/N);
  }

  return 0;
}




