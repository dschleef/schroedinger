
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


int
main (int argc, char *argv[])
{
  int i;
  double ppd_max;
  double ppd;

  schro_init();

  ppd_max = 720 / (2.0*atan(0.5/3.0)*180/M_PI);

  for(i=0;i<100;i++) {
    ppd = ppd_max * (i/100.0);

    printf("%d %g %g %g %g\n",
        i, ppd,
        1.0/schro_encoder_perceptual_weight_constant (ppd),
        1.0/schro_encoder_perceptual_weight_moo (ppd),
        1.0/schro_encoder_perceptual_weight_ccir959 (ppd));
  }

  return 0;
}

