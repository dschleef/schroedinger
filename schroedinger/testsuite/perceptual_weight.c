
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
  int j;
  int n;
  int i;
  SchroEncoder *encoder;
  double min;
  double x;

  schro_init();

  encoder = schro_encoder_new ();

  for(filter=0;filter<8;filter++){
    for(j=0;j<4;j++){
      printf("filter %d n_levels %d\n", filter, j+1);
      n = 3*(j+1)+1;

      min = encoder->subband_weights[filter][j][0];
      for(i=0;i<n;i++){
        if (encoder->subband_weights[filter][j][i] < min) {
          min = encoder->subband_weights[filter][j][i];
        }
      }
      for(i=0;i<n;i++){
        x = encoder->subband_weights[filter][j][i]/min;
        printf("%2d %6.3f %6.3f %d\n", i, encoder->subband_weights[filter][j][i],
            x, quant_index(x));
      }
    }
  }

  return 0;
}

