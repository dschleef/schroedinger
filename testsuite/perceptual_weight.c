
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"

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

      min = encoder->intra_subband_weights[filter][j][0];
      for(i=0;i<n;i++){
        if (encoder->intra_subband_weights[filter][j][i] < min) {
          min = encoder->intra_subband_weights[filter][j][i];
        }
      }
      for(i=0;i<n;i++){
        x = encoder->intra_subband_weights[filter][j][i]/min;
        printf("%2d %6.3f %6.3f %d\n", i, encoder->intra_subband_weights[filter][j][i],
            x, gain_to_quant_index(x));
      }
    }
  }
  for(filter=0;filter<8;filter++){
    for(j=0;j<4;j++){
      printf("filter %d n_levels %d\n", filter, j+1);
      n = 3*(j+1)+1;

      min = encoder->inter_subband_weights[filter][j][0];
      for(i=0;i<n;i++){
        if (encoder->inter_subband_weights[filter][j][i] < min) {
          min = encoder->inter_subband_weights[filter][j][i];
        }
      }
      for(i=0;i<n;i++){
        x = encoder->inter_subband_weights[filter][j][i]/min;
        printf("%2d %6.3f %6.3f %d\n", i, encoder->inter_subband_weights[filter][j][i],
            x, gain_to_quant_index(x));
      }
    }
  }


  return 0;
}

