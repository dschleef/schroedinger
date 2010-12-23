
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>

#include <schroedinger/schrotables.c>

int
main (int argc, char *argv[])
{
  int i;
  double x;
  int j,k;

  schro_init();

  for(i=0;i<100;i++){
    x = i*1;
    j = schro_utils_multiplier_to_quant_index (x);
    k = schro_table_quant[j];
    printf("%g %d %d\n", x, j, k);
  }

  exit(0);
}



