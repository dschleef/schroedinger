
#include <stdio.h>

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>


int
main (int argc, char *argv[])
{
  SchroObmc obmc;
  SchroObmcRegion *region;
  int i;
  int j;
  int k;

  schro_init();

  schro_obmc_init(&obmc, 12, 12, 8, 8);

  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        printf(" %2d", region->weights[j*12+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  schro_obmc_init(&obmc, 6, 6, 4, 4);

  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        printf(" %2d", region->weights[j*6+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return 0;

}



