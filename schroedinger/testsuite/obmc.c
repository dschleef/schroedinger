
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
  //schro_obmc_init(&obmc, 12, 12, 8, 8);

  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=0;j<region->end_y-region->start_y;j++){
      for(i=0;i<region->end_x-region->start_x;i++){
        printf(" %3d", region->weights[j*obmc.x_len+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

#if 0
  schro_obmc_init(&obmc, 6, 6, 4, 4);

  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=0;j<region->end_y-region->start_y;j++){
      for(i=0;i<region->end_x-region->start_x;i++){
        printf(" %2d", region->weights[j*6+i]);
      }
      printf("\n");
    }
    printf("\n");
  }
#endif

  return 0;

}



