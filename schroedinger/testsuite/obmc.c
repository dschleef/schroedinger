
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

  //schro_obmc_init(&obmc, 12, 12, 8, 8, 1, 1, 1);
  //schro_obmc_init(&obmc, 8, 8, 8, 8, 1, 1);
  schro_obmc_init(&obmc, 16, 16, 8, 8, 1, 1, 1);

  printf("shift %d\n\n", obmc.shift);

  printf("single ref:\n");
  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        printf(" %3d", region->weights[0][j*obmc.x_len+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("ref1:\n");
  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        printf(" %3d", region->weights[1][j*obmc.x_len+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  printf("ref2:\n");
  for(k=0;k<9;k++){
    region = obmc.regions + k;
    printf("%d:\n", k);
    for(j=region->start_y;j<region->end_y;j++){
      for(i=region->start_x;i<region->end_x;i++){
        printf(" %3d", region->weights[0][j*obmc.x_len+i]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return 0;

}



