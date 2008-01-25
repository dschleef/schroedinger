
#include <stdio.h>

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>

void print_blocks(void);
void print_regions(void);

int
main (int argc, char *argv[])
{
  schro_init();

  //print_blocks ();
  print_regions ();

  return 0;
}

void
print_regions (void)
{
  SchroObmc obmc;
  SchroObmcRegion *region;
  int i;
  int j;
  int k;

  //schro_obmc_init(&obmc, 12, 6, 8, 4, 1, 1, 1);
  //schro_obmc_init(&obmc, 8, 8, 8, 8, 1, 1, 1);
  //schro_obmc_init(&obmc, 16, 16, 8, 8, 1, 1, 1);
  //schro_obmc_init(&obmc, 12, 12, 8, 8, 1, 1, 1);
  //schro_obmc_init(&obmc, 6, 6, 4, 4, 1, 1, 1);
  //schro_obmc_init(&obmc, 16, 16, 12, 12, 1, 1, 1);
  schro_obmc_init(&obmc, 8, 8, 6, 6, 1, 1, 1);

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
        printf(" %3d", region->weights[2][j*obmc.x_len+i]);
      }
      printf("\n");
    }
    printf("\n");
  }
}


void
print_block (int sep, int len)
{
  SchroObmc obmc;
  SchroObmcRegion *region;
  int i;
  int j;

  printf("%dx%d,%dx%d\n", sep, sep, len, len);

  schro_obmc_init(&obmc, len, len, sep, sep, 1, 1, 1);

  region = obmc.regions + 4;
  for(j=region->start_y;j<region->end_y;j++){
    for(i=region->start_x;i<region->end_x;i++){
      printf(" %3d", region->weights[1][j*obmc.x_len+i]);
    }
    printf("\n");
  }
  printf("\n");
}

void
print_blocks (void)
{
  print_block (4, 4);
  print_block (4, 6);
  print_block (4, 8);
  print_block (8, 8);
  print_block (8, 12);
  print_block (8, 16);
  print_block (12, 12);
  print_block (12, 16);
  print_block (12, 24);
  print_block (16, 16);
  print_block (16, 24);
  print_block (16, 32);
}

