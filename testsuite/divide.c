
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  for(i=-10;i<10;i++){
    printf("%d: %d\n", i, schro_divide(i,3));
  }

  exit(0);
}



