
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>

#define BUFFER_SIZE 10000

int ilog2(unsigned int x);

int
main (int argc, char *argv[])
{
  int i;

  //schro_init();

  for(i=0;i<16;i++){
    printf("%d %d\n", i, ilog2(1<<i));
  }

  return 0;
}

