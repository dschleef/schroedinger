
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrofilter.h>
#include <schroedinger/schro.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>


uint8_t data[300];
uint8_t dest1[100];
uint8_t dest2[100];

void
test (void)
{
  int i;

  for(i=0;i<300;i++){
    data[i] = rand();
  }

  //schro_filter_cwm7 (dest1, data+0, data+100, data+200, 98);
  schro_filter_cwmN_ref (dest1, data+0, data+100, data+200, 98, 5);
  schro_filter_cwmN (dest2, data+0, data+100, data+200, 98, 5);

  for(i=0;i<98;i++){
    if (dest1[i] != dest2[i]) {
      printf ("%5d %5d %5d\n", data[i], data[i+1], data[i+2]);
      printf ("%5d %5d %5d\n", data[100+i], data[100+i+1], data[100+i+2]);
      printf ("%5d %5d %5d\n", data[200+i], data[200+i+1], data[200+i+2]);
      printf ("%d %d\n\n", dest1[i], dest2[i]);
    }
  }

}


int
main (int argc, char *argv[])
{
  int i;

  srand(time(NULL));
  for (i=0;i<100;i++){
    test();
  }

  return 0;
}

