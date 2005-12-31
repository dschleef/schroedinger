
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <string.h>
#include <carid/carid.h>

int16_t b1[128];
int16_t b2[128];

int
main (int argc, char *argv[])
{
  int i;

  for(i=0;i<128;i++) {
    b1[i] = 16;
  }

  memcpy(b2, b1, 128*2);

  //carid_wt (0, b2, 128);

  for(i=0;i<128;i++){
    printf("%d %d\n", i, b2[i]);
  }

  return 0;
}


