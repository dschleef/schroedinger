
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>


int
ilog2(unsigned int x)
{
  int shift=0;

  x>>=1;
  while(x) {
    x>>=1;
    shift++;
  }

  return shift;
}

