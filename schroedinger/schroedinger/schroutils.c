
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>


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

unsigned int
round_up_pow2(unsigned int x, int p)
{
  int y = (1<<p) - 1;
  x += y;
  x &= ~y;
  return x;
}

