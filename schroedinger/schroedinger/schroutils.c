
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schro-stdint.h>


int
muldiv64 (int a, int b, int c)
{
  int64_t x;

  x = a;
  x *= b;
  x /= c;

  return (int)x;
}

