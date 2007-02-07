 
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>

#include "arith.h"

void
arith_exp_init (Arith *arith)
{
  memset (arith, 0, sizeof(*arith));

  arith->code = 0;
  arith->range0 = 0;
  arith->range1 = 0x10000;
  arith->cntr = 0;
  arith->offset = 0;
  arith->contexts[0].mps = 0;
  arith->contexts[0].count[0] = 0;
  arith->contexts[0].count[1] = 0;
  arith->contexts[0].next = 0;
  arith->contexts[0].probability = 0x8000;
  arith->contexts[0].n = 0;
}

void
arith_exp_flush (Arith *arith)
{
}

#define SHIFT \
    arith->range0 <<= 1; \
    arith->range1 <<= 1; \
    arith->cntr++; \
    if (arith->cntr == 8) { \
      arith->data[arith->offset] = arith->range0 >> 16; \
      arith->offset++; \
      arith->range0 &= 0xffff; \
      arith->cntr = 0; \
    }

static void
arith_exp_encode (Arith *arith, int i, int value)
{
  unsigned int range_x_prob;
  unsigned int probability;

  /* note: must be unsigned multiplication */
  probability = arith->contexts[i].probability;
  range_x_prob = (arith->range1 * probability) >> 16;

  value ^= arith->contexts[i].mps;
  if (value) {
    arith->range0 = arith->range0 + range_x_prob;
    arith->range1 -= range_x_prob;
    SHIFT;
    while (arith->range1 < 0x8000) {
      SHIFT;
    }
  } else {
    arith->range1 = range_x_prob;
    if (arith->range1 < 0x8000) {
      SHIFT;
    }
  }
  arith->contexts[i].n++;
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].n == 32) {
    arith->contexts[i].n = 0;
    arith->contexts[i].probability +=
      arith->contexts[i].count[0] * 0x07ff + 16;
    arith->contexts[i].probability >>= 1;
    if (arith->contexts[i].probability < 0x8000) {
      arith->contexts[i].probability = 0x10000 - arith->contexts[i].probability;
      arith->contexts[i].mps ^= 1;
    }
    //SCHRO_ASSERT(arith->contexts[i].probability > 0);
    arith->contexts[i].count[0] = 0;
    arith->contexts[i].count[1] = 0;
  }
}


DEFINE_EFFICIENCY(exp)
DEFINE_SPEED(exp)

