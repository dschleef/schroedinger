
#include <stdio.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include "arith.h"

void
arith_exp_init (Arith *arith)
{

  arith->code = 0;
  arith->range0 = 0;
  arith->range1 = 0xffff;
  arith->cntr = 0;
  arith->offset = 0;
  arith->contexts[0].count[0] = 1;
  arith->contexts[0].count[1] = 1;
  arith->contexts[0].next = 0;
}

void
arith_exp_flush (Arith *arith)
{
}

static void
arith_exp_encode (Arith *arith, int i, int value)
{
  unsigned int count;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;
  
  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = schro_table_division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  count = arith->code - arith->range0 + 1;
  range = arith->range1 - arith->range0 + 1;
  range_x_prob = (range * probability0) >> 16;
  
  if (value) {
    arith->range0 = arith->range0 + range_x_prob;
  } else {
    arith->range1 = arith->range0 + range_x_prob - 1;
  }
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  } 
  
#if 0
  printf("range0 %08x range1 %08x (%d)\n", arith->range[0], arith->range[1],
      arith->range[1] - arith->range[0]);
#endif
  while (arith->range1 - arith->range0 < 0x8000) {

    arith->range0 <<= 1;
    arith->range1 <<= 1;
    arith->range1++;

    arith->cntr++;
#if 0
    printf("shift to range0 %08x range1 %08x (%d) %d\n",
        arith->range[0], arith->range[1], arith->range[1] - arith->range[0],
        arith->cntr);
#endif
    if (arith->cntr == 8) {
      int range;

      arith->data[arith->offset] = arith->range0 >> 16;
      arith->offset++;
      range = arith->range1 - arith->range0;
      arith->range0 &= 0xffff;
      arith->range1 = arith->range0 + range;
      arith->cntr = 0;
#if 0
    printf("push range0 %08x range1 %08x (%d) %d\n",
        arith->range[0], arith->range[1], arith->range[1] - arith->range[0],
        arith->cntr);
#endif
    }
  }
}

DEFINE_EFFICIENCY(exp)
DEFINE_SPEED(exp)

