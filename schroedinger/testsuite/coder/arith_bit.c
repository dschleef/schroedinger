
#include "config.h"

#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include "arith.h"

void
arith_bit_init (Arith *arith)
{
  memset (arith, 0, sizeof(*arith));

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
arith_bit_flush (Arith *arith)
{
}

static void
push_bit (Arith *arith, int value)
{
  arith->output_byte <<= 1;
  arith->output_byte |= value;
  arith->output_bits++;
  if (arith->output_bits == 8) {
    arith->data[arith->offset] = arith->output_byte;
    arith->offset++;
    arith->output_byte = 0;
    arith->output_bits = 0;
  }
}

static void
arith_bit_encode (Arith *arith, int i, int value)
{
  push_bit(arith, value);
}

DEFINE_EFFICIENCY(bit)
DEFINE_SPEED(bit)

