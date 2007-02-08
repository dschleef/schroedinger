
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include "arith.h"

void
arith_dirac_byte_init (Arith *arith)
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
arith_dirac_byte_flush (Arith *arith)
{
  while (arith->cntr < 8) {
    arith->range0 <<= 1;
    arith->cntr++;
  }

  if (arith->range0 >= (1<<24)) {
    arith->data[arith->offset-1]++;
    while (arith->carry) {
      arith->data[arith->offset] = 0x00;
      arith->carry--;
      arith->offset++;
    }
  } else {
    while (arith->carry) {
      arith->data[arith->offset] = 0xff;
      arith->carry--;
      arith->offset++;
    }
  }
  arith->data[arith->offset] = arith->range0 >> 16;
  arith->offset++;

  arith->data[arith->offset] = arith->range0 >> 8;
  arith->offset++;

  arith->data[arith->offset] = arith->range0 >> 0;
  arith->offset++;
}

static void
arith_dirac_byte_encode (Arith *arith, int i, int value)
{
  unsigned int count;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;
  
printf("[%04x %04x]\n", arith->range0, arith->range1);
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
  
  while (arith->range1 - arith->range0 < 0x8000) {

    arith->range0 <<= 1;
    arith->range1 <<= 1;
    arith->range1++;

    arith->cntr++;
    if (arith->cntr == 8) {
      int range;

printf("byte shift\n");
      if (arith->range0 < (1<<24) && arith->range1 >= (1<<24)) {
        arith->carry++;
      } else {
        if (arith->range0 >= (1<<24)) {
          arith->data[arith->offset-1]++;
          while (arith->carry) {
            arith->data[arith->offset] = 0x00;
            arith->carry--;
            arith->offset++;
          }
        } else {
          while (arith->carry) {
            arith->data[arith->offset] = 0xff;
            arith->carry--;
            arith->offset++;
          }
        }
        arith->data[arith->offset] = arith->range0 >> 16;
        arith->offset++;
      }
      range = arith->range1 - arith->range0;
      arith->range0 &= 0xffff;
      arith->range1 = arith->range0 + range;
      arith->cntr = 0;
    }
  }
}

static int
arith_dirac_byte_decode (Arith *arith, int i)
{
  unsigned int count;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;
  int value;
  
printf("[%04x %04x] %04x\n", arith->range0, arith->range1, arith->code);
  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = schro_table_division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  count = arith->code - arith->range0 + 1;
  range = arith->range1 - arith->range0 + 1;
  range_x_prob = (range * probability0) >> 16;
  
  value = count > range_x_prob;
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
  
  while (arith->range1 - arith->range0 < 0x8000) {

    arith->range0 <<= 1;
    arith->range1 <<= 1;
    arith->range1++;

    arith->code <<= 1;
    arith->code |= (arith->data[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;
    if (arith->cntr == 8) {
      int range;

printf("byte shift\n");
      arith->offset++;
#if 0
      if (arith->range0 < (1<<24) && arith->range1 >= (1<<24)) {
        arith->carry++;
      } else {
        if (arith->range0 >= (1<<24)) {
          arith->data[arith->offset-1]++;
          while (arith->carry) {
            arith->data[arith->offset] = 0x00;
            arith->carry--;
            arith->offset++;
          }
        } else {
          while (arith->carry) {
            arith->data[arith->offset] = 0xff;
            arith->carry--;
            arith->offset++;
          }
        }
        arith->data[arith->offset] = arith->range0 >> 16;
        arith->offset++;
      }
#endif
      range = arith->range1 - arith->range0;
      arith->range0 &= 0xffff;
      arith->range1 = arith->range0 + range;
      arith->code &= 0xffff;
      if (arith->code < arith->range0) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }
  return value;
}



DEFINE_EFFICIENCY(dirac_byte)
DEFINE_SPEED(dirac_byte)
DEFINE_ENCODE(dirac_byte)
DEFINE_DECODE(dirac_byte)

