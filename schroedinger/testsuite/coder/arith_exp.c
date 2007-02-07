
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>

#include "arith.h"

#if 0
static uint16_t division_factor[257] = {
      0,     0, 32768, 21845,
  16384, 13107, 10923,  9362,
   8192,  7282,  6554,  5958,
   5461,  5041,  4681,  4369,
   4096,  3855,  3641,  3449,
   3277,  3121,  2979,  2849,
   2731,  2621,  2521,  2427,
   2341,  2260,  2185,  2114,
   2048,  1986,  1928,  1872,
   1820,  1771,  1725,  1680,
   1638,  1598,  1560,  1524,
   1489,  1456,  1425,  1394,
   1365,  1337,  1311,  1285,
   1260,  1237,  1214,  1192,
   1170,  1150,  1130,  1111,
   1092,  1074,  1057,  1040,
   1024,  1008,   993,   978,
    964,   950,   936,   923,
    910,   898,   886,   874,
    862,   851,   840,   830,
    819,   809,   799,   790,
    780,   771,   762,   753,
    745,   736,   728,   720,
    712,   705,   697,   690,
    683,   676,   669,   662,
    655,   649,   643,   636,
    630,   624,   618,   612,
    607,   601,   596,   590,
    585,   580,   575,   570,
    565,   560,   555,   551,
    546,   542,   537,   533,
    529,   524,   520,   516,
    512,   508,   504,   500,
    496,   493,   489,   485,
    482,   478,   475,   471,
    468,   465,   462,   458,
    455,   452,   449,   446,
    443,   440,   437,   434,
    431,   428,   426,   423,
    420,   417,   415,   412,
    410,   407,   405,   402,
    400,   397,   395,   392,
    390,   388,   386,   383,
    381,   379,   377,   374,
    372,   370,   368,   366,
    364,   362,   360,   358,
    356,   354,   352,   350,
    349,   347,   345,   343,
    341,   340,   338,   336,
    334,   333,   331,   329,
    328,   326,   324,   323,
    321,   320,   318,   317,
    315,   314,   312,   311,
    309,   308,   306,   305,
    303,   302,   301,   299,
    298,   297,   295,   294,
    293,   291,   290,   289,
    287,   286,   285,   284,
    282,   281,   280,   279,
    278,   277,   275,   274,
    273,   272,   271,   270,
    269,   267,   266,   265,
    264,   263,   262,   261,
    260,   259,   258,   257,
    256
};
#endif


void
arith_exp_init (Arith *arith)
{
  memset (arith, 0, sizeof(*arith));

  arith->code = 0;
  arith->range0 = 0;
  arith->range1 = 0x10000;
  arith->cntr = 0;
  arith->offset = 0;
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

#if 0
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
#endif

static void
arith_exp_encode (Arith *arith, int i, int value)
{
  //unsigned int range;
  //unsigned int scaler;
  //unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;

  //weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  //scaler = division_factor[weight];
  //probability0 = arith->contexts[i].count[0] * scaler;
  probability0 = arith->contexts[i].probability;
  //range = arith->range1 - arith->range0 + 1;
  range_x_prob = (arith->range1 * probability0) >> 16;

  if (value) {
    arith->range0 = arith->range0 + range_x_prob;
    arith->range1 -= range_x_prob;
  } else {
    arith->range1 = range_x_prob;
  }
#if 0
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }
#endif
  arith->contexts[i].n++;
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].n == 32) {
    arith->contexts[i].n = 0;
    arith->contexts[i].probability += arith->contexts[i].count[0] * 0x07ff + 16;
    arith->contexts[i].probability >>= 1;
    //SCHRO_ASSERT (arith->contexts[i].probability <= 0x10000);
    //SCHRO_ASSERT (arith->contexts[i].probability > 0);
    arith->contexts[i].count[0] = 0;
  }

  while (arith->range1 < 0x8000) {
    arith->range0 <<= 1;
    arith->range1 <<= 1;

    arith->cntr++;
    if (arith->cntr == 8) {
      //int range;

      arith->data[arith->offset] = arith->range0 >> 16;
      arith->offset++;
      arith->range0 &= 0xffff;
      arith->cntr = 0;
    }
  }
#if 0
  do {
    if ((arith->range1 & (1<<15)) == (arith->range0 & (1<<15))) {
      int value;

      value = arith->range0 >> 15;
      push_bit (arith, value);
      while(arith->cntr) {
        push_bit (arith, !value);
        arith->cntr--;
      }

      arith->range0 <<= 1;
      arith->range0 &= 0xffff;
      arith->range1 <<= 1;
      arith->range1 &= 0xffff;
      arith->range1++;
    } else if ((arith->range0 & (1<<14)) && !(arith->range1 & (1<<14))) {
      arith->range0 ^= (1<<14);
      arith->range1 ^= (1<<14);

      arith->range0 <<= 1;
      arith->range0 &= 0xffff;
      arith->range1 <<= 1;
      arith->range1 &= 0xffff;
      arith->range1++;
      arith->cntr++;
    } else {
      break;
    }

  } while (1);
#endif
}


DEFINE_EFFICIENCY(exp)
DEFINE_SPEED(exp)

