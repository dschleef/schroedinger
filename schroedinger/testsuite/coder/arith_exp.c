
#include "config.h"

#include <stdio.h>
#include <string.h>


#define OIL_ENABLE_UNSTABLE_API
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include "arith.h"



const unsigned int lut[256] = {
  //LUT corresponds to window = 16 @ p0=0.5 & 256 @ p=1.0
     0,    2,    5,    8,   11,   15,   20,   24,
    29,   35,   41,   47,   53,   60,   67,   74,
    82,   89,   97,  106,  114,  123,  132,  141,
   150,  160,  170,  180,  190,  201,  211,  222,
   233,  244,  256,  267,  279,  291,  303,  315,
   327,  340,  353,  366,  379,  392,  405,  419,
   433,  447,  461,  475,  489,  504,  518,  533,
   548,  563,  578,  593,  609,  624,  640,  656,
   672,  688,  705,  721,  738,  754,  771,  788,
   805,  822,  840,  857,  875,  892,  910,  928,
   946,  964,  983, 1001, 1020, 1038, 1057, 1076,
  1095, 1114, 1133, 1153, 1172, 1192, 1211, 1231,
  1251, 1271, 1291, 1311, 1332, 1352, 1373, 1393,
  1414, 1435, 1456, 1477, 1498, 1520, 1541, 1562,
  1584, 1606, 1628, 1649, 1671, 1694, 1716, 1738,
  1760, 1783, 1806, 1828, 1851, 1874, 1897, 1920,
  1935, 1942, 1949, 1955, 1961, 1968, 1974, 1980,
  1985, 1991, 1996, 2001, 2006, 2011, 2016, 2021,
  2025, 2029, 2033, 2037, 2040, 2044, 2047, 2050,
  2053, 2056, 2058, 2061, 2063, 2065, 2066, 2068,
  2069, 2070, 2071, 2072, 2072, 2072, 2072, 2072,
  2072, 2071, 2070, 2069, 2068, 2066, 2065, 2063,
  2060, 2058, 2055, 2052, 2049, 2045, 2042, 2038,
  2033, 2029, 2024, 2019, 2013, 2008, 2002, 1996,
  1989, 1982, 1975, 1968, 1960, 1952, 1943, 1934,
  1925, 1916, 1906, 1896, 1885, 1874, 1863, 1851,
  1839, 1827, 1814, 1800, 1786, 1772, 1757, 1742,
  1727, 1710, 1694, 1676, 1659, 1640, 1622, 1602,
  1582, 1561, 1540, 1518, 1495, 1471, 1447, 1422,
  1396, 1369, 1341, 1312, 1282, 1251, 1219, 1186,
  1151, 1114, 1077, 1037,  995,  952,  906,  857,
   805,  750,  690,  625,  553,  471,  376,  255
};

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
  arith->contexts[0].count[0] = 1;
  arith->contexts[0].count[1] = 1;
  arith->contexts[0].next = 0;
  arith->contexts[0].probability = 0x8000;
  arith->contexts[0].n = 0;
}

void
arith_exp_flush (Arith *arith)
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
arith_exp_encode (Arith *arith, int i, int value)
{
  unsigned int range;
  unsigned int probability0;
  unsigned int range_x_prob;
  
  probability0 = arith->contexts[i].probability;
  range = arith->range1;
  range_x_prob = (range * probability0) >> 16;
  
  if (value) {
    arith->range0 = arith->range0 + range_x_prob;
    arith->range1 -= range_x_prob;
  } else {
    arith->range1 = range_x_prob;
  }
  //arith->contexts[i].count[value]++;
  //arith->contexts[i].n++;
  if (value) {
    arith->contexts[i].probability -= lut[arith->contexts[i].probability>>8];
  } else {
    arith->contexts[i].probability += lut[255-(arith->contexts[i].probability>>8)];
  }
#if 0
  if (arith->contexts[i].n == 16) {
    unsigned int scaler;
    unsigned int weight;

    arith->contexts[i].n = 0;
    if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
      arith->contexts[i].count[0] >>= 1;
      arith->contexts[i].count[0]++;
      arith->contexts[i].count[1] >>= 1;
      arith->contexts[i].count[1]++;
    } 
    weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
    scaler = division_factor[weight];
    arith->contexts[i].probability = arith->contexts[i].count[0] * scaler;
  }
#endif
  
  while (arith->range1 < 0x8000) {

    arith->range0 <<= 1;
    arith->range1 <<= 1;

    arith->cntr++;
    if (arith->cntr == 8) {
      if (arith->range0 < (1<<24) &&
          (arith->range0 + arith->range1) >= (1<<24)) {
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
      arith->range0 &= 0xffff;
      arith->cntr = 0;
    }
  }
}

static int
arith_exp_decode (Arith *arith, int i)
{
  unsigned int count;
  unsigned int range;
  unsigned int probability0;
  unsigned int range_x_prob;
  int value;
  
//printf("[%04x %04x] %04x\n", arith->range0, arith->range1, arith->code);
  probability0 = arith->contexts[i].probability;
  count = arith->code - arith->range0 + 1;
  range = arith->range1;
  range_x_prob = (range * probability0) >> 16;
  
  value = count > range_x_prob;
  if (value) {
    arith->range0 = arith->range0 + range_x_prob;
    arith->range1 -= range_x_prob;
  } else {
    arith->range1 = range_x_prob;
  }
  //arith->contexts[i].count[value]++;
  //arith->contexts[i].n++;
  if (value) {
    arith->contexts[i].probability -= lut[arith->contexts[i].probability>>8];
  } else {
    arith->contexts[i].probability += lut[255-(arith->contexts[i].probability>>8)];
  }
#if 0
  if (arith->contexts[i].n == 16) {
    unsigned int scaler;
    unsigned int weight;

    arith->contexts[i].n = 0;
    if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
      arith->contexts[i].count[0] >>= 1;
      arith->contexts[i].count[0]++;
      arith->contexts[i].count[1] >>= 1;
      arith->contexts[i].count[1]++;
    } 
    weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
    scaler = division_factor[weight];
    arith->contexts[i].probability = arith->contexts[i].count[0] * scaler;
  }
#endif
  
  while (arith->range1 < 0x8000) {

    arith->range0 <<= 1;
    arith->range1 <<= 1;

    arith->code <<= 1;
    arith->code |= (arith->data[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;
    if (arith->cntr == 8) {
      arith->offset++;
      arith->range0 &= 0xffff;
      arith->code &= 0xffff;
      if (arith->code < arith->range0) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }
  return value;
}



DEFINE_EFFICIENCY(exp)
DEFINE_SPEED(exp)
DEFINE_ENCODE(exp)
DEFINE_DECODE(exp)


