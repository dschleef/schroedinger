
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <unistd.h>
#include <fcntl.h>

#include <orc-test/orcprofile.h>
#include <orc-test/orcrandom.h>

#define BUFFER_SIZE 1000000

int debug=1;

#define static
static int orig_arith_decode_bit (SchroArith *arith, int i);
static int ref_arith_decode_bit (SchroArith *arith, int i);
static int test_arith_decode_bit (SchroArith *arith, int i);

int
decode(SchroBuffer *buffer, int n, OilProfile *prof, int type)
{
  SchroArith *a;
  int i;
  int j;
  int x = 0;

  orc_profile_init (prof);
  for(j=0;j<10;j++){
    a = schro_arith_new();

    schro_arith_decode_init (a, buffer);

    switch (type) {
      case 0:
        orc_profile_start (prof);
        for(i=0;i<n;i++){
          x += orig_arith_decode_bit (a, 0);
        }
        orc_profile_stop (prof);
        break;
      case 1:
        orc_profile_start (prof);
        for(i=0;i<n;i++){
          x += ref_arith_decode_bit (a, 0);
        }
        orc_profile_stop (prof);
        break;
      case 2:
        orc_profile_start (prof);
        for(i=0;i<n;i++){
          x += test_arith_decode_bit (a, 0);
        }
        orc_profile_stop (prof);
        break;
    }

    a->buffer = NULL;
    schro_arith_free(a);
  }

  return x;
}

void
print_speed (void)
{
  int fd;
  char buffer[100];
  int n;
  int freq0, freq1;

  fd = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", O_RDONLY);
  n = read (fd, buffer, 100);
  close(fd);
  if (n >= 0) {
    buffer[n] = 0;
    freq0 = strtol (buffer, NULL, 0);
  } else {
    freq0 = -1;
  }

  fd = open("/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", O_RDONLY);
  n = read (fd, buffer, 100);
  close(fd);
  if (n >= 0) {
    buffer[n] = 0;
    freq1 = strtol (buffer, NULL, 0);
  } else {
    freq1 = -1;
  }

  printf("cpu speed %d %d\n", freq0, freq1);
}

void
dump_bits (SchroBuffer *buffer, int n)
{
  int i;

  for(i=0;i<n;i++){
    printf("%02x ", buffer->data[i]);
    if ((i&15)==15) {
      printf ("\n");
    }
  }
  printf ("\n");
}

void
encode (SchroBuffer *buffer, int n, int freq)
{
  SchroArith *a;
  int i;
  int bit;

  a = schro_arith_new();

  schro_arith_encode_init (a, buffer);

  for(i=0;i<n;i++){
    bit = ((orc_random(rand_context)>>8)&0xff) < freq;
    schro_arith_encode_bit (a, 0, bit);
  }
  schro_arith_flush (a);

  a->buffer = NULL;
  schro_arith_free(a);
}

int
check (int n, int freq)
{
  SchroBuffer *buffer;
  OilProfile prof;
  double ave, std;
  int x;
  int y;

  buffer = schro_buffer_new_and_alloc (100000);

  encode(buffer, n, freq);

  print_speed();

  x = decode(buffer, n, &prof, 0);
  orc_profile_get_ave_std (&prof, &ave, &std);
  printf("orig %d,%d: %g (%g) %d\n", n, freq, ave, std, x);

  x = decode(buffer, n, &prof, 1);
  orc_profile_get_ave_std (&prof, &ave, &std);
  printf("ref  %d,%d: %g (%g) %d\n", n, freq, ave, std, x);

  y = decode(buffer, n, &prof, 2);
  orc_profile_get_ave_std (&prof, &ave, &std);
  printf("test %d,%d: %g (%g) %d\n", n, freq, ave, std, y);
  if (x != y) {
    printf("BROKEN\n");
  }

  schro_buffer_unref (buffer);

  return 0;
}

int
main (int argc, char *argv[])
{
  //int i;

  schro_init();

  //while(1) check(1000, 128);
  check(100, 128);
#if 0
  for(i=100;i<=1000;i+=100) {
    //check(i, 128);
    check(i, 256);
  }
  check(2000, 256);
  check(3000, 256);
  check(4000, 256);
  check(5000, 256);
  check(100000, 256);
#endif
#if 0
  for(i=0;i<=256;i+=16) {
    check(100, i);
  }
#endif

  return 0;
}

static const uint16_t _lut[256] = {
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

static int
orig_arith_decode_bit (SchroArith *arith, int i)
{
  unsigned int probability0;
  unsigned int range_x_prob;
  unsigned int count;
  int value;
  
  probability0 = arith->probabilities[i];
  count = arith->code - arith->range[0];
  range_x_prob = ((arith->range[1]) * probability0) >> 16;
  
  value = (count >= range_x_prob);
  if (value) {
    arith->range[0] += range_x_prob;
    arith->range[1] -= range_x_prob;
    arith->probabilities[i] -= _lut[arith->probabilities[i]>>8];
  } else {
    arith->range[1] = range_x_prob;
    arith->probabilities[i] += _lut[255-(arith->probabilities[i]>>8)];
  }
    
  while (arith->range[1] <= 0x4000) {
    arith->range[0] <<= 1;
    arith->range[1] <<= 1;
    
    arith->code <<= 1;
    arith->code |= (arith->dataptr[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;

    if (arith->cntr == 8) {
      arith->offset++;
      arith->range[0] &= 0xffff;
      arith->code &= 0xffff;

      if (arith->code < arith->range[0]) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }

  return value;
}


static int
ref_arith_decode_bit (SchroArith *arith, int i)
{
  unsigned int range_x_prob;
  int value;
  int lut_index;

  range_x_prob = ((arith->range[1] - arith->range[0]) * arith->probabilities[i]) >> 16;
  lut_index = arith->probabilities[i]>>8;

  value = (arith->code - arith->range[0] >= range_x_prob);
  arith->probabilities[i] += arith->lut[(value<<8) | lut_index];
  arith->range[1-value] = arith->range[0] + range_x_prob;

  while (arith->range[1] - arith->range[0] <= 0x4000) {
    arith->range[0] <<= 1;
    arith->range[1] <<= 1;

    arith->code <<= 1;
    arith->code |= (arith->dataptr[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;

    if (arith->cntr == 8) {
      int size = arith->range[1] - arith->range[0];
      arith->offset++;
      arith->range[0] &= 0xffff;
      arith->range[1] = arith->range[0] + size;
      arith->code &= 0xffff;

      if (arith->code < arith->range[0]) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }

  return value;
}


static int
test_arith_decode_bit (SchroArith *arith, int i)
{
  //unsigned int range_x_prob;
  int value;
  //int lut_index;

#if 0
  printf("[%d,%d] code %d prob %d\n",
      arith->range[0], arith->range[1], arith->code, arith->probabilities[i]);
#endif
#if 1
  __asm__ __volatile__ (
    ".set range0, 0\n"
    ".set range1, 4\n"
    ".set code, 8\n"
    ".set range_size, 12\n"
    ".set probabilities, 16\n"
    ".set lut, 0x98\n"

    // range_x_prob
    "  mov range1(%[arith]), %%edx\n"
    "  sub range0(%[arith]), %%edx\n"
#if 0
    "  movzwl probabilities(%[arith],%[i],2), %%eax\n"
    "  imul %%eax, %%edx\n"
    "  shr $0x10, %%edx\n"
#else
    "  movw probabilities(%[arith],%[i],2), %%ax\n"
    "  cmp $0x10000, %%edx\n"
    "  je fullrange\n"
    "  mul %%dx\n"
    // result in %%dx
    "  movzwl %%dx, %%eax\n"
    "fullrange:\n"
    "  mov %%eax, %%edx\n"
#endif

    // lut_index
#if 0
    "  movzbl probabilities+1(%[arith],%[i],2), %%eax\n"
#else
    "  movzwl probabilities(%[arith],%[i],2), %%eax\n"
    "  movzbl %%ah, %%eax\n"
#endif

    // value
    "  mov code(%[arith]), %%ebx\n"
    "  sub range0(%[arith]), %%ebx\n"
#if 0
    "  mov %%edx, %%ecx\n"
    "  subl %%ebx, %%ecx\n"
    "  subl $1, %%ecx\n"
    "  shr $31, %%ecx\n"
#else
    "  cmp %%ebx, %%edx\n"
    "  setbe %%cl\n"
    "  movzbl %%cl, %%ecx\n"
#endif
    "  mov %%ecx, %[value]\n"

    // probabilities[i]
    "  shl $8, %%ecx\n"
    "  or %%ecx, %%eax\n"
    "  mov lut(%[arith], %%eax, 2), %%bx\n"
    "  addw %%bx, probabilities(%[arith],%[i],2)\n"

    // range[1^value]
    "  mov %[value], %%eax\n"
    "  xor $1, %%eax\n"
    "  add range0(%[arith]), %%edx\n"
    "  mov %%edx, range0(%[arith],%%eax,4)\n"
    : [value] "=m" (value),
      [arith] "+r" (arith),
      [i] "+r" (i)
    :
    : "edx", "ebx", "eax", "ecx"
    );
#endif
#if 0
  printf("[%d,%d] code %d prob %d, value %d\n",
      arith->range[0], arith->range[1], arith->code, arith->probabilities[i],
      value);
#endif

#if 0
  range_x_prob = ((arith->range[1] - arith->range[0]) * arith->probabilities[i]) >> 16;
  lut_index = arith->probabilities[i]>>8;

  value = (arith->code - arith->range[0] >= range_x_prob);
  arith->probabilities[i] += arith->lut[(value<<8) | lut_index];
  arith->range[1^value] = arith->range[0] + range_x_prob;
#endif

  while (arith->range[1] - arith->range[0] <= 0x4000) {
    arith->range[0] <<= 1;
    arith->range[1] <<= 1;

    arith->code <<= 1;
    arith->code |= (arith->dataptr[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;

    if (arith->cntr == 8) {
      int size = arith->range[1] - arith->range[0];
      arith->offset++;
      arith->range[0] &= 0xffff;
      arith->range[1] = arith->range[0] + size;
      arith->code &= 0xffff;

      if (arith->code < arith->range[0]) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }

  return value;
}

