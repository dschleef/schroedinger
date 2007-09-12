
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <unistd.h>
#include <fcntl.h>

#include <liboil/liboilprofile.h>
#include <liboil/liboilrandom.h>

#define BUFFER_SIZE 1000000

int debug=1;

static int test_arith_decode_bit (SchroArith *arith, int i);

int
decode(SchroBuffer *buffer, int n, OilProfile *prof, int type)
{
  SchroArith *a;
  int i;
  int j;
  int x = 0;

  oil_profile_init (prof);
  for(j=0;j<10;j++){
    a = schro_arith_new();

    schro_arith_decode_init (a, buffer);

    switch (type) {
      case 0:
        oil_profile_start (prof);
        for(i=0;i<n;i++){
          x += schro_arith_decode_bit (a, 0);
        }
        oil_profile_stop (prof);
        break;
      case 1:
        oil_profile_start (prof);
        for(i=0;i<n;i++){
          x += test_arith_decode_bit (a, 0);
        }
        oil_profile_stop (prof);
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
    bit = oil_rand_u8() < freq;
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
  oil_profile_get_ave_std (&prof, &ave, &std);
  printf("ref  %d,%d: %g (%g) %d\n", n, freq, ave, std, x);

  y = decode(buffer, n, &prof, 1);
  oil_profile_get_ave_std (&prof, &ave, &std);
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

  while(1) check(1000, 128);
  check(1000, 128);
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


static int
test_arith_decode_bit (SchroArith *arith, int i)
{
  unsigned int range_x_prob;
  int value;
  int lut_index;

  range_x_prob = (arith->range[1] * arith->contexts[i].probability) >> 16;
  lut_index = arith->contexts[i].probability>>8;

  value = (arith->code - arith->range[0] >= range_x_prob);
  arith->contexts[i].probability += arith->lut[(value<<8) | lut_index];
  if (value) {
    arith->range[0] += range_x_prob;
    arith->range[1] -= range_x_prob;
  } else {
    arith->range[1] = range_x_prob;
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


