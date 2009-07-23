
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schroorc.h>

#include <orc-test/orcprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 128

extern uint8_t packed_data[];

int16_t a[N];
int16_t b[N];
int16_t c[N];

int16_t table[65536];

int orc_profile_get_min (OrcProfile *prof)
{
  int i;
  int min;
  min = prof->hist_time[0];
  for(i=0;i<10;i++){
    if (prof->hist_count[i] > 0) {
      if (prof->hist_time[i] < min) {
        min = prof->hist_time[i];
      }
    }
  }
  return min;
}


#define SHIFT 8
static void
_schro_unpack_shift_in (SchroUnpack *unpack)
{
  if (unpack->n_bits_left >= 16) {
    unsigned int val;

    val = (unpack->data[0]<<8) | unpack->data[1];
    unpack->shift_register |=
      val<<(16 - unpack->n_bits_in_shift_register);

    unpack->data+=2;
    unpack->n_bits_left -= 16;
    unpack->n_bits_in_shift_register += 16;
    return;
  }

#if 0
  if (unpack->n_bits_left >= 32) {
    /* the fast path */
    if (unpack->n_bits_in_shift_register == 0) {
      unpack->shift_register = (unpack->data[0]<<24) | (unpack->data[1]<<16) |
        (unpack->data[2]<<8) | (unpack->data[3]);
      unpack->data += 4;
      unpack->n_bits_left -= 32;
      unpack->n_bits_in_shift_register = 32;
    } else {
      while (unpack->n_bits_in_shift_register <= 24) {
        unpack->shift_register |=
          unpack->data[0]<<(24 - unpack->n_bits_in_shift_register);
        unpack->data++;
        unpack->n_bits_left -= 8;
        unpack->n_bits_in_shift_register += 8;
      }
    }
    return;
  }
#endif

  if (unpack->n_bits_left == 0) {
    unsigned int value = (unpack->guard_bit)?0xffffffff:0;

    unpack->overrun += 32 - unpack->n_bits_in_shift_register;
    unpack->shift_register |= (value >> unpack->n_bits_in_shift_register);
    unpack->n_bits_in_shift_register = 32;
    return;
  }

  while (unpack->n_bits_left >= 8 && unpack->n_bits_in_shift_register <= 24) {
    unpack->shift_register |=
      unpack->data[0]<<(24 - unpack->n_bits_in_shift_register);
    unpack->data++;
    unpack->n_bits_left -= 8;
    unpack->n_bits_in_shift_register += 8;
  }

  if (unpack->n_bits_left > 0 &&
      unpack->n_bits_in_shift_register + unpack->n_bits_left <= 32) {
    unsigned int value;

    value = unpack->data[0]>>(8-unpack->n_bits_left);
    unpack->shift_register |=
      value<<(32 - unpack->n_bits_in_shift_register - unpack->n_bits_left);
    unpack->data++;
    unpack->n_bits_in_shift_register += unpack->n_bits_left;
    unpack->n_bits_left = 0;
  }
}

static void
_schro_unpack_shift_out_fast (SchroUnpack *unpack, int n)
{
  unpack->shift_register <<= n;
  unpack->n_bits_in_shift_register -= n;
  unpack->n_bits_read += n;
}


void
my_unpack_decode_sint_s16 (int16_t *dest, SchroUnpack *unpack, int n)
{
  int i;
  int m;
  int j;

  while (n > 0) {
    if (unpack->n_bits_in_shift_register < SHIFT) {
      _schro_unpack_shift_in (unpack);
    }
    i = unpack->shift_register >> (32-SHIFT);
    if (schro_table_unpack_sint[i][0] == 0) {
      dest[0] = schro_unpack_decode_sint_slow (unpack);
      dest++;
      n--;
    } else {
      m = MIN(n, schro_table_unpack_sint[i][0]);
      for(j=0;j<m;j++){
        dest[j] = schro_table_unpack_sint[i][1+2*j];
      }
      _schro_unpack_shift_out_fast (unpack, schro_table_unpack_sint[i][2*m]);
      dest += m;
      n -= m;
    }
  }
}




void
dequantise_speed (int n)
{
  OrcProfile prof1;
  OrcProfile prof2;
  OrcProfile prof3;
  double ave1;
  double ave2;
  double ave3;
  int i;

  orc_profile_init (&prof1);
  orc_profile_init (&prof2);
  orc_profile_init (&prof3);

  for(i=0;i<10;i++) {
    SchroUnpack unpack;
    SchroUnpack unpack2;
    int slice_y_length;

    schro_unpack_init_with_data (&unpack, packed_data, 67, 1);
    schro_unpack_decode_bits (&unpack, 7);

    slice_y_length = schro_unpack_decode_bits (&unpack, 10);
    schro_unpack_limit_bits_remaining (&unpack, slice_y_length);

    schro_unpack_copy (&unpack2, &unpack);

    orc_profile_start (&prof1);
    schro_unpack_decode_sint_s16 (a, &unpack, 128);
    orc_profile_stop (&prof1);

    orc_profile_start (&prof2);
    my_unpack_decode_sint_s16 (b, &unpack2, 128);
    orc_profile_stop (&prof2);
#if 0
    orc_profile_start (&prof3);
    schro_unpack_decode_sint_s16 (data, &unpack, n);
    orc_profile_stop (&prof3);
#endif
  }

  ave1 = orc_profile_get_min (&prof1);
  ave2 = orc_profile_get_min (&prof2);
  ave3 = orc_profile_get_min (&prof3);
  printf("%d %g %g %g\n", n, ave1, ave2, ave3);
}


int
main (int argc, char *argv[])
{
  int i;

  schro_init();
  orc_init();

  for(i=0;i<200;i++){
    dequantise_speed (i);
  }

  return 0;
}

uint8_t packed_data[] = {
  8,
  57,
  168,
  0,
  3,
  181,
  43,
  199,
  224,
  92,
  185,
  115,
  237,
  223,
  249,
  127,
  33,
  62,
  139,
  213,
  160,
  80,
  104,
  167,
  21,
  224,
  52,
  30,
  33,
  195,
  131,
  69,
  168,
  244,
  63,
  254,
  92,
  248,
  253,
  110,
  156,
  160,
  95,
  241,
  163,
  64,
  255,
  248,
  241,
  57,
  87,
  255,
  229,
  195,
  255,
  231,
  11,
  128,
  0,
  0,
  255,
  255,
  255,
  255,
  249,
  255,
  255,
};

