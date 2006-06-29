
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>

#include <liboil/liboilprofile.h>
#include <liboil/liboilrandom.h>

#define BUFFER_SIZE 10000

int debug=1;

void
decode(SchroBuffer *buffer, int n, OilProfile *prof)
{
  SchroArith *a;
  int i;
  int j;

  oil_profile_init (prof);
  for(j=0;j<10;j++){
    a = schro_arith_new();

    schro_arith_decode_init (a, buffer);
    schro_arith_context_init (a, 0, 1, 1);

    oil_profile_start (prof);
    for(i=0;i<n;i++){
      schro_arith_context_decode_bit (a, 0);
    }
    oil_profile_stop (prof);

    schro_arith_free(a);
  }
}

void
encode (SchroBuffer *buffer, int n, int freq)
{
  SchroArith *a;
  int i;
  int bit;

  a = schro_arith_new();

  schro_arith_encode_init (a, buffer);
  schro_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n;i++){
    bit = oil_rand_u8() < freq;
    schro_arith_context_encode_bit (a, 0, bit);
  }
  schro_arith_flush (a);

  schro_arith_free(a);
}

int
check (int n, int freq)
{
  SchroBuffer *buffer;
  OilProfile prof;
  double ave, std;

  buffer = schro_buffer_new_and_alloc (1000);

  encode(buffer, n, freq);
  decode(buffer, n, &prof);

  oil_profile_get_ave_std (&prof, &ave, &std);

  printf("%d,%d: %g (%g)\n", n, freq, ave, std);

  return 0;
}

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  for(i=100;i<=1000;i+=100) {
    check(i, 128);
  }
  for(i=0;i<=256;i+=16) {
    check(100, i);
  }

  return 0;
}

