
#include "config.h"

#include <stdio.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>

#include <math.h>

#include "arith.h"
#include "../common.h"

#define N 10000


int encode_arith_dirac (unsigned char *out_data, unsigned char *in_data, int n);
int encode_arith_dirac_byte (unsigned char *out_data, unsigned char *in_data, int n);
int encode_arith_dirac_both (unsigned char *out_data, unsigned char *in_data, int n);
int encode_arith_exp (unsigned char *out_data, unsigned char *in_data, int n);

void decode_arith_dirac_byte (unsigned char *out_data, unsigned char *in_data, int n);
void decode_arith_dirac_both (unsigned char *out_data, unsigned char *in_data, int n);
void decode_arith_exp (unsigned char *out_data, unsigned char *in_data, int n);


unsigned char out_data[N];
unsigned char in_data[N];
unsigned char c_data[N];

static void
dumpbits (unsigned char *bits, int n)
{
  int i;

  for(i=0;i<n;i++){
    if ((i&0xf) == 0) {
      printf("%04x: ", i);
    }
    if ((i&0xf) < 0xf) {
      printf("%02x ", bits[i]);
    } else {
      printf("%02x\n", bits[i]);
    }
  }
  if ((n & 0xf) < 0xf) {
    printf("\n");
  }
}

int
main (int argc, char *argv[])
{
  int x;
  int i;

  schro_init();

  x = 100;
  for(i=0;i<N;i++){
    in_data[i] = rand_u8() < x;
  }

  dumpbits(in_data + 9900, 100);

  //n = encode_arith_dirac (out_data, in_data, N);
  //dumpbits(out_data, n);

  encode_arith_exp (out_data, in_data, N);
  //dumpbits(out_data, n);

  decode_arith_exp (c_data, out_data, N);
  dumpbits(c_data + 9900, 100);

  return 0;
}


