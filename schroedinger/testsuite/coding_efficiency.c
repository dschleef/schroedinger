
#include <stdio.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>

#include "arith_qm.h"

#define N 10000

SchroBuffer *data;

static int
efficiency_arith (int x)
{
  SchroArith *a;
  int i;
  int bits;
  int value;

  a = schro_arith_new();

  schro_arith_encode_init (a, data);
  schro_arith_init_contexts (a);

  for(i=0;i<N;i++) {
    value = (oil_rand_u8() < x);
    schro_arith_context_encode_bit (a, 0, value);
  }
  schro_arith_flush (a);

  bits = a->offset*8;

  schro_arith_free(a);

  return bits;
}

static int
efficiency_arith_qm (int x)
{
  int i;
  ArithQM coder = { 0 };
  int value;

  arith_qm_init (&coder);
  coder.data = data->data;
  for(i=0;i<10000;i++) {
    value = (oil_rand_u8() < x);
    arith_qm_encode (&coder, 0, value);
  }
  arith_qm_flush(&coder);

  return 8*coder.bp;
}





int
main (int argc, char *argv[])
{
  int x;
  int a, b;

  schro_init();

  data = schro_buffer_new_and_alloc (N);

  for(x = 0; x <= 128; x += 1) {
    a = efficiency_arith (x);
    b = efficiency_arith_qm (x);

    printf("%g %g %g\n", x/256.0, (double)a/N, (double)b/N);
  }

  return 0;
}


