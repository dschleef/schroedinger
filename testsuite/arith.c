
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>

#define BUFFER_SIZE 10000

int debug=0;
int verbose = 0;

static void
decode(SchroBuffer *dest, SchroBuffer *src)
{
  SchroArith *a;
  int i;
  int j;
  int value;
  int bit;

  a = schro_arith_new();

  schro_arith_decode_init (a, src);

  for(i=0;i<dest->length;i++){
    value = 0;
    if (verbose) printf("%d:\n", i);
    for(j=0;j<8;j++){
      if (verbose) printf("[%04x %04x] %04x -> ", a->range[0], a->range[1],
          a->code);
      bit = schro_arith_decode_bit (a, 0);
      if (verbose) printf("%d\n", bit);
      value |= bit << (7-j);
    }
    dest->data[i] = value;
  }

  schro_arith_free(a);
}

static void
encode (SchroBuffer *dest, SchroBuffer *src)
{
  SchroArith *a;
  int i;
  int j;
  int bit;

  a = schro_arith_new();

  schro_arith_encode_init (a, dest);

  for(i=0;i<src->length;i++){
    if (verbose) printf("%d:\n", i);
    for(j=0;j<8;j++){
      bit = (src->data[i]>>(7-j))&1;
      if (verbose) printf("[%04x %04x] %d\n", a->range[0], a->range[1], bit);
      schro_arith_encode_bit (a, 0, bit);
    }
  }
  schro_arith_flush (a);

  dest->length = a->offset;

  schro_arith_free(a);
}

SchroBuffer *buffer1;
SchroBuffer *buffer2;
SchroBuffer *buffer3;

static int
check (int n)
{
  buffer1->length = n;
  buffer3->length = n;

  encode(buffer2, buffer1);
#if 0
  for(i=0;i<4;i++){
    printf("%02x\n", buffer2[i]);
  }
#endif
  decode(buffer3, buffer2);

  if (memcmp (buffer1->data, buffer3->data, n)) {
    return 0;
  }
  return 1;
}

int
main (int argc, char *argv[])
{
  int i;
  int n;
  int fail=0;
  int j;

  schro_init();

  buffer1 = schro_buffer_new_and_alloc (1000);
  buffer2 = schro_buffer_new_and_alloc (1000);
  buffer3 = schro_buffer_new_and_alloc (1000);

  for (j = 0; j < 40; j++){
    int value;
    value = 0xff & rand();

    for(i=0;i<100;i++){
      buffer1->data[i] = value;
    }

    for(n=0;n<100;n++){
      if (debug) {
        printf("Checking: encoding/decoding constant array (size=%d value=%d)\n",
            n, value);
      }
      if (!check(n)){
        printf("Failed: encoding/decoding constant array (size=%d value=%d)\n",
            n, value);
        fail = 1;
      }
    }
  }

  for (j = 0; j < 8; j++) {
    int mask;
    mask = (1<<(j+1)) - 1;
    for(i=0;i<100;i++){
      buffer1->data[i] = mask & rand();
    }

    for(n=0;n<100;n++){
      if (debug) {
        printf("Checking: encoding/decoding masked random array (size=%d mask=%02x)\n",
            n, mask);
      }
      if (!check(n)){
        printf("Failed: encoding/decoding masked random array (size=%d mask=%02x)\n",
            n, mask);
        fail = 1;
      }
    }
  }

  for (j = 0; j < 8; j++) {
    int mask;
    mask = (1<<(j+1)) - 1;
    for(i=0;i<100;i++){
      buffer1->data[i] = mask & rand();
    }

    for(n=0;n<100;n++){
      if (debug) {
        printf("Checking: encoding/decoding masked random array (size=%d mask=%02x)\n",
            n, mask);
      }
      if (!check(n)){
        printf("Failed: encoding/decoding masked random array (size=%d mask=%02x)\n",
            n, mask);
        fail = 1;
      }
    }
  }

  return fail;
}

