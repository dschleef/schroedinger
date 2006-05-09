
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>

#define BUFFER_SIZE 10000

int debug=1;

void
decode(uint8_t *dest, uint8_t *src, int n_bytes)
{
  SchroArith *a;
  SchroBuffer *buffer;
  SchroBits *bits;
  int i;
  int j;
  int value;
  int bit;

  a = schro_arith_new();

  buffer = schro_buffer_new_with_data (src, n_bytes);
  bits = schro_bits_new ();
  schro_bits_decode_init (bits, buffer);

  schro_arith_decode_init (a, bits);
  schro_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    value = 0;
    printf("%d:\n", i);
    for(j=0;j<8;j++){
      printf("[%04x %04x] %04x -> ", a->low, a->high, a->code);
      bit = schro_arith_context_decode_bit (a, 0);
      printf("%d\n", bit);
      value |= bit << (7-j);
    }
    dest[i] = value;
  }

  schro_arith_free(a);
  schro_bits_free (bits);
}

void
encode (uint8_t *dest, uint8_t *src, int n_bytes)
{
  SchroArith *a;
  SchroBuffer *buffer;
  SchroBits *bits;
  int i;
  int j;
  int bit;

  a = schro_arith_new();

  buffer = schro_buffer_new_with_data (dest, BUFFER_SIZE);
  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);
#if 0
  a->data = dest;
  a->size = BUFFER_SIZE;
#endif

  schro_arith_encode_init (a, bits);
  schro_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    printf("%d:\n", i);
    for(j=0;j<8;j++){
      bit = (src[i]>>(7-j))&1;
      printf("[%04x %04x] %d\n", a->low, a->high, bit);
      schro_arith_context_encode_bit (a, 0, bit);
    }
  }
  schro_arith_flush (a);

  schro_arith_free(a);
}

uint8_t buffer1[BUFFER_SIZE];
uint8_t buffer2[BUFFER_SIZE];
uint8_t buffer3[BUFFER_SIZE];

int
check (int n)
{

  encode(buffer2, buffer1, n);
#if 0
  for(i=0;i<4;i++){
    printf("%02x\n", buffer2[i]);
  }
#endif
  decode(buffer3, buffer2, n);

  if (memcmp (buffer1, buffer3, n)) {
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

  for (j = 0; j < 40; j++){
    int value;
    value = 0xff & random();

    for(i=0;i<100;i++){
      buffer1[i] = value;
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
      buffer1[i] = mask & random();
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
      buffer1[i] = mask & random();
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

