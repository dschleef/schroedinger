
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <carid/caridarith.h>

#define BUFFER_SIZE 10000

int debug=0;

void
decode(uint8_t *dest, uint8_t *src, int n_bytes)
{
  CaridArith *a;
  int i;
  int j;
  int value;

  a = carid_arith_new();

  a->data = src;

  carid_arith_decode_init (a);
  carid_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    value = 0;
    //printf("%d:\n", i);
    for(j=0;j<8;j++){
      value |= carid_arith_context_binary_decode (a, 0) << (7-j);
      //printf("[%04x %04x] %04x\n", a->low, a->high, a->code);
    }
    dest[i] = value;
  }

  carid_arith_free(a);
}

void
encode (uint8_t *dest, uint8_t *src, int n_bytes)
{
  CaridArith *a;
  int i;
  int j;

  a = carid_arith_new();

  a->data = dest;
  a->size = BUFFER_SIZE;

  carid_arith_encode_init (a);
  carid_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    //printf("%d:\n", i);
    for(j=0;j<8;j++){
      carid_arith_context_binary_encode (a, 0, (src[i]>>(7-j))&1);
      //printf("[%04x %04x]\n", a->low, a->high);
    }
  }
  carid_arith_flush (a);

  carid_arith_free(a);
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
      if (!check(10)){
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
      if (!check(10)){
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
      if (!check(10)){
        printf("Failed: encoding/decoding masked random array (size=%d mask=%02x)\n",
            n, mask);
        fail = 1;
      }
    }
  }

  return fail;
}

