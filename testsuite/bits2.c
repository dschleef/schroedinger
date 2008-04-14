
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <schroedinger/schro.h>

void
dump_bits (SchroBits *bits, int n)
{
  int i;
  
  for(i=0;i<n*8;i++){
    printf(" %d", (bits->buffer->data[(i>>3)] >> (7 - (i&7))) & 1);
  }
  printf("\n");
}

#define N 100

int ref[N];

int
main (int argc, char *argv[])
{
  int i;
  SchroBuffer *buffer = schro_buffer_new_and_alloc (300);
  SchroBits *bits;
  int value;
  int fail = 0;
  int n;

  schro_init();

  srand(time(NULL));

  bits = schro_bits_new();

  printf("unsigned int\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = rand() & 0x7;
    schro_bits_encode_uint(bits,ref[i]);
  }
  schro_bits_flush (bits);
  n = schro_bits_get_offset (bits);
  schro_bits_free (bits);

  bits = schro_bits_new();
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_uint (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed int\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (rand() & 0xf) - 8;
    schro_bits_encode_sint (bits,ref[i]);
  }
  schro_bits_flush (bits);
  n = schro_bits_get_offset (bits);
  schro_bits_free (bits);

  bits = schro_bits_new();
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_sint (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  return fail;
}



