
#include <stdio.h>

#include <schroedinger/schro.h>

void
dump_bits (SchroBits *bits, int n)
{
  int i;
  
  for(i=0;i<n*8;i++){
    printf(" %d", (bits->buffer->data[(i>>3)] >> (7 - (i&7))) & 1);
  }
  printf(" (%d bytes)\n", n);
}


int
main (int argc, char *argv[])
{
  int i;
  SchroBuffer *buffer = schro_buffer_new_and_alloc (100);
  SchroBits *bits;
  int value;
  int fail = 0;
  int n;

  schro_init();

  printf("unsigned int\n");
  for(i=0;i<21;i++) {
    bits = schro_bits_new();
    schro_bits_encode_init (bits, buffer);
    schro_bits_encode_uint(bits,i);
    schro_bits_flush (bits);
    n = schro_bits_get_offset (bits);
    schro_bits_free (bits);

    bits = schro_bits_new();
    schro_bits_decode_init (bits, buffer);
    printf("%3d:", i);
    dump_bits (bits, n);
    value = schro_bits_decode_uint (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
    schro_bits_free (bits);
  }
  printf("\n");

  printf("signed int\n");
  for(i=-5;i<6;i++) {
    bits = schro_bits_new();
    schro_bits_encode_init (bits, buffer);
    schro_bits_encode_sint(bits,i);
    schro_bits_flush (bits);
    n = schro_bits_get_offset (bits);
    schro_bits_free (bits);

    bits = schro_bits_new();
    schro_bits_decode_init (bits, buffer);
    printf("%3d:", i);
    dump_bits (bits, n);
    value = schro_bits_decode_sint (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
    schro_bits_free (bits);
  }
  printf("\n");

  return fail;
}



