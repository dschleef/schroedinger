
#include <stdio.h>

#include <schroedinger/schro.h>

void
dump_bits (SchroBits *bits)
{
  int i;
  
  for(i=0;i<bits->offset;i++){
    printf(" %d", (bits->buffer->data[(i>>3)] >> (7 - (i&7))) & 1);
  }
  printf(" (%d bits)\n", bits->offset);
}


int
main (int argc, char *argv[])
{
  int i;
  SchroBuffer *buffer = schro_buffer_new_and_alloc (100);
  SchroBits *bits;
  int value;
  int fail = 0;

  schro_init();

  bits = schro_bits_new();

  printf("unsigned int\n");
  for(i=0;i<11;i++) {
    schro_bits_encode_init (bits, buffer);
    schro_bits_encode_uint(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    schro_bits_decode_init (bits, buffer);
    value = schro_bits_decode_uint (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("signed int\n");
  for(i=-5;i<6;i++) {
    schro_bits_encode_init (bits, buffer);
    schro_bits_encode_sint(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    schro_bits_decode_init (bits, buffer);
    value = schro_bits_decode_sint (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  return fail;
}



