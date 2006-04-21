
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <schro/schro.h>

void
dump_bits (SchroBits *bits)
{
  int i;
  
  for(i=0;i<bits->offset;i++){
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

  schro_init();

  srand(time(NULL));

  bits = schro_bits_new();

  printf("unsigned unary\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    schro_bits_encode_uu(bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_uu (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed unary\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    schro_bits_encode_su(bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_su (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned truncated unary (n=3)\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x3;
    schro_bits_encode_ut(bits,ref[i], 3);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_ut (bits, 3);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned exp-Golomb\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    schro_bits_encode_uegol(bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_uegol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed exp-Golomb\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    schro_bits_encode_segol (bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_segol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned exp-exp-Golomb\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    schro_bits_encode_ue2gol(bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_ue2gol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed exp-exp-Golomb\n");
  schro_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    schro_bits_encode_se2gol (bits,ref[i]);
  }
  schro_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = schro_bits_decode_se2gol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }


  return fail;

}



