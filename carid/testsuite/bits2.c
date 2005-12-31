
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <carid/caridbuffer.h>
#include <carid/caridbits.h>

void
dump_bits (CaridBits *bits)
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
  CaridBuffer *buffer = carid_buffer_new_and_alloc (300);
  CaridBits *bits;
  int value;
  int fail = 0;

  srand(time(NULL));

  bits = carid_bits_new();

  printf("unsigned unary\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    carid_bits_encode_uu(bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_uu (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed unary\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    carid_bits_encode_su(bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_su (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned truncated unary (n=3)\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x3;
    carid_bits_encode_ut(bits,ref[i], 3);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_ut (bits, 3);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned exp-Golomb\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    carid_bits_encode_uegol(bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_uegol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed exp-Golomb\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    carid_bits_encode_segol (bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_segol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("unsigned exp-exp-Golomb\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = random() & 0x7;
    carid_bits_encode_ue2gol(bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_ue2gol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }

  printf("signed exp-exp-Golomb\n");
  carid_bits_encode_init (bits, buffer);
  for(i=0;i<N;i++) {
    ref[i] = (random() & 0xf) - 8;
    carid_bits_encode_se2gol (bits,ref[i]);
  }
  carid_bits_decode_init (bits, buffer);
  for(i=0;i<N;i++) {
    value = carid_bits_decode_se2gol (bits);
    if (value != ref[i]) {
      printf("decode failed (%d != %d) at offset %d\n", value, ref[i], i);
      fail = 1;
    }
  }


  return fail;

}



