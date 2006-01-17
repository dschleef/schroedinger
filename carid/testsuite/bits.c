
#include <stdio.h>

#include <carid/carid.h>

void
dump_bits (CaridBits *bits)
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
  CaridBuffer *buffer = carid_buffer_new_and_alloc (100);
  CaridBits *bits;
  int value;
  int fail = 0;

  carid_init();

  bits = carid_bits_new();

  printf("unsigned unary\n");
  for(i=0;i<5;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_uu(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_uu (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("signed unary\n");
  for(i=-5;i<6;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_su(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_su (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("unsigned truncated unary (n=4)\n");
  for(i=0;i<5;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_ut(bits,i,4);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_ut (bits, 4);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("unsigned exp-Golomb\n");
  for(i=0;i<11;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_uegol(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_uegol (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("signed exp-Golomb\n");
  for(i=-5;i<6;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_segol(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_segol (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("unsigned exp-exp-Golomb\n");
  for(i=0;i<11;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_ue2gol(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_ue2gol (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("signed exp-exp-Golomb\n");
  for(i=-5;i<6;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_se2gol(bits,i);
    printf("%3d:", i);
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_se2gol (bits);
    if (value != i) {
      printf("decode failed (%d != %d)\n", value, i);
      fail = 1;
    }
  }
  printf("\n");

  printf("unsigned exp-Golomb\n");
  for(i=0;i<31;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_uegol(bits,(1<<i));
    printf("%3d:", (1<<i));
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_uegol (bits);
    if (value != (1<<i)) {
      printf("decode failed (%d != %d)\n", value, (1<<i));
      fail = 1;
    }
  }
  printf("\n");

  printf("unsigned exp-exp-Golomb\n");
  for(i=0;i<31;i++) {
    carid_bits_encode_init (bits, buffer);
    carid_bits_encode_ue2gol(bits,(1<<i));
    printf("%3d:", (1<<i));
    dump_bits (bits);
    carid_bits_decode_init (bits, buffer);
    value = carid_bits_decode_ue2gol (bits);
    if (value != (1<<i)) {
      printf("decode failed (%d != %d)\n", value, (1<<i));
      fail = 1;
    }
  }
  printf("\n");

  return fail;

}



