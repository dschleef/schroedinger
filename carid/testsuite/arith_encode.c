
#include <stdio.h>
#include <stdlib.h>
#include <carid/carid.h>

#define BUFFER_SIZE 10000
uint8_t input_buffer[BUFFER_SIZE];
uint8_t output_buffer[BUFFER_SIZE];

#if 0
int
main (int argc, char *argv[])
{
  CaridArith *a;
  int i;
  int value;

  a = carid_arith_new();

  a->data = buffer;
  for(i=0;i<1000;i++) {
    buffer[i] = random();
  }

  carid_arith_decode_init (a);
  carid_arith_context_init (a, 0, 1, 1);

  for(i=0;i<10000;i++){
    printf("hi=%d lo=%d code=%04x count0=%d count1=%d\n",
        a->high, a->low, a->code, a->contexts[0].count0,
        a->contexts[0].count1);
    value = carid_arith_context_binary_decode (a, 0);
    printf(" --> %d\n", value);
  }

  carid_arith_free(a);

  return 0;
}
#endif

int
main (int argc, char *argv[])
{
  CaridArith *a;
  int i;
  FILE *file;
  int n_bytes;
  int j;
  CaridBuffer *output_buffer;
  CaridBits *bits;

  carid_init();

  file = fopen("test_file","r");
  n_bytes = fread (input_buffer, 1, BUFFER_SIZE, file);
  fclose(file);

  a = carid_arith_new();

  output_buffer = carid_buffer_new_and_alloc (BUFFER_SIZE);
  bits = carid_bits_new ();
  carid_bits_encode_init (bits, output_buffer);

  carid_arith_encode_init (a, bits);
  carid_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    for(j=0;j<8;j++){
      carid_arith_context_encode_bit (a, 0, (input_buffer[i]>>(7-j))&1);
    }
  }
  carid_arith_flush (a);

  file = fopen("test_file_arith.out","w");
  n_bytes = fwrite (output_buffer, 1, a->bits->offset, file);
  fclose(file);

  return 0;
}

