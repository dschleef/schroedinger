
#include <stdio.h>
#include <stdlib.h>
#include <schro/schro.h>

#define BUFFER_SIZE 10000
uint8_t input_buffer[BUFFER_SIZE];
uint8_t output_buffer[BUFFER_SIZE];

#if 0
int
main (int argc, char *argv[])
{
  SchroArith *a;
  int i;
  int value;

  a = schro_arith_new();

  a->data = buffer;
  for(i=0;i<1000;i++) {
    buffer[i] = random();
  }

  schro_arith_decode_init (a);
  schro_arith_context_init (a, 0, 1, 1);

  for(i=0;i<10000;i++){
    printf("hi=%d lo=%d code=%04x count0=%d count1=%d\n",
        a->high, a->low, a->code, a->contexts[0].count0,
        a->contexts[0].count1);
    value = schro_arith_context_binary_decode (a, 0);
    printf(" --> %d\n", value);
  }

  schro_arith_free(a);

  return 0;
}
#endif

int
main (int argc, char *argv[])
{
  SchroArith *a;
  int i;
  FILE *file;
  int n_bytes;
  int j;
  SchroBuffer *output_buffer;
  SchroBits *bits;

  schro_init();

  file = fopen("test_file","r");
  n_bytes = fread (input_buffer, 1, BUFFER_SIZE, file);
  fclose(file);

  a = schro_arith_new();

  output_buffer = schro_buffer_new_and_alloc (BUFFER_SIZE);
  bits = schro_bits_new ();
  schro_bits_encode_init (bits, output_buffer);

  schro_arith_encode_init (a, bits);
  schro_arith_context_init (a, 0, 1, 1);

  for(i=0;i<n_bytes;i++){
    for(j=0;j<8;j++){
      schro_arith_context_encode_bit (a, 0, (input_buffer[i]>>(7-j))&1);
    }
  }
  schro_arith_flush (a);

  file = fopen("test_file_arith.out","w");
  n_bytes = fwrite (output_buffer, 1, a->bits->offset, file);
  fclose(file);

  return 0;
}

