
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>

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
    buffer[i] = rand();
  }

  schro_arith_decode_init (a);

  for(i=0;i<10000;i++){
    printf("hi=%d lo=%d code=%04x count0=%d count1=%d\n",
        a->high, a->low, a->code, a->contexts[0].count0,
        a->contexts[0].count1);
    value = schro_arith_binary_decode (a, 0);
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

  schro_init();

  file = fopen("test_file","r");
  n_bytes = fread (input_buffer, 1, BUFFER_SIZE, file);
  fclose(file);

  a = schro_arith_new();

  output_buffer = schro_buffer_new_and_alloc (BUFFER_SIZE);
  schro_arith_encode_init (a, output_buffer);

  for(i=0;i<n_bytes;i++){
    for(j=0;j<8;j++){
      schro_arith_encode_bit (a, j, (input_buffer[i]>>(7-j))&1);
    }
  }
  schro_arith_flush (a);

  file = fopen("test_file_arith.out","w");
  n_bytes = fwrite (output_buffer, 1, a->offset, file);
  fclose(file);

  return 0;
}

