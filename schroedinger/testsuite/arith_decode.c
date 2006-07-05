
#include <stdio.h>
#include <stdlib.h>
#include <schroedinger/schro.h>

#define BUFFER_SIZE 10000
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
  SchroBuffer *input_buffer;

  schro_init();

  input_buffer = schro_buffer_new_and_alloc (BUFFER_SIZE);
  file = fopen("test_file_arith.out","r");
  n_bytes = fread (input_buffer->data, 1, BUFFER_SIZE, file);
  fclose (file);
  input_buffer->length = n_bytes;

  a = schro_arith_new();

  schro_arith_decode_init (a, input_buffer);
  schro_arith_context_init (a, 0, 1, 1);

  i = 0;
  while (a->offset < n_bytes) {
    for(j=0;j<8;j++){
      output_buffer[i] |= schro_arith_context_decode_bit (a, 0) << (7-j);
    }
    i++;
  }

  file = fopen("test_file.out","w");
  (void)fwrite (output_buffer, 1, i, file);
  fclose (file);

  schro_arith_free(a);

  return 0;
}

