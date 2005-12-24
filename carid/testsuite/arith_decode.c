
#include <stdio.h>
#include <stdlib.h>
#include <carid/caridarith.h>

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

  file = fopen("test_file_arith.out","r");
  n_bytes = fread (input_buffer, 1, BUFFER_SIZE, file);
  fclose (file);

  a = carid_arith_new();

  a->data = input_buffer;
  a->size = BUFFER_SIZE;

  carid_arith_encode_init (a);
  carid_arith_context_init (a, 0, 1, 1);

  i = 0;
  while (a->offset < n_bytes) {
    for(j=0;j<8;j++){
      output_buffer[i] |= carid_arith_context_binary_decode (a, 0) << (7-j);
    }
    i++;
  }

  file = fopen("test_file.out","w");
  fwrite (output_buffer, 1, i, file);
  fclose (file);

  carid_arith_free(a);

  return 0;
}

