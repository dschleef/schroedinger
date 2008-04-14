
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <schroedinger/schro.h>

int fail;

void
dump_bits (SchroBits *bits, int n)
{
  int i;
  
  for(i=0;i<n*8;i++){
    printf(" %d", (bits->buffer->data[(i>>3)] >> (7 - (i&7))) & 1);
  }
  printf("\n");
}


int test (void)
{
  SchroBuffer *buffer = schro_buffer_new_and_alloc (300);
  SchroBits *bits;
  SchroBits bits2;
  int i;
  int chunk1;
  int chunk2;
  int ref[10];
  int x;

  chunk1 = rand()&0x1f;

  bits = schro_bits_new ();
  schro_bits_encode_init (bits, buffer);

  for(i=0;i<chunk1;i++){
    schro_bits_encode_bit (bits, rand()&1);
  }

  chunk2 = 0;
  for(i=0;i<10;i++){
    ref[i] = rand() & 0xf;
    schro_bits_encode_sint (bits, ref[i]);
    chunk2 += schro_bits_estimate_sint (ref[i]);
  }

  for(i=0;i<100;i++){
    schro_bits_encode_bit (bits, rand()&1);
  }

  schro_bits_flush (bits);
  schro_bits_free (bits);


  bits = schro_bits_new ();
  schro_bits_decode_init (bits, buffer);

  schro_bits_copy (&bits2, bits);

  schro_bits_skip_bits (&bits2, chunk1);
  schro_bits_set_length (&bits2, chunk2);

  printf("chunk1 %d\n", chunk1);
  for(i=0;i<10;i++){
    x = schro_bits_decode_sint (&bits2);
    printf("%d %d\n", ref[i], x);
    if (x != ref[i]) fail = TRUE;
  }

  schro_bits_free (bits);

  return 0;
}

int
main (int argc, char *argv[])
{
  schro_init();

  srand(time(NULL));

  test();

  return fail;
}



