
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>

int fail;

void
dump_bits (SchroPack *pack, int n)
{
  int i;
  
  for(i=0;i<n*8;i++){
    printf(" %d", (pack->buffer->data[(i>>3)] >> (7 - (i&7))) & 1);
  }
  printf("\n");
}

void
schro_unpack_dump (SchroUnpack *unpack)
{
  printf("data[] = %02x %02x %02x %02x\n",
      unpack->data[0], unpack->data[1], unpack->data[2], unpack->data[3]);
  printf("n_bits_left = %d\n", unpack->n_bits_left);
  printf("n_bits_read = %d\n", unpack->n_bits_read);
  printf("shift_register = %08x\n", unpack->shift_register);
  printf("n_bits_in_shift_register = %d\n", unpack->n_bits_in_shift_register);
  printf("guard_bit = %d\n", unpack->guard_bit);
  printf("overrun = %d\n", unpack->overrun);
  exit(1);
}


int test1 (void)
{
  SchroBuffer *buffer = schro_buffer_new_and_alloc (100);
  SchroPack *pack;
  SchroUnpack unpack;
  int i;
  int ref[800];
  int x;
  int guard;

  for(i=0;i<800;i++){
    ref[i] = rand()&1;
  }
  guard = rand()&1;

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);
  for(i=0;i<800;i++){
    schro_pack_encode_bit (pack, ref[i]);
  }
  schro_pack_flush(pack);
  schro_pack_free (pack);


  schro_unpack_init_with_data (&unpack, buffer->data, buffer->length, guard);

  for(i=0;i<800;i++){
    x = schro_unpack_decode_bit (&unpack);
    if (x != ref[i]) {
      printf("test1 failed at bit %d\n", i);
      schro_unpack_dump(&unpack);
      return 0;
    }
  }
  for(i=0;i<100;i++){
    x = schro_unpack_decode_bit (&unpack);
    if (x != guard) {
      printf("test1 failed at guard bit %d\n", i);
      schro_unpack_dump(&unpack);
      return 0;
    }
  }

  schro_buffer_unref (buffer);

  return 1;
}

int test2 (void)
{
  SchroBuffer *buffer = schro_buffer_new_and_alloc (1000);
  SchroPack *pack;
  SchroUnpack unpack;
  int i;
  int ref[100];
  int x;
  int n_bytes;

  memset (buffer->data, 0, 1000);
  for(i=0;i<100;i++){
    ref[i] = rand()&0xff;
  }

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);
  for(i=0;i<100;i++){
    schro_pack_encode_uint (pack, ref[i]);
  }
  schro_pack_flush(pack);
  n_bytes = schro_pack_get_offset (pack);
  schro_pack_free (pack);


  schro_unpack_init_with_data (&unpack, buffer->data, n_bytes, 1);

  for(i=0;i<100;i++){
    x = schro_unpack_decode_uint (&unpack);
    if (x != ref[i]) {
      printf("test2 failed at symbol %d (%d should be %d)\n", i, x, ref[i]);
      schro_unpack_dump(&unpack);
      return 0;
    }
  }

  schro_buffer_unref (buffer);

  return 1;
}

int test3 (void)
{
  SchroBuffer *buffer = schro_buffer_new_and_alloc (1000);
  SchroPack *pack;
  SchroUnpack unpack;
  SchroUnpack unpack2;
  int i;
  int ref[100];
  int x;
  int n_bytes;
  int n_bits = 0;
  int n_bits2 = 0;

  memset (buffer->data, 0, 1000);
  for(i=0;i<100;i++){
    ref[i] = (rand()&0xff) - 128;
  }

  pack = schro_pack_new ();
  schro_pack_encode_init (pack, buffer);
  n_bits = 0;
  for(i=0;i<50;i++){
    schro_pack_encode_sint (pack, ref[i]);
    n_bits += schro_pack_estimate_sint (ref[i]);
  }
  n_bits2 = 0;
  for(i=50;i<100;i++){
    schro_pack_encode_sint (pack, ref[i]);
    n_bits2 += schro_pack_estimate_sint (ref[i]);
  }
  schro_pack_flush(pack);
  n_bytes = schro_pack_get_offset (pack);
  schro_pack_free (pack);


  schro_unpack_init_with_data (&unpack, buffer->data, n_bytes, 1);
  schro_unpack_copy (&unpack2, &unpack);

  schro_unpack_limit_bits_remaining (&unpack, n_bits);
  for(i=0;i<50;i++){
    x = schro_unpack_decode_sint (&unpack);
    if (x != ref[i]) {
      SCHRO_ERROR("test3 failed at symbol %d (%d should be %d)", i, x, ref[i]);
      schro_unpack_dump(&unpack);
      return 0;
    }
  }
  for(i=0;i<10;i++){
    x = schro_unpack_decode_sint (&unpack);
    if (x != 0) {
      SCHRO_ERROR("test3 failed at symbol %d (%d should be 0)", i, x);
      schro_unpack_dump(&unpack);
      return 0;
    }
  }

  schro_unpack_skip_bits (&unpack2, n_bits);
  schro_unpack_limit_bits_remaining (&unpack2, n_bits2);
  for(i=50;i<100;i++){
    x = schro_unpack_decode_sint (&unpack2);
    if (x != ref[i]) {
      SCHRO_ERROR("test3 failed at symbol %d (%d should be %d)", i, x, ref[i]);
      schro_unpack_dump(&unpack2);
      return 0;
    }
  }
  for(i=0;i<10;i++){
    x = schro_unpack_decode_sint (&unpack2);
    if (x != 0) {
      SCHRO_ERROR("test3 failed at symbol %d (%d should be 0)", i, x);
      schro_unpack_dump(&unpack2);
      return 0;
    }
  }

  schro_buffer_unref (buffer);

  return 1;
}

int test4 (void)
{
  SchroBuffer *buffer = schro_buffer_new_and_alloc (100);
  SchroPack *pack;
  SchroUnpack unpack;
  int i;
  int16_t x;
  int ok = 1;


  for(i=-1000;i<1000;i++){
    pack = schro_pack_new ();
    schro_pack_encode_init (pack, buffer);
    schro_pack_encode_sint (pack, i);
    schro_pack_flush(pack);
    schro_pack_free (pack);

    schro_unpack_init_with_data (&unpack, buffer->data, buffer->length, 1);
    schro_unpack_decode_sint_s16 (&x, &unpack, 1);

    if (i != x) {
      printf("%d %d %c\n", i, x, (i==x)?' ':'x');
      ok = 0;
    }
  }

  schro_buffer_unref (buffer);

  return ok;
}


int
main (int argc, char *argv[])
{
  int ok = 1;
  int i;

  schro_init();

  srand(time(NULL));

  for(i=0;i<100;i++){
    ok &= test1();
  }
  for(i=0;i<100;i++){
    ok &= test2();
  }
  for(i=0;i<100;i++){
    ok &= test3();
  }

  ok &= test4();

  if (ok) exit(0);
  exit(1);
}



