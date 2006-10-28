
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include <liboil/liboilprofile.h>
#include <liboil/liboilrandom.h>

#define BUFFER_SIZE 10000

int debug=1;

static void test_arith_reload_nextcode (SchroArith *arith)
{
}

static void
fixup_range_ref (SchroArith *arith)
{
  do {
    if ((arith->range[1] & (1<<15)) == (arith->range[0] & (1<<15))) {
      /* do nothing */
    } else if ((arith->range[0] & (1<<14)) && !(arith->range[1] & (1<<14))) {
      arith->code ^= (1<<14);
      arith->range[0] ^= (1<<14);
      arith->range[1] ^= (1<<14);
    } else {
      break;
    }

    arith->range[0] <<= 1;
    arith->range[1] <<= 1;
    arith->range[1]++;

    arith->code <<= 1;
    arith->code |= (arith->nextcode >> 31);
    arith->nextcode <<= 1;
    arith->nextbits--;
    if (arith->nextbits == 0) {
      test_arith_reload_nextcode(arith);
    }
  } while (1);
}

static void
fixup_range_1 (SchroArith *arith)
{
  int i;
  int n;
  int flip;

  do {
//printf("got: %04x %04x %04x\n", arith->range[0], arith->range[1], arith->code);
    i = ((arith->range[1]&0xf000)>>8) | ((arith->range[0]&0xf000)>>12);
//printf("i %02x\n", i);

    n = schro_table_arith_shift[i] & 0xf;
    if (n == 0) return;
    flip = schro_table_arith_shift[i] & 0x8000;
//printf("n,flip %d %d\n", n, flip);

    arith->range[0] <<= n;
    arith->range[1] <<= n;
    arith->range[1] |= (1<<n)-1;

    arith->code <<= n;
    arith->code |= (arith->nextcode >> ((32-n)&0x1f));
    arith->nextcode <<= n;
    arith->nextbits-=n;

    arith->code ^= flip;
    arith->range[0] ^= flip;
    arith->range[1] ^= flip;
  } while(n >= 3);
}

int
main (int argc, char *argv[])
{
  int i;
  SchroArith *arith_orig;
  SchroArith *arith_ref;
  SchroArith *arith_test;

  schro_init();

  arith_orig = schro_arith_new();
  arith_ref = schro_arith_new();
  arith_test = schro_arith_new();

  for(i=0;i<1000000;i++){
    int a,b,c;

    arith_orig->nextcode = 0xaaaa5555;
    arith_orig->nextbits = 32;
    do {
      a = rand() & 0xffff;
      b = rand() & 0xffff;
      if (b<=a) { int tmp = b;b=a;a=tmp; }
      c = rand() & 0xffff;
      if (c<=b) { int tmp = c;c=b;b=tmp; }
      if (b<=a) { int tmp = b;b=a;a=tmp; }
    } while (c-a < 1);
    arith_orig->range[0] = a;
    arith_orig->code = b;
    arith_orig->range[1] = c;

    memcpy (arith_ref, arith_orig, sizeof(*arith_orig));
    fixup_range_ref (arith_ref);

    memcpy (arith_test, arith_orig, sizeof(*arith_orig));
    fixup_range_1 (arith_test);

    if (arith_ref->range[0] != arith_test->range[0] ||
        arith_ref->range[1] != arith_test->range[1] ||
        arith_ref->code != arith_test->code ||
        arith_ref->nextcode != arith_test->nextcode) {
      printf("orig: %04x %04x %04x\n", arith_orig->range[0], arith_orig->range[1],
          arith_orig->code);
      printf("ref:  %04x %04x %04x\n", arith_ref->range[0], arith_ref->range[1],
          arith_ref->code);
      printf("test: %04x %04x %04x\n", arith_test->range[0], arith_test->range[1],
          arith_test->code);
    }
  }

  return 0;
}


