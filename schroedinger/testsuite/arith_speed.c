
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>

#include <liboil/liboilprofile.h>
#include <liboil/liboilrandom.h>

#define BUFFER_SIZE 1000000

int debug=1;

static void test_arith_reload_nextcode (SchroArith *arith);
static int arith_context_decode_bit_test (SchroArith *arith, int i);
static int arith_context_decode_bit_ref (SchroArith *arith, int i);

int
decode(SchroBuffer *buffer, int n, OilProfile *prof, int type)
{
  SchroArith *a;
  int i;
  int j;
  int x = 0;

  oil_profile_init (prof);
  for(j=0;j<10;j++){
    a = schro_arith_new();

    schro_arith_decode_init (a, buffer);
    schro_arith_init_contexts (a);

    switch (type) {
      case 0:
        oil_profile_start (prof);
        for(i=0;i<n;i++){
          x += arith_context_decode_bit_ref (a, 0);
        }
        oil_profile_stop (prof);
        break;
      case 1:
        oil_profile_start (prof);
        for(i=0;i<n;i++){
          x += arith_context_decode_bit_test (a, 0);
        }
        oil_profile_stop (prof);
        break;
    }

    a->buffer = NULL;
    schro_arith_free(a);
  }

  return x;
}

void
dump_bits (SchroBuffer *buffer, int n)
{
  int i;

  for(i=0;i<n;i++){
    printf("%02x ", buffer->data[i]);
    if ((i&15)==15) {
      printf ("\n");
    }
  }
  printf ("\n");
}

void
encode (SchroBuffer *buffer, int n, int freq)
{
  SchroArith *a;
  int i;
  int bit;

  a = schro_arith_new();

  schro_arith_encode_init (a, buffer);
  schro_arith_init_contexts (a);

  for(i=0;i<n;i++){
    bit = oil_rand_u8() < freq;
    schro_arith_context_encode_bit (a, 0, bit);
  }
  schro_arith_flush (a);

  a->buffer = NULL;
  schro_arith_free(a);
}

int
check (int n, int freq)
{
  SchroBuffer *buffer;
  OilProfile prof;
  double ave, std;
  int x;
  int y;

  buffer = schro_buffer_new_and_alloc (100000);

  encode(buffer, n, freq);

  x = decode(buffer, n, &prof, 0);
  oil_profile_get_ave_std (&prof, &ave, &std);
  printf("ref  %d,%d: %g (%g) %d\n", n, freq, ave, std, x);

  y = decode(buffer, n, &prof, 1);
  oil_profile_get_ave_std (&prof, &ave, &std);
  printf("test %d,%d: %g (%g) %d\n", n, freq, ave, std, y);
  if (x != y) {
    printf("BROKEN\n");
  }

  schro_buffer_unref (buffer);

  return 0;
}

void *x;

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  /* force it to be non-inlined */
  x = &arith_context_decode_bit_ref;
  x = &arith_context_decode_bit_test;

  //while(1) check(1000, 128);
  check(1000, 128);
  for(i=100;i<=1000;i+=100) {
    //check(i, 128);
    check(i, 256);
  }
  check(2000, 256);
  check(3000, 256);
  check(4000, 256);
  check(5000, 256);
  check(100000, 256);
#if 0
  for(i=0;i<=256;i+=16) {
    check(100, i);
  }
#endif

  return 0;
}



static void
test_arith_reload_nextcode (SchroArith *arith)
{
  while(arith->nextbits <= 24) {
    if (arith->dataptr < arith->maxdataptr) {
      arith->nextcode |= arith->dataptr[0] << (24-arith->nextbits);
    } else {
      arith->nextcode |= 0xff << (24-arith->nextbits);
    }
    arith->nextbits+=8;
    arith->dataptr++;
    arith->offset++;
  }
}


static int
arith_context_decode_bit_ref (SchroArith *arith, int i)
{
  unsigned int count;
  unsigned int value;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;

  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = arith->division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  count = arith->code - arith->range[0] + 1;
  range = arith->range[1] - arith->range[0] + 1;
  range_x_prob = (range * probability0) >> 16;
  value = (count > range_x_prob);

  arith->range[1 - value] = arith->range[0] + range_x_prob - 1 + value;
  arith->contexts[i].count[value]++;

  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }

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

  return value;
}


#if 0
static int
arith_context_decode_bit_test (SchroArith *arith, int i)
{
  SchroArithContext *context = arith->contexts + i;

#include <schroedinger/schrooffsets.h>
  __asm__ __volatile__ (
      //calc_count_range(arith);
      "  movzwl a_range(%0), %%ecx\n"
      "  movzwl a_code(%0), %%eax\n"
      "  movzwl a_range+2(%0), %%edx\n"
#if 0
      "  subl $1, %%ecx\n"
      "  subl %%ecx, %%esi\n"
      "  subl %%ecx, %%edx\n"
#else
      "  negl %%ecx\n"
      "  leal 1(%%eax,%%ecx,1), %%esi\n"
      "  leal 1(%%edx,%%ecx,1), %%ecx\n"
#endif
      //"  movl %%esi, a_count(%0)\n"
      //"  movl %%ecx, a_range_value(%0)\n"

      //calc_prob0(arith, context);
      "  movzwl c_count(%1), %%eax\n"
      "  addw (c_count + 2)(%1), %%ax\n"
      "  movzwl a_division_factor(%0,%%eax,2), %%eax\n"
#if 1
      "  mulw c_count(%1)\n"
#else
      "  movzwl c_count(%1), %%edx\n"
      "  imul %%edx, %%eax\n"
#endif

      // calc_value()
#if 1
      "  imul %%ecx, %%eax\n"
      "  shrl $16, %%eax\n"
#else
      "  cmp $0x10000, %%ecx\n"
      "  je skipmul\n"
      "  mul %%cx\n"
      "  mov %%dx, %%ax\n"
      "skipmul:\n"
#endif

      "  mov %%esi, %%ecx\n"
#if 0
      "  subl %%eax, %%ecx\n"
      "  neg %%ecx\n"
      "  shrl $31, %%ecx\n"
      "  and $1, %%ecx\n"
#else
      "  cmpl %%eax, %%ecx\n"
      "  setg %%cl\n"
      "  movzbl %%cl, %%ecx\n"
#endif

      "  xor $1, %%ecx\n"
      "  addw a_range(%0), %%ax\n"
      "  subw %%cx, %%ax\n"
      "  movw %%ax, a_range(%0,%%ecx,2)\n"
      "  xor $1, %%ecx\n"
      "  addw $1, c_count(%1, %%ecx, 2)\n"
      "  movw %%cx, a_value(%0)\n"

      //maybe_shift_context(context);
#if 0
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  shrw $8, %%cx\n"
#if 0
      "  shrw %%cl, c_count(%1)\n"
      "  addw %%cx, c_count(%1)\n"
      "  shrw %%cl, c_count+2(%1)\n"
      "  addw %%cx, c_count+2(%1)\n"
#else
      "  movw %%cx, %%ax\n"
      "  shl $16, %%eax\n"
      "  orl %%ecx, %%eax\n"
      "  movl c_count(%1), %%edx\n"
      "  shrl %%cl, %%edx\n"
      "  addl %%eax, %%edx\n"
      "  and $0x00ff00ff, %%edx\n"
      "  movl %%edx, c_count(%1)\n"
#endif
#else
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  cmp $255, %%cx\n"
      "  jle noshift\n"
#if 0
      "  shrw $1, c_count(%1)\n"
      "  addw $1, c_count(%1)\n"
      "  shrw $1, c_count+2(%1)\n"
      "  addw $1, c_count+2(%1)\n"
#else
      "  movl c_count(%1), %%ecx\n"
      "  shrl $1, %%ecx\n"
      "  addl $0x00010001, %%ecx\n"
      "  and $0x00ff00ff, %%ecx\n"
      "  mov %%ecx, c_count(%1)\n"
#endif
      "noshift:\n"
#endif

      //fixup_range(arith);
      // i = ((arith->range[1]&0xf000)>>8) | ((arith->range[0]&0xf000)>>12);
      // fixup = arith->fixup_shift[i];
      "  movw a_range + 2(%0), %%ax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      // if (n == 0) return;
      "  test %%eax, %%eax\n"
      "  je fixup_done\n"

      // n = arith->fixup_shift[i] & 0xf;
      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      // arith->range[0] <<= n;
      "  shlw %%cl, a_range(%0)\n"
      // arith->range[1] <<= n;
      // arith->range[1] |= (1<<n)-1;
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      // arith->code <<= n;
      // arith->code |= (arith->nextcode >> ((32-n)&0x1f));
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      // arith->nextcode <<= n;
      "  shll %%cl, a_nextcode(%0)\n"
      // arith->nextbits-=n;
      "  subl %%ecx, a_nextbits(%0)\n"

      // flip = arith->fixup_shift[i] & 0x8000;
      "  andw $0x8000, %%ax\n"
      // arith->code ^= flip;
      "  xorw %%ax, a_code(%0)\n"
      // arith->range[0] ^= flip;
      "  xorw %%ax, a_range(%0)\n"
      // arith->range[1] ^= flip;
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jl fixup_nextcode\n"
      "fixup_loop:\n"
      "  movzwl a_range+2(%0), %%eax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      "  test %%eax, %%eax\n"
      "  je fixup_nextcode\n"

      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      "  shlw %%cl, a_range(%0)\n"
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      "  shll %%cl, a_nextcode(%0)\n"
      "  subl %%ecx, a_nextbits(%0)\n"

      "  andw $0x8000, %%ax\n"
      "  xorw %%ax, a_code(%0)\n"
      "  xorw %%ax, a_range(%0)\n"
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jge fixup_loop\n"
      "fixup_nextcode:\n"
      "  movl $24, %%ecx\n"
      "  subl a_nextbits(%0), %%ecx\n"
      "  jb fixup_done\n"

      "  movl a_dataptr(%0), %%eax\n"
      "  cmpl a_maxdataptr(%0), %%eax\n"
      "  jge past_end\n"

      "  movzbl 0(%%eax), %%edx\n"
      "  jmp cont\n"

      "past_end:\n"
      "  movl $0xff, %%edx\n"

      "cont:\n"
      "  shll %%cl, %%edx\n"
      "  orl %%edx, a_nextcode(%0)\n"

      "  addl $8, a_nextbits(%0)\n"
      "  addl $1, a_dataptr(%0)\n"
      "  addl $1, a_offset(%0)\n"
      "  jmp fixup_nextcode\n"

      "fixup_done:\n"

      :
      : "r" (arith), "r" (context)
      : "esi", "eax", "ecx", "edx", "memory");

  return arith->value;
}
#endif


static int
arith_context_decode_bit_test (SchroArith *arith, int i)
{
  SchroArithContext *context = arith->contexts + i;

#include <schroedinger/schrooffsets.h>
  __asm__ __volatile__ (
      //calc_count_range(arith);
      "  movzwl a_range(%0), %%ecx\n"
      "  movzwl a_code(%0), %%eax\n"
      "  movzwl a_range+2(%0), %%edx\n"
#if 0
      "  subl $1, %%ecx\n"
      "  subl %%ecx, %%esi\n"
      "  subl %%ecx, %%edx\n"
#else
      "  negl %%ecx\n"
      "  leal 1(%%eax,%%ecx,1), %%eax\n"
      "  leal 1(%%edx,%%ecx,1), %%ecx\n"
#endif
      "  movl %%eax, a_count(%0)\n"
      //"  movl %%ecx, a_range_value(%0)\n"

      //calc_prob0(arith, context);
      "  movzwl c_count(%1), %%eax\n"
      "  addw (c_count + 2)(%1), %%ax\n"
      "  movzwl a_division_factor(%0,%%eax,2), %%eax\n"
#if 1
      "  mulw c_count(%1)\n"
#else
      "  movzwl c_count(%1), %%edx\n"
      "  imul %%edx, %%eax\n"
#endif

      // calc_value()
#if 1
      "  imul %%ecx, %%eax\n"
      "  shrl $16, %%eax\n"
#else
      "  cmp $0x10000, %%ecx\n"
      "  je skipmul\n"
      "  mul %%cx\n"
      "  mov %%dx, %%ax\n"
      "skipmul:\n"
#endif

      "  mov a_count(%0), %%ecx\n"
#if 0
      "  subl %%eax, %%ecx\n"
      "  neg %%ecx\n"
      "  shrl $31, %%ecx\n"
      "  and $1, %%ecx\n"
#else
      "  cmpl %%eax, %%ecx\n"
      "  setg %%cl\n"
      "  movzbl %%cl, %%ecx\n"
#endif

      "  xor $1, %%ecx\n"
      "  addw a_range(%0), %%ax\n"
      "  subw %%cx, %%ax\n"
      "  movw %%ax, a_range(%0,%%ecx,2)\n"
      "  xor $1, %%ecx\n"
      "  addw $1, c_count(%1, %%ecx, 2)\n"
      "  movw %%cx, a_value(%0)\n"

      //maybe_shift_context(context);
#if 0
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  shrw $8, %%cx\n"
#if 0
      "  shrw %%cl, c_count(%1)\n"
      "  addw %%cx, c_count(%1)\n"
      "  shrw %%cl, c_count+2(%1)\n"
      "  addw %%cx, c_count+2(%1)\n"
#else
      "  movw %%cx, %%ax\n"
      "  shl $16, %%eax\n"
      "  orl %%ecx, %%eax\n"
      "  movl c_count(%1), %%edx\n"
      "  shrl %%cl, %%edx\n"
      "  addl %%eax, %%edx\n"
      "  and $0x00ff00ff, %%edx\n"
      "  movl %%edx, c_count(%1)\n"
#endif
#else
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  cmp $255, %%cx\n"
      "  jle noshift\n"
#if 0
      "  shrw $1, c_count(%1)\n"
      "  addw $1, c_count(%1)\n"
      "  shrw $1, c_count+2(%1)\n"
      "  addw $1, c_count+2(%1)\n"
#else
      "  movl c_count(%1), %%ecx\n"
      "  shrl $1, %%ecx\n"
      "  addl $0x00010001, %%ecx\n"
      "  and $0x00ff00ff, %%ecx\n"
      "  mov %%ecx, c_count(%1)\n"
#endif
      "noshift:\n"
#endif

      //fixup_range(arith);
      // i = ((arith->range[1]&0xf000)>>8) | ((arith->range[0]&0xf000)>>12);
      // fixup = arith->fixup_shift[i];
      "  movw a_range + 2(%0), %%ax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      // if (n == 0) return;
      "  test %%eax, %%eax\n"
      "  je fixup_done\n"

      // n = arith->fixup_shift[i] & 0xf;
      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      // arith->range[0] <<= n;
      "  shlw %%cl, a_range(%0)\n"
      // arith->range[1] <<= n;
      // arith->range[1] |= (1<<n)-1;
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      // arith->code <<= n;
      // arith->code |= (arith->nextcode >> ((32-n)&0x1f));
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      // arith->nextcode <<= n;
      "  shll %%cl, a_nextcode(%0)\n"
      // arith->nextbits-=n;
      "  subl %%ecx, a_nextbits(%0)\n"

      // flip = arith->fixup_shift[i] & 0x8000;
      "  andw $0x8000, %%ax\n"
      // arith->code ^= flip;
      "  xorw %%ax, a_code(%0)\n"
      // arith->range[0] ^= flip;
      "  xorw %%ax, a_range(%0)\n"
      // arith->range[1] ^= flip;
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jl fixup_nextcode\n"
      "fixup_loop:\n"
      "  movzwl a_range+2(%0), %%eax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      "  test %%eax, %%eax\n"
      "  je fixup_nextcode\n"

      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      "  shlw %%cl, a_range(%0)\n"
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      "  shll %%cl, a_nextcode(%0)\n"
      "  subl %%ecx, a_nextbits(%0)\n"

      "  andw $0x8000, %%ax\n"
      "  xorw %%ax, a_code(%0)\n"
      "  xorw %%ax, a_range(%0)\n"
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jge fixup_loop\n"
      "fixup_nextcode:\n"
      "  movl $24, %%ecx\n"
      "  subl a_nextbits(%0), %%ecx\n"
      "  jb fixup_done\n"

      "  movl a_dataptr(%0), %%eax\n"
      "  cmpl a_maxdataptr(%0), %%eax\n"
      "  jge past_end\n"

      "  movzbl 0(%%eax), %%edx\n"
      "  jmp cont\n"

      "past_end:\n"
      "  movl $0xff, %%edx\n"

      "cont:\n"
      "  shll %%cl, %%edx\n"
      "  orl %%edx, a_nextcode(%0)\n"

      "  addl $8, a_nextbits(%0)\n"
      "  addl $1, a_dataptr(%0)\n"
      "  addl $1, a_offset(%0)\n"
      "  jmp fixup_nextcode\n"

      "fixup_done:\n"

      :
      : "r" (arith), "r" (context)
      : "eax", "ecx", "edx", "memory");

  return arith->value;
}

