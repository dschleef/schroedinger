
#include "config.h"

#include <stdio.h>
#include <math.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroorc.h>

#include "schroedinger/schrotables.c"

#if 0
static int
dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return ((q * quant_factor - quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

static int
quantise (int value, int quant_factor, int quant_offset)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = ((-value)<<2) - quant_offset + (quant_factor>>1);
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = -x;
  } else {
    x = (value<<2) - quant_offset + (quant_factor>>1);
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = x;
  }
  return value;
}
#endif

void test_quant  (int quant_index, int ack);
void test_dequant  (int quant_index);

#define N 2000

int16_t a[N];
int16_t b[N];
int16_t c[N];
int16_t d[N];

int
main (int argc, char *argv[])
{
  int i;

  schro_init();
  schro_tables_init();

#if 0
  for(i=0;i<65536;i+=256) {
    printf("%d\n", i);
    test_quant (3, i);
  }
#endif
  for(i=0;i<61;i++) {
    test_quant (i, 0);
  }
  for(i=0;i<61;i++) {
    test_dequant (i);
  }

  return 0;
}

void
test_quant  (int quant_index, int ack)
{
  int quant_factor = schro_table_quant[quant_index];
  int quant_offset = schro_table_offset_1_2[quant_index];
  int quant_shift = quant_index/4 + 2;
  int error = FALSE;
  int i;
  int bad_count;

  //printf("index %d:\n", quant_index);

  for(i=0;i<2000;i++){
    a[i] = i - 1000;
    b[i] = i - 1000;
    c[i] = i - 1000;
    d[i] = i - 1000;
  }

  for(i=0;i<N;i++){
    a[i] = schro_quantise(a[i],quant_factor,quant_offset);
  }

  if (quant_index == 0) {
    /* do nothing */
  } else if ((quant_index & 3) == 0) {
    orc_quantdequant2_s16 (b, d, quant_shift,
        quant_offset - quant_factor/2, quant_factor, quant_offset + 2, N);
  } else if (quant_index != 3) {
    int inv_quant;

    inv_quant = schro_table_inverse_quant[quant_index];

    orc_quantdequant1_s16 (b, d, inv_quant,
        quant_offset - quant_factor/2 - (quant_index > 8),
        quant_shift, quant_factor, quant_offset + 2, N);
  } else {
    int inv_quant;

    inv_quant = schro_table_inverse_quant[quant_index];

    orc_quantdequant3_s16 (b, d, inv_quant,
        quant_offset - quant_factor/2,
        quant_shift + 16, quant_factor,
        quant_offset + 2,
        32768, N);
  }

  bad_count=0;
  for(i=0;i<N;i++){
    //if (a[i] < b[i] - 1 || a[i] > b[i] + 1) error = TRUE;
    //if (a[i] != b[i]) error = TRUE;
    if (a[i] != b[i]) bad_count++;
  }
  printf("index %d: %d\n", quant_index, bad_count);

  if (error) {
    for(i=0;i<N;i++){
      printf("%d %d  %d %d  %d %d %c\n", i, c[i], a[i],
          schro_dequantise (a[i], quant_factor, quant_offset),
          b[i], d[i],
          (a[i]!=b[i]) ? '*':' ');
    }
  }
}

void
test_dequant  (int quant_index)
{
  int quant_factor = schro_table_quant[quant_index];
  int quant_offset = schro_table_offset_3_8[quant_index];
  int error = FALSE;
  int i;

  printf("index %d:\n", quant_index);

  for(i=0;i<N;i++){
    a[i] = i - 1000;
    b[i] = schro_quantise(a[i],quant_factor,quant_offset);
    c[i] = b[i];
  }

  for(i=0;i<N;i++){
    b[i] = schro_dequantise(b[i],quant_factor,quant_offset);
  }

  if (quant_index == 0) {
    /* do nothing */
  } else {
    orc_dequantise_s16 (c, c, quant_factor, quant_offset, N);
  }

  for(i=0;i<N;i++){
    if (b[i] < c[i] - 1 || b[i] > c[i] + 1) error = TRUE;
  }

  if (error) {
    for(i=0;i<N;i++){
      printf("%d %d %d %d %c\n", i, a[i], b[i], c[i], (b[i]!=c[i]) ? '*':' ');
    }
  }
}

