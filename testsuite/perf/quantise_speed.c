
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <orc/orc.h>
#include <orc-test/orcprofile.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <schroedinger/schrotables.c>

#define N 10000

int16_t a[N];
int16_t b[N];
int16_t c[N];

int16_t table[65536];

int orc_profile_get_min (OrcProfile *prof)
{
  int i;
  int min;
  min = prof->hist_time[0];
  for(i=0;i<10;i++){
    if (prof->hist_count[i] > 0) {
      if (prof->hist_time[i] < min) {
        min = prof->hist_time[i];
      }
    }
  }
  return min;
}

#if 0
static int
__schro_dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}
#endif

static int
__schro_quantise_shift (int value, int quant_shift)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)>>quant_shift;
    value = -x;
  } else {
    x = value>>quant_shift;
    value = x;
  }
  return value;
}

#if 0
static int
__schro_quantise_table (int value, int16_t *table)
{
  return table[value];
}
#endif

void
schro_quantise_s16_ref (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;
  int quant_factor = schro_table_quant[quant_index];
#if 0
  int quant_offset = is_intra ? schro_table_offset_1_2[quant_index] :
    schro_table_offset_3_8[quant_index];
#endif

  for(i=0;i<n;i++){
    unsigned int x;
    int value = src[i];

    if (src[i] == 0) {
      dest[i] = 0;
    } else if (src[i] < 0) {
      x = (-src[i])<<2;
      if (x < quant_factor) {
        x = 0;
      } else {
        x /= quant_factor;
      }
      value = -x; } else {
      x = value<<2;
      if (x < quant_factor) {
        x = 0;
      } else {
        x /= quant_factor;
      }
      value = x;
    }
    dest[i] = value;
  }
}

void
schro_quantise_s16_table (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;

  for(i=0;i<n;i++){
    dest[i] = table[src[i] + 32768];
  }
}

void
schro_quantise_s16_shift (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  if ((quant_index & 3) == 0) {
    int i;

    for(i=0;i<n;i++){
      dest[i] = __schro_quantise_shift (src[i], quant_index >> 2);
    }
  } else {
    schro_quantise_s16_ref (dest, src, quant_index, is_intra, n);
  }
}

void
schro_quantise_s16_test (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  int quant_factor = schro_table_quant[quant_index];
#if 0
  int quant_offset = is_intra ? schro_table_offset_1_2[quant_index] :
    schro_table_offset_3_8[quant_index];
#endif
  int inv_quant_factor;
  int i;

  //inv_quant_factor = ((1LL<<32) + quant_factor/2) / quant_factor;
  inv_quant_factor = ((1LL<<16) + quant_factor/2) / quant_factor;

  for(i=0;i<n;i++){
    unsigned int x;
    int value = src[i];

    if (src[i] == 0) {
      dest[i] = 0;
    } else if (src[i] < 0) {
      x = (-src[i])<<2;
      x = (x * inv_quant_factor) >> 16;
      value = -x; } else {
      x = value<<2;
      x = (x * inv_quant_factor) >> 16;
      value = x;
    }
    dest[i] = value;
  }
}

void
generate_table (int16_t *table, int quant_index, schro_bool is_intra)
{
  int i;
  for(i=0;i<65536;i++){
    table[i] = i-32768;
  }
  schro_quantise_s16_ref (table, table, quant_index, is_intra, 65536);
}

void
quantise_speed (int quant_index)
{
  OrcProfile prof1;
  OrcProfile prof2;
  OrcProfile prof3;
  double ave1;
  double ave2;
  double ave3;
  int i;
  int j;
  schro_bool is_intra = TRUE;

  orc_profile_init (&prof1);
  orc_profile_init (&prof2);
  orc_profile_init (&prof3);

  for(i=0;i<10;i++) {
    for(j=0;j<N;j++){
      a[j] = (rand () & 0xff) - 128;
    }
    generate_table (table, quant_index, is_intra);

    orc_profile_start (&prof1);
    schro_quantise_s16_ref (b, a, quant_index, FALSE, N);
    orc_profile_stop (&prof1);
    orc_profile_start (&prof2);
    schro_quantise_s16_table (c, a, quant_index, FALSE, N);
    orc_profile_stop (&prof2);
    orc_profile_start (&prof3);
    schro_quantise_s16_test (c, a, quant_index, FALSE, N);
    orc_profile_stop (&prof3);

#if 0
    for(j=0;j<N;j++){
      printf("%d %d: %d %d %d\n", quant_index, j, a[j], b[j], c[j]);
      if (b[j] != c[j]) {
        fprintf(stderr, "mismatch\n");
        exit (1);
      }
    }
#endif
  }

  ave1 = orc_profile_get_min (&prof1);
  ave2 = orc_profile_get_min (&prof2);
  ave3 = orc_profile_get_min (&prof3);
  printf("%d %g %g %g\n", quant_index, ave1, ave2, ave3);
}


int
main (int argc, char *argv[])
{
  int i;

  orc_init();

  for(i=0;i<60;i++){
    quantise_speed (i);
  }

  return 0;
}

