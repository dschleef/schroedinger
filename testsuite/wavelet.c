
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schroorc.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "common.h"

int16_t tmp[100];
int16_t tmp2[100];
int16_t *frame_data;

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

static void synth_schro_ext (int16_t *a, int filter, int n);
static void split_schro_ext (int16_t *a, int n, int filter);
static void dump (int16_t *a, int n);
static void dump_cmp (int16_t *a, int16_t *b, int n);


static void
gen_const(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=100;
  }
}

static void
gen_ramp(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i*100 + 50)/n;
  }
}

static void
gen_alternating(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i&1)*100;
  }
}

static void
gen_spike(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i==(n/2))*100;
  }
}

static void
gen_edge(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i<(n/2))*100;
  }
}

static void
gen_random(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=rand()&0x7f;
  }
}

typedef struct _Generator Generator;
struct _Generator {
  char *name;
  void (*create)(int16_t *dest, int len);
};

Generator generators[] = {
  { "constant", gen_const },
  { "ramp", gen_ramp },
  { "alternating", gen_alternating },
  { "spike", gen_spike },
  { "edge", gen_edge },
  { "random", gen_random }
};

static void
local_test(int filter)
{
  int16_t *a = tmp + 10;
  int n = 20;
  int i;

  for(i=0;i<sizeof(generators)/sizeof(generators[0]);i++){
    printf("  test \"%s\":\n", generators[i].name);
    generators[i].create(a,n);
    dump(a,n);
    split(a,n,filter);
    deinterleave(a,n);
    dump(a,n);
    interleave(a,n);
    synth(a,n,filter);
    dump(a,n);

    split_schro_ext(a,n,filter);
    deinterleave(a,n);
    dump(a,n);
    interleave(a,n);
    synth_schro_ext(a,n,filter);
    dump(a,n);
  }
  printf("\n");
}

static void
random_test(int filter)
{
  int16_t *a = tmp + 10;
  int n = 20;
  int i;
  int16_t b[100];
  int16_t c[100];
  int failed = 0;

  printf("  testing random arrays (split):\n");
  for(i=0;i<100;i++){
#if 0
    {
      int i;
      for(i=0;i<n;i++){
        a[i] = (i==i)*100;
      }
    }
#endif
    gen_random(a,n);
    memcpy(b,a,n*sizeof(int16_t));
    memcpy(c,a,n*sizeof(int16_t));

    split(a,n,filter);
    deinterleave(a,n);
    split_schro_ext(b,n,filter);
    deinterleave(b,n);

    if (memcmp(a,b,n*sizeof(int16_t)) != 0) {
      dump(c,n);
      dump(a,n);
      dump_cmp(b,a,n);
      printf("\n");
      failed++;
      //if (failed >=5) break;
    }
  }
  if (!failed) {
    printf("  OK\n");
  }
  printf("\n");

  printf("  testing random arrays (synth):\n");
  for(i=0;i<100;i++){
    gen_random(a,n);
    memcpy(b,a,n*sizeof(int16_t));

    synth(a,n,filter);
    deinterleave(a,n);
    synth_schro_ext(b,n,filter);
    deinterleave(b,n);

    if (memcmp(a,b,n*sizeof(int16_t)) != 0) {
      dump(a,n);
      dump_cmp(b,a,n);
      printf("\n");
      failed++;
      if (failed >=5) break;
    }
  }
  if (!failed) {
    printf("  OK\n");
  }
  printf("\n");
}


int
main (int argc, char *argv[])
{
  int filter;

  schro_init();

  for(filter=0;filter<=SCHRO_WAVELET_DAUBECHIES_9_7;filter++){
    printf("Filter %d:\n", filter);
    local_test(filter);
    random_test(filter);
  }

  return 0;
}

static void
dump (int16_t *a, int n)
{
  int i;
  for(i=0;i<n;i++){
    printf("%3d ", a[i]);
  }
  printf("\n");
}

static void
dump_cmp (int16_t *a, int16_t *b, int n)
{
  int i;
  for(i=0;i<n;i++){
    if (a[i] == b[i]) {
      printf("%3d ", a[i]);
    } else {
      printf("\033[00;01;37;41m%3d\033[00m ", a[i]);
    }
  }
  printf("\n");
}

static void
split_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  orc_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_split_ext_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_split_ext_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_split_ext_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_split_ext_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_ext_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_split_ext_daub97(hi, lo, n/2);
      break;
  }
  orc_interleave2_s16 (a, hi, lo, n/2);

}

static void
synth_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  orc_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_synth_ext_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_synth_ext_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_synth_ext_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_synth_ext_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_ext_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_synth_ext_daub97(hi, lo, n/2);
      break;
  }

  orc_interleave2_s16 (a, hi, lo, n/2);
}





