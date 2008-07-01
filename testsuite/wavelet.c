
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int16_t tmp[100];
int16_t tmp2[100];
int16_t *frame_data;

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

static void synth(int16_t *a, int filter, int n);
static void split (int16_t *a, int filter, int n);
static void synth_schro_ext (int16_t *a, int filter, int n);
static void split_schro_ext (int16_t *a, int n, int filter);
static void deinterleave (int16_t *a, int n);
static void interleave (int16_t *a, int n);
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
interleave (int16_t *a, int n)
{
  int i;
  for(i=0;i<n/2;i++){
    tmp2[i*2] = a[i];
    tmp2[i*2 + 1] = a[n/2 + i];
  }
  for(i=0;i<n;i++){
    a[i] = tmp2[i];
  }
}

static void
deinterleave (int16_t *a, int n)
{
  int i;
  for(i=0;i<n/2;i++){
    tmp2[i] = a[i*2];
    tmp2[n/2 + i] = a[i*2+1];
  }
  for(i=0;i<n;i++){
    a[i] = tmp2[i];
  }
}

static void
extend(int16_t *a, int n)
{
  a[-8] = a[0];
  a[-7] = a[1];
  a[-6] = a[0];
  a[-5] = a[1];
  a[-4] = a[0];
  a[-3] = a[1];
  a[-2] = a[0];
  a[-1] = a[1];
  a[n+0] = a[n-2];
  a[n+1] = a[n-1];
  a[n+2] = a[n-2];
  a[n+3] = a[n-1];
  a[n+4] = a[n-2];
  a[n+5] = a[n-1];
  a[n+6] = a[n-2];
  a[n+7] = a[n-1];
}

static void
synth(int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (a[i] + a[i+2] + 1)>>1;
      }
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      for(i=0;i<n;i+=2){
        a[i] -= (a[i+1] + 1)>>1;
      }
      for(i=0;i<n;i+=2){
        a[i+1] += a[i];
      }
      break;
    case SCHRO_WAVELET_FIDELITY:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
      break;
  }
}

static void
split (int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (a[i] + a[i+2] + 1)>>1;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      for(i=0;i<n;i+=2){
        a[i+1] -= a[i];
      }
      for(i=0;i<n;i+=2){
        a[i] += (a[i+1] + 1)>>1;
      }
      break;
    case SCHRO_WAVELET_FIDELITY:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
      break;
  }
}

static void
split_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

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
  oil_interleave2_s16 (a, hi, lo, n/2);

}

static void
synth_schro_ext (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

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

  oil_interleave2_s16 (a, hi, lo, n/2);
}





#ifdef unused
static void
schro_split_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 7, 4 };
  static int16_t stage2_offset_shift[] = { 2, 2 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
#endif

  lo[0] -= (-hi[1] + 9 * hi[0] + 9 * hi[1] - hi[2] + 8)>>4;
  lo[n-2] -= (-hi[n-3] + 9 * hi[n-2] + 8 * hi[n-1] + 8)>>4;
  lo[n-1] -= (-2*hi[n-2] + 18 * hi[n-1] + 8)>>4;
  oil_mas4_add_s16 (lo + 1, lo + 1, hi, stage1_weights, stage1_offset_shift, n - 3);

  hi[0] += (lo[0] + lo[0] + 2)>>2;
  oil_mas2_add_s16 (hi + 1, hi + 1, lo, stage2_weights, stage2_offset_shift, n - 1);
}

static void
schro_split_53 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 0, 1 };
  static int16_t stage2_offset_shift[] = { 2, 2 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (a[i] + a[i+2] + 1)>>1;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
#endif

  lo[n-1] -= (hi[n-1] + hi[n-1] + 1)>>1;
  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage1_offset_shift, n - 1);

  hi[0] += (lo[0] + lo[0] + 2)>>2;
  oil_mas2_add_s16 (hi + 1, hi + 1, lo, stage2_weights, stage2_offset_shift, n - 1);
}

static void
schro_split_135 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 7, 4 };
  static int16_t stage2_offset_shift[] = { 16, 5 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
#endif

  lo[0] -= (-hi[1] + 9 * hi[0] + 9 * hi[1] - hi[2] + 8)>>4;
  lo[n-2] -= (-hi[n-3] + 9 * hi[n-2] + 8 * hi[n-1] + 8)>>4;
  lo[n-1] -= (-2*hi[n-2] + 18 * hi[n-1] + 8)>>4;
  oil_mas4_add_s16 (lo + 1, lo + 1, hi, stage1_weights, stage1_offset_shift, n - 3);

  hi[0] += (-lo[1] + 9 * lo[0] + 9 * lo[0] - lo[1] + 16)>>5;
  hi[1] += (-lo[0] + 9 * lo[0] + 9 * lo[1] - lo[2] + 16)>>5;
  hi[n-1] += (-lo[n-3] + 9 * lo[n-2] + 9 * lo[n-1] - lo[n-2] + 16)>>5;
  oil_mas4_add_s16 (hi + 2, hi + 2, lo, stage2_weights, stage2_offset_shift, n - 3);
}

static void
schro_split_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

  for(i=0;i<n;i++) {
    lo[i] -= hi[i];
  }
  for(i=0;i<n;i++) {
    hi[i] += ((lo[i] + 1)>>1);
  }
}

static void
schro_split_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
  static int16_t stage1_offset_shift[] = { 128, 8 };
  static int16_t stage2_offset_shift[] = { 127, 8 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
#endif

  hi[0] += (-8*lo[3] + 21*lo[2] - 46*lo[1] + 161*lo[0] +
      161*lo[0] - 46*lo[1] + 21*lo[2] - 8*lo[3] + 128) >> 8;
  hi[1] += (-8*lo[2] + 21*lo[1] - 46*lo[0] + 161*lo[0] +
      161*lo[1] - 46*lo[2] + 21*lo[3] - 8*lo[4] + 128) >> 8;
  hi[2] += (-8*lo[1] + 21*lo[0] - 46*lo[0] + 161*lo[1] +
      161*lo[2] - 46*lo[3] + 21*lo[4] - 8*lo[5] + 128) >> 8;
  hi[3] += (-8*lo[0] + 21*lo[0] - 46*lo[1] + 161*lo[2] +
      161*lo[3] - 46*lo[4] + 21*lo[5] - 8*lo[6] + 128) >> 8;
  hi[n-3] += (-8*lo[n-7] + 21*lo[n-6] - 46*lo[n-5] + 161*lo[n-4] +
      161*lo[n-3] - 46*lo[n-2] + 21*lo[n-1] - 8*lo[n-2] + 128) >> 8;
  hi[n-2] += (-8*lo[n-6] + 21*lo[n-5] - 46*lo[n-4] + 161*lo[n-3] +
      161*lo[n-2] - 46*lo[n-1] + 21*lo[n-2] - 8*lo[n-3] + 128) >> 8;
  hi[n-1] += (-8*lo[n-5] + 21*lo[n-4] - 46*lo[n-3] + 161*lo[n-2] +
      161*lo[n-1] - 46*lo[n-2] + 21*lo[n-3] - 8*lo[n-4] + 128) >> 8;

  oil_mas8_add_s16 (hi + 4, hi + 4, lo, stage1_weights, stage1_offset_shift, n - 7);

  lo[0] -= (-2*hi[3] + 10*hi[2] - 25*hi[1] + 81*hi[0] +
      81*hi[1] - 25*hi[2] + 10*hi[3] - 2*hi[4] + 128) >> 8;
  lo[1] -= (-2*hi[2] + 10*hi[1] - 25*hi[0] + 81*hi[1] +
      81*hi[2] - 25*hi[3] + 10*hi[4] - 2*hi[5] + 128) >> 8;
  lo[2] -= (-2*hi[1] + 10*hi[0] - 25*hi[1] + 81*hi[2] +
      81*hi[3] - 25*hi[4] + 10*hi[5] - 2*hi[6] + 128) >> 8;
  lo[n-4] -= (-2*hi[n-7] + 10*hi[n-6] - 25*hi[n-5] + 81*hi[n-4] +
      81*hi[n-3] - 25*hi[n-2] + 10*hi[n-1] - 2*hi[n-1] + 128) >> 8;
  lo[n-3] -= (-2*hi[n-6] + 10*hi[n-5] - 25*hi[n-4] + 81*hi[n-3] +
      81*hi[n-2] - 25*hi[n-1] + 10*hi[n-1] - 2*hi[n-2] + 128) >> 8;
  lo[n-2] -= (-2*hi[n-5] + 10*hi[n-4] - 25*hi[n-3] + 81*hi[n-2] +
      81*hi[n-1] - 25*hi[n-1] + 10*hi[n-2] - 2*hi[n-3] + 128) >> 8;
  lo[n-1] -= (-2*hi[n-4] + 10*hi[n-3] - 25*hi[n-2] + 81*hi[n-1] +
      81*hi[n-1] - 25*hi[n-2] + 10*hi[n-3] - 2*hi[n-4] + 128) >> 8;

  oil_mas8_add_s16 (lo + 3, lo + 3, hi, stage2_weights, stage2_offset_shift, n - 7);
}

static void
schro_split_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -6497, -6497 };
  static int16_t stage2_weights[] = { -217, -217 };
  static int16_t stage3_weights[] = { 3616, 3616 };
  static int16_t stage4_weights[] = { 1817, 1817 };
  static int16_t stage12_offset_shift[] = { 2047, 12 };
  static int16_t stage34_offset_shift[] = { 2048, 12 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
#endif

  lo[n-1] -= (6497*hi[n-1] + 6497 * hi[n-1] + 2048)>>12;
  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage12_offset_shift, n - 1);

  hi[0] -= (217*lo[0] + 217 * lo[0] + 2048)>>12;
  oil_mas2_add_s16 (hi + 1, hi + 1, lo, stage2_weights, stage12_offset_shift, n - 1);

  lo[n-1] += (3616*hi[n-1] + 3616 * hi[n-1] + 2048)>>12;
  oil_mas2_add_s16 (lo, lo, hi, stage3_weights, stage34_offset_shift, n - 1);

  hi[0] += (1817*lo[0] + 1817 * lo[0] + 2048)>>12;
  oil_mas2_add_s16 (hi + 1, hi + 1, lo, stage4_weights, stage34_offset_shift, n - 1);
}


#if 0
void
split_schro (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_split_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_split_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_split_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_split_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_split_daub97(hi, lo, n/2);
      break;
  }
  oil_interleave2_s16 (a, hi, lo, n/2);

}
#endif

static void
schro_synth_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 1, 2 };
  static int16_t stage2_offset_shift[] = { 8, 4 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
#endif

  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage2_weights, stage2_offset_shift, n);
}

static void
schro_synth_53 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 1, 2 };
  static int16_t stage2_offset_shift[] = { 1, 1 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (a[i] + a[i+2] + 1)>>1;
      }
#endif

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage2_offset_shift, n);
}

static void
schro_synth_135 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 15, 5 };
  static int16_t stage2_offset_shift[] = { 8, 4 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-a[i-3] + 9 * a[i-1] + 9 * a[i+1] - a[i+3] + 16)>>5;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
#endif

  lo[-1] = lo[0];
  lo[-2] = lo[1];
  lo[n] = lo[n-2];
  oil_mas4_add_s16 (hi, hi, lo - 2, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  oil_mas4_add_s16 (lo, lo, hi-1, stage2_weights, stage2_offset_shift, n);
}

static void
schro_synth_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

#if 0
      for(i=0;i<n;i+=2){
        a[i] -= (a[i+1] + 1)>>1;
      }
      for(i=0;i<n;i+=2){
        a[i+1] += a[i];
      }
#endif

  for(i=0;i<n;i++) {
    hi[i] -= ((lo[i] + 1)>>1);
  }
  for(i=0;i<n;i++) {
    lo[i] += hi[i];
  }
}

static void
schro_synth_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static int16_t stage2_weights[] = { 8, -21, 46, -161, -161, 46, -21, 8 };
  static int16_t stage1_offset_shift[] = { 128, 8 };
  static int16_t stage2_offset_shift[] = { 127, 8 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (-2*a[i-6] + 10*a[i-4] - 25*a[i-2] + 81*a[i] +
            81*a[i+2] - 25*a[i+4] + 10*a[i+6] - 2*a[i+8] + 128) >> 8;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (-8*a[i-7] + 21*a[i-5] - 46*a[i-3] + 161*a[i-1] +
            161*a[i+1] - 46*a[i+3] + 21*a[i+5] - 8*a[i+7] + 128) >> 8;
      }
#endif

  hi[-3] = hi[3];
  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  hi[n+2] = hi[n-3];
  hi[n+3] = hi[n-4];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage1_weights, stage1_offset_shift, n);

  lo[-4] = lo[3];
  lo[-3] = lo[2];
  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];
  lo[n+2] = lo[n-4];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage2_weights, stage2_offset_shift, n);
}

static void
schro_synth_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -1817, -1817 };
  static int16_t stage2_weights[] = { -3616, -3616 };
  static int16_t stage3_weights[] = { 217, 217 };
  static int16_t stage4_weights[] = { 6497, 6497 };
  static int16_t stage12_offset_shift[] = { 2047, 12 };
  static int16_t stage34_offset_shift[] = { 2048, 12 };

#if 0
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (1817*a[i-1] + 1817 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] -= (3616*a[i] + 3616 * a[i+2] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (217*a[i-1] + 217 * a[i+1] + 2048)>>12;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i+1] += (6497*a[i] + 6497 * a[i+2] + 2048)>>12;
      }
#endif

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage12_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage3_weights, stage34_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage4_weights, stage34_offset_shift, n);
}

static void
synth_schro (int16_t *a, int n, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n/2);

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_synth_desl93 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_synth_53 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
      schro_synth_135 (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_synth_haar (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_fidelity (hi, lo, n/2);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_synth_daub97(hi, lo, n/2);
      break;
  }

  oil_interleave2_s16 (a, hi, lo, n/2);
}
#endif

