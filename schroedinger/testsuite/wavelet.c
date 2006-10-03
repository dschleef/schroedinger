
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

int filtershift[] = { 1, 1, 1, 0, 1, 2, 0, 1 };

void synth(int16_t *a, int filter, int n);
void split (int16_t *a, int filter, int n);
void synth_schro (int16_t *a, int filter, int n);
void split_schro (int16_t *a, int n, int filter);
void deinterleave (int16_t *a, int n);
void interleave (int16_t *a, int n);
void dump (int16_t *a, int n);


void
gen_const(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=100;
  }
}

void
gen_ramp(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i*100 + 50)/n;
  }
}

void
gen_alternating(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i&1)*100;
  }
}

void
gen_spike(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i==(n/2))*100;
  }
}

void
gen_edge(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=(i<(n/2))*100;
  }
}

void
gen_random(int16_t *a, int n)
{
  int i;

  for(i=0;i<n;i++){
    a[i]=random()&0x7f;
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

void
local_test(int filter)
{
  int16_t *a = tmp + 10;
  int n = 20;
  int $;

  for($=0;$<sizeof(generators)/sizeof(generators[0]);$++){
    printf("  test \"%s\":\n", generators[$].name);
    generators[$].create(a,n);
    dump(a,n);
    split(a,n,filter);
    deinterleave(a,n);
    dump(a,n);
    interleave(a,n);
    synth(a,n,filter);
    dump(a,n);

    split_schro(a,n,filter);
    deinterleave(a,n);
    dump(a,n);
    interleave(a,n);
    synth_schro(a,n,filter);
    dump(a,n);
  }
  printf("\n");
}

void
random_test(int filter)
{
  int16_t *a = tmp + 10;
  int n = 20;
  int $;
  int16_t b[100];
  int failed = 0;

  printf("  testing random arrays (split):\n");
  for($=0;$<100;$++){
    gen_random(a,n);
    memcpy(b,a,n*sizeof(int16_t));

    split(a,n,filter);
    deinterleave(a,n);
    split_schro(b,n,filter);
    deinterleave(b,n);

    if (memcmp(a,b,n*sizeof(int16_t)) != 0) {
      dump(a,n);
      dump(b,n);
      printf("\n");
      failed++;
      if (failed >=5) break;
    }
  }
  if (!failed) {
    printf("  OK\n");
  }
  printf("\n");

  printf("  testing random arrays (synth):\n");
  for($=0;$<100;$++){
    gen_random(a,n);
    memcpy(b,a,n*sizeof(int16_t));

    synth(a,n,filter);
    deinterleave(a,n);
    synth_schro(b,n,filter);
    deinterleave(b,n);

    if (memcmp(a,b,n*sizeof(int16_t)) != 0) {
      dump(a,n);
      dump(b,n);
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

  for(filter=0;filter<=SCHRO_WAVELET_DAUB_9_7;filter++){
    printf("Filter %d:\n", filter);
    local_test(filter);
    random_test(filter);
  }

  return 0;
}

void
dump (int16_t *a, int n)
{
  int i;
  for(i=0;i<n;i++){
    printf("%3d ", a[i]);
  }
  printf("\n");
}

void
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

void
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

void
extend(int16_t *a, int n)
{
  int i;
  for(i=1;i<8;i++){
    a[-i] = a[i];
    a[(n-1)+i] = a[(n-1)-i];
  }
}

void
synth(int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      break;
    case SCHRO_WAVELET_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] -= (a[i-1] + a[i+1] + 2)>>2;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] += (a[i] + a[i+2] + 1)>>1;
      }
      break;
    case SCHRO_WAVELET_13_5:
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
    case SCHRO_WAVELET_HAAR_2:
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
    case SCHRO_WAVELET_DAUB_9_7:
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

void
split (int16_t *a, int n, int filter)
{
  int i;

  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (-a[i-2] + 9 * a[i] + 9 * a[i+2] - a[i+4] + 8)>>4;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_5_3:
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i + 1] -= (a[i] + a[i+2] + 1)>>1;
      }
      extend(a,n);
      for(i=0;i<n;i+=2){
        a[i] += (a[i-1] + a[i+1] + 2)>>2;
      }
      break;
    case SCHRO_WAVELET_13_5:
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
    case SCHRO_WAVELET_HAAR_2:
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
    case SCHRO_WAVELET_DAUB_9_7:
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

void
schro_split_desl93 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 7, 4 };
  static int16_t stage2_offset_shift[] = { 2, 2 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage1_weights, stage1_offset_shift, n);

  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);

  for(i=0;i<n;i++){
    a[2*i] = hi[i];
    a[2*i+1] = lo[i];
  }
}

void
schro_split_53 (int16_t *hi, int16_t *lo, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 0, 1 };
  static int16_t stage2_offset_shift[] = { 2, 2 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_split_135 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 7, 4 };
  static int16_t stage2_offset_shift[] = { 16, 5 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi-1, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];
  lo[-2] = lo[1];
  lo[n] = lo[n-2];

  oil_mas4_add_s16 (hi, hi, lo - 2, stage2_weights, stage2_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_split_haar (int16_t *a, int n)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  for(i=0;i<n;i++) {
    lo[i] -= hi[i];
  }
  for(i=0;i<n;i++) {
    hi[i] += ((lo[i] + 1)>>1);
  }

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_split_fidelity (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
  static int16_t stage1_offset_shift[] = { 128, 8 };
  static int16_t stage2_offset_shift[] = { 127, 8 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  lo[-4] = lo[3];
  lo[-3] = lo[2];
  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];
  lo[n+2] = lo[n-4];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage1_weights, stage1_offset_shift, n);

  hi[-3] = hi[3];
  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  hi[n+2] = hi[n-3];
  hi[n+3] = hi[n-4];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage2_weights, stage2_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_split_daub97 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -6497, -6497 };
  static int16_t stage2_weights[] = { -217, -217 };
  static int16_t stage3_weights[] = { 3616, 3616 };
  static int16_t stage4_weights[] = { 1817, 1817 };
  static int16_t stage12_offset_shift[] = { 2047, 12 };
  static int16_t stage34_offset_shift[] = { 2048, 12 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage12_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage3_weights, stage34_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage4_weights, stage34_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
split_schro (int16_t *a, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_split_desl93 (a, n/2);
      break;
    case SCHRO_WAVELET_5_3:
      schro_split_53 (a, n/2);
      break;
    case SCHRO_WAVELET_13_5:
      schro_split_135 (a, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_split_haar (a, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_fidelity (a, n/2);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_split_daub97(a,n/2);
      break;
  }
}




void
schro_synth_desl93 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 1, 2 };
  static int16_t stage2_offset_shift[] = { 8, 4 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

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

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

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

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_synth_53 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -1, -1 };
  static int16_t stage2_weights[] = { 1, 1 };
  static int16_t stage1_offset_shift[] = { 1, 2 };
  static int16_t stage2_offset_shift[] = { 1, 1 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

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

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage2_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_synth_135 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static int16_t stage1_offset_shift[] = { 15, 5 };
  static int16_t stage2_offset_shift[] = { 8, 4 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

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

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  lo[-1] = lo[0];
  lo[-2] = lo[1];
  lo[n] = lo[n-2];
  oil_mas4_add_s16 (hi, hi, lo - 2, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  oil_mas4_add_s16 (lo, lo, hi-1, stage2_weights, stage2_offset_shift, n);

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_synth_haar (int16_t *a, int n)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int i;

#if 0
      for(i=0;i<n;i+=2){
        a[i] -= (a[i+1] + 1)>>1;
      }
      for(i=0;i<n;i+=2){
        a[i+1] += a[i];
      }
#endif

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

  for(i=0;i<n;i++) {
    hi[i] -= ((lo[i] + 1)>>1);
  }
  for(i=0;i<n;i++) {
    lo[i] += hi[i];
  }

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_synth_fidelity (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static int16_t stage2_weights[] = { 8, -21, 46, -161, -161, 46, -21, 8 };
  static int16_t stage1_offset_shift[] = { 128, 8 };
  static int16_t stage2_offset_shift[] = { 127, 8 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

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

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

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

  oil_interleave2_s16 (a, hi, lo, n);
}

void
schro_synth_daub97 (int16_t *a, int n)
{
  static int16_t stage1_weights[] = { -1817, -1817 };
  static int16_t stage2_weights[] = { -3616, -3616 };
  static int16_t stage3_weights[] = { 217, 217 };
  static int16_t stage4_weights[] = { 6497, 6497 };
  static int16_t stage12_offset_shift[] = { 2047, 12 };
  static int16_t stage34_offset_shift[] = { 2048, 12 };
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;

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

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  oil_deinterleave2_s16 (hi, lo, a, n);

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

  oil_interleave2_s16 (a, hi, lo, n);
}

void
synth_schro (int16_t *a, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_synth_desl93 (a, n/2);
      break;
    case SCHRO_WAVELET_5_3:
      schro_synth_53 (a, n/2);
      break;
    case SCHRO_WAVELET_13_5:
      schro_synth_135 (a, n/2);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_synth_haar (a, n/2);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_fidelity (a, n/2);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_synth_daub97(a,n/2);
      break;
  }
}

