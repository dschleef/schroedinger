
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <liboil/liboil.h>

int filtershift[] = { 1, 1, 1, 0, 1, 2, 0, 1 };

typedef struct _Picture Picture;
struct _Picture {
  int16_t *data;
  int stride;
  int width;
  int height;
};

#define OFFSET(ptr,x) (void *)(((uint8_t *)ptr) + (x))

void dump(Picture *p);
void iwt_ref(Picture *p, int filter);
void iiwt_ref(Picture *p, int filter);

void schro_split_ext (int16_t *hi, int16_t *lo, int n, int filter);
void schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter);

void
gen_const (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = 100;
    }
  }
}

void
gen_vramp (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (100*j+p->height/2)/p->height;
    }
  }
}

void
gen_hramp (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (100*i+p->width/2)/p->width;
    }
  }
}

void
gen_valt (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (j&1)*100;
    }
  }
}

void
gen_halt (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (i&1)*100;
    }
  }
}

void
gen_checkerboard (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = ((i+j)&1)*100;
    }
  }
}


typedef struct _Generator Generator;
struct _Generator {
  char *name;
  void (*create)(Picture *p);
};

Generator generators[] = {
  { "constant", gen_const },
  { "hramp", gen_hramp },
  { "vramp", gen_vramp },
  { "halt", gen_halt },
  { "valt", gen_valt },
  { "checkerboard", gen_checkerboard }
};


int16_t tmp[20*20];

void
local_test (int filter)
{
  int $;
  Picture pict;
  Picture *p = &pict;

  p->data = tmp;
  p->stride = 40;
  p->width = 20;
  p->height = 20;

  for($=0;$<sizeof(generators)/sizeof(generators[0]);$++){
    printf("  test \"%s\":\n", generators[$].name);
    generators[$].create(p);
    dump(p);
    iwt_ref(p,filter);
    dump(p);
    iiwt_ref(p,filter);
    dump(p);
  }
}

int
main (int argc, char *argv[])
{
  int filter;

  schro_init();
    
  for(filter=0;filter<=SCHRO_WAVELET_DAUB_9_7;filter++){
    printf("Filter %d:\n", filter);
    local_test(filter);
    //random_test(filter);
  }

  return 0;
}

void dump(Picture *p)
{
  int i;
  int j;
  int16_t *data;

  printf("-----\n");
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      printf("%3d ", data[i]);
    }
    printf("\n");
  }
  printf("-----\n");
}

void
copy (int16_t *d, int ds, int16_t *s, int ss, int n)
{
  int i;
  int16_t *xd, *xs;
  for(i=0;i<n;i++){
    xd = OFFSET(d,ds * i);
    xs = OFFSET(s,ss * i);
    *xd = *xs;
  }
}

void
rshift (Picture *p, int n)
{
  int i;
  int j;
  int16_t *data;

  if (n==0) return;
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] >>= n;
    }
  }
}

void
lshift (Picture *p, int n)
{
  int i;
  int j;
  int16_t *data;

  if (n==0) return;
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] <<= n;
    }
  }
}

void iwt_ref(Picture *p, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t tmp3[100];
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  lshift(p, filtershift[filter]);

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    oil_deinterleave2_s16 (hi, lo, data, p->width/2);
    schro_split_ext (hi, lo, p->width/2, filter);
    copy(data, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
    copy(data + p->width/2, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
  }

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(tmp3, sizeof(int16_t), data, p->stride, p->height);
    oil_deinterleave2_s16 (hi, lo, tmp3, p->height/2);
    schro_split_ext (hi, lo, p->height/2, filter);
    copy(data, p->stride, hi, sizeof(int16_t), p->height/2);
    copy(OFFSET(data, p->height/2*p->stride), p->stride, lo, sizeof(int16_t),
        p->height/2);
  }

}

void iiwt_ref(Picture *p, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t tmp3[100];
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(hi, sizeof(int16_t), data, p->stride, p->height/2);
    copy(lo, sizeof(int16_t), OFFSET(data, p->height/2*p->stride), p->stride,
        p->height/2);
    schro_synth_ext (hi, lo, p->height/2, filter);
    oil_interleave2_s16 (tmp3, hi, lo, p->height/2);
    copy(data, p->stride, tmp3, sizeof(int16_t), p->height);
  }

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    copy(hi, sizeof(int16_t), data, sizeof(int16_t), p->width/2);
    copy(lo, sizeof(int16_t), data + p->width/2, sizeof(int16_t), p->width/2);
    schro_synth_ext (hi, lo, p->width/2, filter);
    oil_interleave2_s16 (data, hi, lo, p->width/2);
  }

  rshift(p, filtershift[filter]);
}

void
schro_split_ext (int16_t *hi, int16_t *lo, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_split_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_split_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_split_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_split_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_split_ext_daub97(hi, lo, n);
      break;
  }
}

void
schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESL_9_3:
      schro_synth_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_synth_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_synth_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      schro_synth_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUB_9_7:
      schro_synth_ext_daub97(hi, lo, n);
      break;
  }
}


