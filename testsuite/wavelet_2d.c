
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

int fail = 0;

typedef struct _Picture Picture;
struct _Picture {
  int16_t *data;
  int stride;
  int width;
  int height;
};

void dump(Picture *p);
void dump_cmp(Picture *p, Picture *ref);
void iwt_ref(Picture *p, int filter);
void iiwt_ref(Picture *p, int filter);
void iwt_test(Picture *p, int filter);
void iiwt_test(Picture *p, int filter);

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

void
gen_hline (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (j==(p->height/2))*100;
    }
  }
}

void
gen_point (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = (j==(p->height/2) && i==(p->width/2))*100;
    }
  }
}

void
gen_random (Picture *p)
{
  int i;
  int j;
  int16_t *data;

  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    for(i=0;i<p->width;i++){
      data[i] = oil_rand_u8()&0xf;
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
  { "checkerboard", gen_checkerboard },
  { "hline", gen_hline },
  { "point", gen_point },
  { "random", gen_random }
};


Picture *
picture_new (int width, int height)
{
  Picture *p;

  p = malloc(sizeof(Picture));
  p->data = malloc (width * height * sizeof(int16_t));
  p->stride = width * sizeof(int16_t);
  p->width = width;
  p->height = height;

  return p;
}

void
picture_free (Picture *p)
{
  free(p->data);
  free(p);
}

void
picture_copy (Picture *dest, Picture *src)
{
  int i;
  int j;
  int16_t *d, *s;

  for(j=0;j<dest->height;j++){
    d = OFFSET(dest->data,j*dest->stride);
    s = OFFSET(src->data,j*src->stride);
    for(i=0;i<dest->width;i++){
      d[i] = s[i];
    }
  }
}

int
picture_compare (Picture *dest, Picture *src)
{
  int i;
  int j;
  int16_t *d, *s;

  for(j=0;j<dest->height;j++){
    d = OFFSET(dest->data,j*dest->stride);
    s = OFFSET(src->data,j*src->stride);
    for(i=0;i<dest->width;i++){
      if (d[i] != s[i]) return 0;
    }
  }
  return 1;
}

void
local_test (int filter)
{
  int i;
  Picture *p;
  Picture *ref;

  p = picture_new (20, 20);
  ref = picture_new (20, 20);

  for(i=0;i<sizeof(generators)/sizeof(generators[0]);i++){
    printf("  test \"%s\":\n", generators[i].name);
    generators[i].create(ref);
    picture_copy (p, ref);
    dump(ref);
    iwt_ref(ref,filter);
    iwt_test(p,filter);
    dump(ref);
    dump_cmp(p, ref);
    if (!picture_compare(p, ref)) {
      printf("  FAILED\n");
      fail = 1;
    }
  }
  picture_free(p);
  picture_free(ref);
}

void
random_test (int filter)
{
  int i;
  Picture *p;
  Picture *ref;

  p = picture_new (20, 20);
  ref = picture_new (20, 20);

  printf("  Random tests:\n");
  for(i=0;i<100;i++){
    gen_random(ref);
    picture_copy (p, ref);
    iwt_ref(ref,filter);
    iwt_test(p,filter);
    if (!picture_compare(p, ref)) {
      printf("  FAILED\n");
      dump_cmp(p, ref);
      fail = 1;
      goto out;
    }
  }
  printf("  OK\n");
out:
  picture_free(p);
  picture_free(ref);
}

void
inv_test (int filter)
{
  int i;
  Picture *p;
  Picture *ref;

  p = picture_new (20, 20);
  ref = picture_new (20, 20);

  for(i=0;i<sizeof(generators)/sizeof(generators[0]);i++){
    printf("  test \"%s\":\n", generators[i].name);
    generators[i].create(ref);
    iwt_ref(ref,filter);
    picture_copy (p, ref);
    iiwt_ref(ref,filter);
    iiwt_test(p,filter);
    dump(ref);
    dump_cmp(p, ref);
    if (!picture_compare(p, ref)) {
      printf("  FAILED\n");
      fail = 1;
    }
  }
  picture_free(p);
  picture_free(ref);
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
    inv_test(filter);
  }

  return fail;
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

void dump_cmp(Picture *p, Picture *ref)
{
  int i;
  int j;
  int16_t *data;
  int16_t *rdata;

  printf("-----\n");
  for(j=0;j<p->height;j++){
    data = OFFSET(p->data,j*p->stride);
    rdata = OFFSET(ref->data,j*p->stride);
    for(i=0;i<p->width;i++){
      if (data[i] == rdata[i]) {
        printf("%3d ", data[i]);
      } else {
        printf("\033[00;01;37;41m%3d\033[00m ", data[i]);
      }
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
    oil_interleave2_s16 (tmp3, hi, lo, p->height/2);
    copy(data, p->stride, tmp3, sizeof(int16_t), p->height);
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
    copy(tmp3, sizeof(int16_t), data, p->stride, p->height);
    oil_deinterleave2_s16 (hi, lo, tmp3, p->height/2);
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
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      schro_split_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_split_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      schro_split_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_split_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_split_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_split_ext_daub97(hi, lo, n);
      break;
  }
}

void
schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter)
{
  switch (filter) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      schro_synth_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_synth_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      schro_synth_ext_135 (hi, lo, n);
      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
      schro_synth_ext_haar (hi, lo, n);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_synth_ext_fidelity (hi, lo, n);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_synth_ext_daub97(hi, lo, n);
      break;
  }
}

void iwt_test(Picture *p, int filter)
{
  int16_t *tmp;

  tmp = malloc((p->width + 32)*sizeof(int16_t));

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      schro_iwt_desl_9_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iwt_5_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      schro_iwt_13_5 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iwt_haar0 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iwt_haar1 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iwt_fidelity (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_iwt_daub_9_7(p->data, p->stride, p->width, p->height, tmp);
      break;
  }

  free(tmp);
}

void iiwt_test(Picture *p, int filter)
{
  int16_t *tmp;

  tmp = malloc((p->width + 32)*sizeof(int16_t));

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      schro_iiwt_desl_9_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iiwt_5_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      schro_iiwt_13_5 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_0:
      schro_iiwt_haar0 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_HAAR_1:
      schro_iiwt_haar1 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_FIDELITY:
      schro_iiwt_fidelity (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      schro_iiwt_daub_9_7(p->data, p->stride, p->width, p->height, tmp);
      break;
  }

  free(tmp);
}





