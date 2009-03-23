
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "common.h"

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

int fail = 0;

void iwt_ref(SchroFrameData *p, int filter);
void iiwt_ref(SchroFrameData *p, int filter);
void iwt_test(SchroFrameData *p, int filter);
void iiwt_test(SchroFrameData *p, int filter);

void schro_split_ext (int16_t *hi, int16_t *lo, int n, int filter);
void schro_synth_ext (int16_t *hi, int16_t *lo, int n, int filter);

void
fwd_test (int filter, int width, int height)
{
  int i;
  SchroFrame *test;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (fd_ref, name, i);
    printf("  forward test \"%s\":\n", name);
    fflush(stdout);

    schro_frame_convert (test, ref);
    iwt_ref(fd_ref,filter);
    iwt_test(fd_test,filter);
    if (!frame_data_compare(fd_test, fd_ref)) { 
      frame_data_dump (fd_test, fd_ref);
      fail = TRUE;
    }
  }
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
inv_test (int filter, int width, int height)
{
  int i;
  SchroFrame *test;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (fd_ref, name, i);
    printf("  reverse test \"%s\":\n", name);
    fflush(stdout);

    iwt_ref(fd_ref,filter);
    schro_frame_convert (test, ref);
    iiwt_ref(fd_ref,filter);
    iiwt_test(fd_test,filter);
    if (!frame_data_compare(fd_test, fd_ref)) { 
      frame_data_dump (fd_test, fd_ref);
      fail = TRUE;
    }
  }
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
fwd_random_test (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_ref, name, 0);
  printf("  forward test \"%s\":\n", name);
  fflush(stdout);

  schro_frame_convert (test, ref);
  iwt_ref(fd_ref,filter);
  iwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    frame_data_dump (fd_test, fd_ref);
    fail = TRUE;
  }
  
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
inv_random_test (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_ref, name, 0);
  printf("  reverse test \"%s\":\n", name);
  fflush(stdout);

  iwt_ref(fd_ref,filter);
  schro_frame_convert (test, ref);
  iiwt_ref(fd_ref,filter);
  iiwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    frame_data_dump (fd_test, fd_ref);
    fail = TRUE;
  }
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

int
main (int argc, char *argv[])
{
  int filter;
  int width;
  int height;

  schro_init();

  for(filter=0;filter<=SCHRO_WAVELET_DAUBECHIES_9_7;filter++){
    printf("Filter %d:\n", filter);
    fwd_test(filter, 20, 20);
    inv_test(filter, 20, 20);
  }

  for(width = 4; width <= 40; width+=2) {
    for(height = 4; height <= 40; height+=2) {
      printf("Size %dx%d:\n", width, height);
      for(filter=0;filter<=SCHRO_WAVELET_DAUBECHIES_9_7;filter++){
        printf("  filter %d:\n", filter);
        if (filter == SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7 && (width < 0 || height <= 6)) {
          continue;
        }
        if (filter == SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7 && (width < 0 || height <= 6)) {
          continue;
        }
        if (filter == SCHRO_WAVELET_FIDELITY && (width < 16 || height < 16)) {
          continue;
        }
        fwd_random_test(filter, width, height);
        inv_random_test(filter, width, height);
      }
    }
  }

  return fail;
}

void
rshift (SchroFrameData *p, int n)
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
lshift (SchroFrameData *p, int n)
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

void iwt_ref(SchroFrameData *p, int filter)
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

void iiwt_ref(SchroFrameData *p, int filter)
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
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_split_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_split_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
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
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_synth_ext_desl93 (hi, lo, n);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_synth_ext_53 (hi, lo, n);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
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

void iwt_test(SchroFrameData *p, int filter)
{
  int16_t *tmp;

  tmp = malloc((p->width + 32)*sizeof(int16_t));

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_iwt_desl_9_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iwt_5_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
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

void iiwt_test(SchroFrameData *p, int filter)
{
  int16_t *tmp;

  tmp = malloc((p->width + 32)*sizeof(int16_t));

  switch (filter) {
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7:
      schro_iiwt_desl_9_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      schro_iiwt_5_3 (p->data, p->stride, p->width, p->height, tmp);
      break;
    case SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7:
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





