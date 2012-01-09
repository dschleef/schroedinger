
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

int filtershift[] = { 1, 1, 1, 0, 1, 0, 1 };

int fail = 0;
int verbose = 1;

void iwt_ref(SchroFrameData *p, int filter);
void iiwt_ref(SchroFrameData *p, int filter);
void iwt_test(SchroFrameData *p, int filter);
void iiwt_test(SchroFrameData *p, int filter);


void
fwd_test (int filter, int width, int height)
{
  int i;
  SchroFrame *test;
  SchroFrame *ref;
  SchroFrame *orig;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  SchroFrameData *fd_orig;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (fd_orig, name, i);
    printf("  forward test \"%s\":\n", name);
    fflush(stdout);

    schro_frame_convert (ref, orig);
    schro_frame_convert (test, orig);
    iwt_ref(fd_ref,filter);
    iwt_test(fd_test,filter);
    if (!frame_data_compare(fd_test, fd_ref)) { 
      printf("  failed\n");
      if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
      fail = TRUE;
    }
  }
  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
inv_test (int filter, int width, int height)
{
  int i;
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_orig;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  for(i=0;i<test_pattern_get_n_generators();i++){
    test_pattern_generate (fd_orig, name, i);
    printf("  reverse test \"%s\":\n", name);
    fflush(stdout);

    iwt_ref(fd_orig,filter);
    schro_frame_convert (test, orig);
    schro_frame_convert (ref, orig);
    iiwt_ref(fd_ref,filter);
    iiwt_test(fd_test,filter);
    if (!frame_data_compare(fd_test, fd_ref)) { 
      printf("  failed\n");
      if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
      fail = TRUE;
    }
  }
  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
fwd_random_test (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_orig;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_orig, name, 0);
  printf("  forward test \"%s\":\n", name);
  fflush(stdout);

  schro_frame_convert (ref, orig);
  schro_frame_convert (test, orig);
  iwt_ref(fd_ref,filter);
  iwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    printf("  failed\n");
    if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
    fail = TRUE;
  }
  
  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
inv_random_test (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_orig;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_orig, name, 0);
  printf("  reverse test \"%s\":\n", name);
  fflush(stdout);

  iwt_ref(fd_orig,filter);
  schro_frame_convert (test, orig);
  schro_frame_convert (ref, orig);
  iiwt_ref(fd_ref,filter);
  iiwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    printf("  failed\n");
    if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
    fail = TRUE;
  }
  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
fwd_random_test_s32 (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_orig;
  SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S32_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_orig, name, 0);
  printf("  forward test 32-bit \"%s\":\n", name);
  fflush(stdout);

  schro_frame_convert (ref, orig);
  schro_frame_convert (test, orig);
  iwt_ref(fd_ref,filter);
  iwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    printf("  failed\n");
    if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
    fail = TRUE;
  }
  
  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}

void
inv_random_test_s32 (int filter, int width, int height)
{
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_test;
  SchroFrameData *fd_orig;
  SchroFrameData *fd_ref;
  char name[TEST_PATTERN_NAME_SIZE] = { 0 };

  orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_orig = orig->components + 0;
  test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S32_444,
      width, height);
  fd_test = test->components + 0;
  ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444,
      width, height);
  fd_ref = ref->components + 0;

  test_pattern_generate (fd_orig, name, 0);
  printf("  reverse test 32-bit \"%s\":\n", name);
  fflush(stdout);

  iwt_ref(fd_orig,filter);
  schro_frame_convert (test, orig);
  schro_frame_convert (ref, orig);
  iiwt_ref(fd_ref,filter);
  iiwt_test(fd_test,filter);
  if (!frame_data_compare(fd_test, fd_ref)) { 
    printf("  failed\n");
    if (verbose) frame_data_dump_full (fd_test, fd_ref, fd_orig);
    fail = TRUE;
  }
  schro_frame_unref (orig);
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

  for(width = 2; width <= 40; width+=2) {
    for(height = 2; height <= 40; height+=2) {
      printf("Size %dx%d:\n", width, height);
      for(filter=0;filter<=SCHRO_WAVELET_DAUBECHIES_9_7;filter++){
        printf("  filter %d:\n", filter);
        fwd_random_test(filter, width, height);
        inv_random_test(filter, width, height);
        fwd_random_test_s32(filter, width, height);
        inv_random_test_s32(filter, width, height);
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
  int16_t tmp3[100], *tmpbuf;
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;
  tmpbuf = tmp3 + 8;

  lshift(p, filtershift[filter]);

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    copy(tmpbuf, sizeof(int16_t), data, sizeof(int16_t), p->width);
    split (tmpbuf, p->width, filter);
    orc_deinterleave2_s16 (lo, hi, tmpbuf, p->width/2);
    copy(data, sizeof(int16_t), lo, sizeof(int16_t), p->width/2);
    copy(data + p->width/2, sizeof(int16_t), hi, sizeof(int16_t), p->width/2);
  }

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(tmpbuf, sizeof(int16_t), data, p->stride, p->height);
    split (tmpbuf, p->height, filter);
    copy(data, p->stride, tmpbuf, sizeof(int16_t), p->height);
  }

}

void iiwt_ref(SchroFrameData *p, int filter)
{
  int16_t tmp1[100], *hi;
  int16_t tmp2[100], *lo;
  int16_t tmp3[100], *tmpbuf;
  int16_t *data;
  int i;

  hi = tmp1 + 4;
  lo = tmp2 + 4;
  tmpbuf = tmp3 + 8;

  for(i=0;i<p->width;i++){
    data = OFFSET(p->data,i*sizeof(int16_t));
    copy(tmpbuf, sizeof(int16_t), data, p->stride, p->height);
    synth (tmpbuf, p->height, filter);
    copy(data, p->stride, tmpbuf, sizeof(int16_t), p->height);
  }

  for(i=0;i<p->height;i++){
    data = OFFSET(p->data,i*p->stride);
    copy(hi, sizeof(int16_t), data, sizeof(int16_t), p->width/2);
    copy(lo, sizeof(int16_t), data + p->width/2, sizeof(int16_t), p->width/2);
    orc_interleave2_s16 (tmpbuf, hi, lo, p->width/2);
    synth (tmpbuf, p->width, filter);
    copy(data, sizeof(int16_t), tmpbuf, sizeof(int16_t), p->width);
  }

  rshift(p, filtershift[filter]);
}

void iwt_test(SchroFrameData *p, int filter)
{
  int16_t *tmp;

  tmp = malloc(((p->width+8)*2)*sizeof(int32_t));

  schro_wavelet_transform_2d (p, filter, tmp);

  free(tmp);
}

void iiwt_test(SchroFrameData *p, int filter)
{
  int16_t *tmp;

  tmp = malloc(((p->width+8)*2)*sizeof(int32_t));

  schro_wavelet_inverse_transform_2d (p, p, filter, tmp);

  free(tmp);
}





