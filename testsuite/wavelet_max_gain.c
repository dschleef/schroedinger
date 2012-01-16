
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
schro_frame_data_get_wavelet_fd (SchroFrameData *dest, SchroFrameData *src,
    int level, int pos)
{
  dest->format = src->format;
  dest->data = src->data;
  dest->width = src->width >> level;
  dest->height = src->height >> level;
  dest->stride = src->stride << level;

  if (pos & 1) {
    dest->data = SCHRO_OFFSET(dest->data, sizeof(int32_t) * dest->width);
  }
  if (pos & 2) {
    dest->data = SCHRO_OFFSET(dest->data, dest->stride>>1);
  }
}

void
schro_frame_data_clear (SchroFrameData *fd)
{
  int j;

  for(j=0;j<fd->height;j++){
    int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    memset (line, 0, sizeof(int32_t) * fd->width);
  }
}

void
schro_frame_data_set_center (SchroFrameData *fd)
{
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, fd->height/2-1);
  line[fd->width/2-1] = 256;
}

int
schro_frame_data_get_center (SchroFrameData *fd)
{
  int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, fd->height/2-1);
  return line[fd->width/2-1];
}

void
schro_frame_data_saturate (SchroFrameData *fd)
{
  int j;
  int i;

  for(j=0;j<fd->height;j++){
    int32_t *line = SCHRO_FRAME_DATA_GET_LINE (fd, j);
    for(i=0;i<fd->width;i++){
      if (line[i]<0) line[i]=-16;
      else if (line[i]>0) line[i]=16;
    }
  }

}

int
get_shift (int n)
{
  int i = 0;
  n--;
  while (n > 0) {
    i++;
    n >>= 1;
  }
  return i;
}

void
gain_test_s32 (int filter)
{
  SchroFrame *test;
  SchroFrame *orig;
  SchroFrame *ref;
  SchroFrameData *fd_orig;
  //SchroFrameData *fd_test;
  SchroFrameData *fd_ref;
  SchroFrameData fd_trans;
  int max;
  int level;
  int pos;
  int i;
  int width = 256;
  int height = 256;
  int level_max;

  printf("  Gain table filter=%d\n", filter);

  for(level=1;level<=6;level++){
    level_max = 0;
    for(pos=0;pos<4;pos++){

      orig = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S32_444,
          width, height);
      fd_orig = orig->components + 0;
      test = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S32_444,
          width, height);
      //fd_test = test->components + 0;
      ref = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S32_444,
          width, height);
      fd_ref = ref->components + 0;

      schro_frame_data_clear (fd_orig);
      schro_frame_data_get_wavelet_fd (&fd_trans, fd_orig, level, pos);
      schro_frame_data_set_center (&fd_trans);
      schro_frame_convert (ref, orig);

      fflush(stdout);

      for(i=0;i<level;i++) {
        schro_frame_data_get_wavelet_fd (&fd_trans, fd_orig, level - 1 - i, 0);
        iiwt_test(&fd_trans, filter);
      }

      if (0) frame_data_dump (fd_orig, fd_ref);

      schro_frame_data_saturate (fd_orig);

      for(i=0;i<level;i++) {
        schro_frame_data_get_wavelet_fd (&fd_trans, fd_orig, i, 0);
        iwt_test(&fd_trans, filter);
      }

      if (0) frame_data_dump (fd_orig, fd_ref);

      schro_frame_data_get_wavelet_fd (&fd_trans, fd_orig, level, pos);
      max = schro_frame_data_get_center (&fd_trans);
      level_max = MAX(max,level_max);

      //printf("%d %d: max %d (%g)\n", level, pos, max, max/16.0);

    }
    printf("%d: level_max %d shift %d\n", level, level_max, get_shift(level_max) - 4);
  }
  

  schro_frame_unref (orig);
  schro_frame_unref (test);
  schro_frame_unref (ref);
}


int
main (int argc, char *argv[])
{
  int filter;

  schro_init();

  for(filter=0;filter<=SCHRO_WAVELET_DAUBECHIES_9_7;filter++){
    gain_test_s32 (filter);
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
  int16_t tmp1[256+16], *hi;
  int16_t tmp2[256+16], *lo;
  int16_t tmp3[256+16], *tmpbuf;
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
  int16_t tmp1[256+16], *hi;
  int16_t tmp2[256+16], *lo;
  int16_t tmp3[256+16], *tmpbuf;
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





