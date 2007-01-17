
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schropredict.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>
#include <schroedinger/schrooil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <liboil/liboilrandom.h>
#include <liboil/liboil.h>

int frame_compare (SchroFrame *dest, SchroFrame *src);
void frame_dump (SchroFrame *dest, SchroFrame *src);
void frame_create_test_pattern(SchroFrame *frame, int type);
void upsample_line (uint8_t *dest, uint8_t *src, int n);
void ref_h_upsample (SchroFrame *dest, SchroFrame *src);
void ref_v_upsample (SchroFrame *dest, SchroFrame *src);
void test_h_upsample (SchroFrame *dest, SchroFrame *src);
void test_v_upsample (SchroFrame *dest, SchroFrame *src);

int
main (int argc, char *argv[])
{
  SchroFrame *frame;
  SchroFrame *frame_ref;
  SchroFrame *frame_test;
  int width, height;

  width = 20;
  height = 16;

  schro_init();

  frame = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8, width, height, width/2, height/2);
  frame_ref = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8, width, height, width/2, height/2);
  frame_test = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8, width, height, width/2, height/2);

  frame_create_test_pattern(frame,0);

  //ref_h_upsample (frame_ref, frame);
  //schro_frame_h_upsample (frame_test, frame);
  ref_v_upsample (frame_ref, frame);
  //schro_frame_v_upsample (frame_test, frame);
  test_v_upsample (frame_test, frame);

  frame_dump (frame_test, frame_ref);
  //frame_dump (frame_ref, frame_ref);

  if (!frame_compare (frame_ref, frame_test)) {
    printf("horiz upsample not equal\n");
  }

  //schro_frame_v_upsample (frame_ref, frame);

  return 0;
}


void
upsample_line (uint8_t *dest, uint8_t *src, int n)
{
  int i;
  int x;

  for(i=0;i<n-1;i++){
    x  =   3 * src[CLAMP(i-4,0,n-1)];
    x += -11 * src[CLAMP(i-3,0,n-1)];
    x +=  25 * src[CLAMP(i-2,0,n-1)];
    x += -56 * src[CLAMP(i-1,0,n-1)];
    x += 167 * src[CLAMP(i,0,n-1)];
    x += 167 * src[CLAMP(i+1,0,n-1)];
    x += -56 * src[CLAMP(i+2,0,n-1)];
    x +=  25 * src[CLAMP(i+3,0,n-1)];
    x += -11 * src[CLAMP(i+4,0,n-1)];
    x +=   3 * src[CLAMP(i+5,0,n-1)];
    x += 128;
    x >>= 8;
    dest[i] = CLAMP(x, 0, 255);
  }
  dest[i] = src[i];
}

void
ref_h_upsample (SchroFrame *dest, SchroFrame *src)
{
  int j,k;
  uint8_t *d;
  uint8_t *s;

  for(k=0;k<3;k++){
    for(j=0;j<dest->components[k].height;j++){
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      upsample_line (d, s, dest->components[k].width);
    }
  }
}

void
ref_v_upsample (SchroFrame *dest, SchroFrame *src)
{
  int i,j,k;
  uint8_t *d;
  uint8_t *s;
  uint8_t tmp1[100];
  uint8_t tmp2[100];

  for(k=0;k<3;k++){
    for(i=0;i<dest->components[k].width;i++){
      for(j=0;j<dest->components[k].height;j++){
        s = OFFSET(src->components[k].data, src->components[k].stride * j + i);
        tmp1[j] = *s;
      }
      upsample_line (tmp2, tmp1, dest->components[k].height);
      for(j=0;j<dest->components[k].height;j++){
        d = OFFSET(dest->components[k].data, dest->components[k].stride * j + i);
        *d = tmp2[j];
      }
    }
  }
}

int
frame_compare (SchroFrame *dest, SchroFrame *src)
{
  int i,j,k;
  uint8_t *d;
  uint8_t *s;

  for(k=0;k<3;k++){
    for(j=0;j<dest->components[k].height;j++){
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      for(i=0;i<dest->components[k].width;i++){
        if (d[i] != s[i]) return FALSE;
      }
    }
  }
  return TRUE;
}

void frame_dump(SchroFrame *p, SchroFrame *ref)
{
  int i;
  int j;
  int k;
  uint8_t *data;
  uint8_t *rdata;

  for(k=0;k<3;k++){
    printf("-----\n");
    for(j=0;j<p->components[k].height;j++){
      data = OFFSET(p->components[k].data,j*p->components[k].stride);
      rdata = OFFSET(ref->components[k].data,j*p->components[k].stride);
      for(i=0;i<p->components[k].width;i++){
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
}

void
frame_create_test_pattern(SchroFrame *frame, int type)
{
  int i,j,k;
  uint8_t *data;

  for(k=0;k<3;k++){
    for(j=0;j<frame->components[k].height;j++){
      data = OFFSET(frame->components[k].data,j*frame->components[k].stride);
      for(i=0;i<frame->components[k].width;i++) {
        //data[i] = 100;
        //data[i] = i*10;
        //data[i] = j*10;
        data[i] = oil_rand_u8();
      }
    }
  }
}


void
test_upsample_line (uint8_t *dest, uint8_t *src, int n)
{
  static const int16_t weights[10] = { 3, -11, 25, -56, 167, 167, -56, 25, -11, 3 };
  static const int16_t offset_shift[2] = { 128, 8 };
  uint8_t tmp[16];
  int i;

  tmp[0] = src[0];
  tmp[1] = src[0];
  tmp[2] = src[0];
  tmp[3] = src[0];
  for(i=0;i<10;i++){
    tmp[4 + i] = src[i];
  }
  oil_mas10_across_u8 (dest, tmp, weights, offset_shift, 4);

  oil_mas10_across_u8 (dest + 4, src, weights, offset_shift, n - 9);

  for(i=0;i<9;i++){
    tmp[i] = src[n - 9 + i];
  }
  tmp[9] = src[n-1];
  tmp[10] = src[n-1];
  tmp[11] = src[n-1];
  tmp[12] = src[n-1];
  tmp[13] = src[n-1];

  oil_mas10_across_u8 (dest + n - 5, tmp, weights, offset_shift, 4);

  dest[n-1] = src[n - 1];
}

void
test_h_upsample (SchroFrame *dest, SchroFrame *src)
{
  int j,k;
  uint8_t *d;
  uint8_t *s;

  for(k=0;k<3;k++){
    for(j=0;j<dest->components[k].height;j++){
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      test_upsample_line (d, s, dest->components[k].width);
    }
  }
}

void
test_v_upsample (SchroFrame *dest, SchroFrame *src)
{
  static const int16_t offset_shift[2] = { 128, 8 };
  int j,k;
  uint8_t *d;
  uint8_t *s;

  for(k=0;k<3;k++){
    static const int16_t weights[][10] = {
      { 128, 167, -56, 25, -11, 3, 0, 0, 0, 0 },
      { -39, 167, 167, -56, 25, -11, 3, 0, 0, 0 },
      { 17, -56, 167, 167, -56, 25, -11, 3, 0, 0 },
      { -8, 25, -56, 167, 167, -56, 25, -11, 3, 0 },
      { 3, -11, 25, -56, 167, 167, -56, 25, -11, 3 },
      { 0, 3, -11, 25, -56, 167, 167, -56, 25, -8 },
      { 0, 0, 3, -11, 25, -56, 167, 167, -56, 17 },
      { 0, 0, 0, 3, -11, 25, -56, 167, 167, -39 },
      { 0, 0, 0, 0, 3, -11, 25, -56, 167, 128 }
    };
    if (dest->components[k].height < 10) {
      int i,l;
      for(j=0;j<dest->components[k].height-1;j++){
        d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
        s = src->components[k].data;
        for(i=0;i<dest->components[k].width;i++){
          int x;

          x = offset_shift[0];
          for(l=0;l<10;l++){
            x += weights[4][l] *
              s[CLAMP(j+l-4,0,dest->components[k].height-1)*
              src->components[k].stride + i];

          }
          x >>= offset_shift[1];
          d[i] = CLAMP(x,0,255);
        }
      }
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      memcpy(d, s, dest->components[k].width);
    } else {
      for(j=0;j<4;j++){
        d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
        s = src->components[k].data;
        oil_mas10_u8 (d, s, src->components[k].stride, weights[j], offset_shift,
            dest->components[k].width);
      }
      for(;j<dest->components[k].height-5;j++){
        d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
        s = OFFSET(src->components[k].data, src->components[k].stride * (j-4));
        oil_mas10_u8 (d, s, src->components[k].stride, weights[4], offset_shift,
            dest->components[k].width);
      }
      for(;j<dest->components[k].height-1;j++){
        d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
        s = OFFSET(src->components[k].data, src->components[k].stride * (dest->components[k].height-10));
        oil_mas10_u8 (d, s, src->components[k].stride,
            weights[j + 10 - dest->components[k].height], offset_shift,
            dest->components[k].width);
      }
      d = OFFSET(dest->components[k].data, dest->components[k].stride * j);
      s = OFFSET(src->components[k].data, src->components[k].stride * j);
      memcpy(d, s, dest->components[k].width);
    }
  }
}

