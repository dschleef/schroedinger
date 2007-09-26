
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

void ref_frame_downsample (SchroFrame *dest, SchroFrame *src);

int frame_compare (SchroFrame *dest, SchroFrame *src);
void frame_dump (SchroFrame *dest, SchroFrame *src);
void frame_create_test_pattern(SchroFrame *frame, int type);

void
test (int width, int height)
{
  SchroFrame *frame;
  SchroFrame *frame_ref;
  SchroFrame *frame_test;
  int ok;

  SCHRO_WARNING("size %dx%d", width, height);

  frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8_420, width, height);
  frame_ref = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8_420,
      ROUND_UP_SHIFT(width, 1), ROUND_UP_SHIFT(height, 1));
  frame_test = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8_420,
      ROUND_UP_SHIFT(width, 1), ROUND_UP_SHIFT(height, 1));

  frame_create_test_pattern (frame, 0);

  ref_frame_downsample (frame_ref, frame);
  schro_frame_downsample (frame_test, frame);

  ok = frame_compare (frame_ref, frame_test);
  if (!ok) {
    SCHRO_ERROR("compare failed (size=%dx%d)", width, height);
    frame_dump (frame_test, frame_ref);
    frame_dump (frame_ref, frame_test);
    exit(1);
  }

  schro_frame_unref (frame_ref);
  schro_frame_unref (frame_test);
  schro_frame_unref (frame);
}

int
main (int argc, char *argv[])
{
  int i;
  int width;
  int height;

  schro_init();

  for(i=0;i<1;i++){
    for(width=1;width<40;width++){
      for(height=1;height<40;height++){
        test (width, height);
      }
    }
  }

  printf("PASS\n");
  return 0;
}


int
component_get (SchroFrameComponent *src, int i, int j)
{
  uint8_t *data;

  i = CLAMP(i,0,src->width-1);
  j = CLAMP(j,0,src->height-1);
  data = OFFSET(src->data, j*src->stride);

  return data[i];
}

void
ref_frame_component_downsample (SchroFrameComponent *dest,
    SchroFrameComponent *src)
{
  static const int taps[12] = { 4, -4, -8, 4, 46, 86, 86, 46, 4, -8, -4, 4 };
  int i,j;
  int k,l;
  uint8_t *ddata;

  for(j=0;j<dest->height;j++){
    ddata = OFFSET(dest->data, dest->stride * j);
    for(i=0;i<dest->width;i++){
      int x = 0;
      for(l=0;l<12;l++){
        int y = 0;
        for(k=0;k<12;k++){
          y += component_get (src, (i*2-5) + k, (j*2-5) + l) * taps[k];
        }
        x += CLAMP((y + 128) >> 8,0,255) * taps[l];
      }
      ddata[i] = CLAMP((x + 128) >> 8,0,255);
    }
  }
}

void
ref_frame_downsample (SchroFrame *dest, SchroFrame *src)
{
  ref_frame_component_downsample (dest->components+0, src->components+0);
  ref_frame_component_downsample (dest->components+1, src->components+1);
  ref_frame_component_downsample (dest->components+2, src->components+2);
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


