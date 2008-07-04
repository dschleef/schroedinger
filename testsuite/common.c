
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrofft.h>
#include <schroedinger/schrooil.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"


double sgn(double x)
{
  if (x<0) return -1;
  if (x>0) return 1;
  return 0;
}

double
random_std (void)
{
  double x;
  double y;

  while (1) {
    x = -5.0 + rand () * (1.0/RAND_MAX) * 10;
    y = rand () * (1.0/RAND_MAX);

    if (y < exp(-x*x*0.5)) return x;
  }
}

double
random_triangle (void)
{
  return rand () * (1.0/RAND_MAX) - rand () * (1.0/RAND_MAX);
}

int
gain_to_quant_index (double x)
{
  int i = 0;

  x *= x;
  x *= x;
  while (x*x > 2) {
    x *= 0.5;
    i++;
  }

  return i;
}

double
sum_f64 (double *a, int n)
{
  double sum = 0;
  int i;
  for(i=0;i<n;i++){
    sum += a[i];
  }
  return sum;
}

double
multsum_f64 (double *a, double *b, int n)
{
  double sum = 0;
  int i;
  for(i=0;i<n;i++){
    sum += a[i]*b[i];
  }
  return sum;
}



/* Test patterns */

static void
gen_random (SchroFrameData *fd, int type)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = oil_rand_u8();
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = oil_rand_u8();
      }
    }
  }
}

static int gen_const_array[] = { 0, 1, 127, 128, 129, 254, 255 };

#define CONST 255

static void
gen_const (SchroFrameData *fd, int type)
{
  int i,j;
  int value = gen_const_array[type];

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    for(j=0;j<fd->height;j++){
      uint8_t *data;
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = value;
      }
    }
  } else {
    for(j=0;j<fd->height;j++){
      int16_t *data;
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = value;
      }
    }
  }
}

static void
gen_vert_lines (SchroFrameData *fd, int type)
{
  int i,j;
  int pitch;

  pitch = type + 2;
  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((i%pitch)==0);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((i%pitch)==0);
      }
    }
  }
}

static void
gen_horiz_lines (SchroFrameData *fd, int type)
{
  int i,j;
  int pitch;

  pitch = type + 2;
  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((j%pitch)==0);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((j%pitch)==0);
      }
    }
  }
}

static void
gen_vert_bands (SchroFrameData *fd, int type)
{
  int i,j;
  int pitch;

  pitch = type + 1;
  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((i/pitch)&1);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((i/pitch)&1);
      }
    }
  }
}

static void
gen_horiz_bands (SchroFrameData *fd, int type)
{
  int i,j;
  int pitch;

  pitch = type + 1;
  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((j/pitch)&1);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*((j/pitch)&1);
      }
    }
  }
}

static void
gen_vert_edge (SchroFrameData *fd, int type)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*(i*2 < fd->width);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*(i*2 < fd->width);
      }
    }
  }
}

static void
gen_horiz_edge (SchroFrameData *fd, int type)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*(j*2 < fd->height);
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = CONST*(j*2 < fd->height);
      }
    }
  }
}

static void
gen_vert_ramp (SchroFrameData *fd, int type)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = ((i<<4)>>type)&0xff;
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = ((i<<4)>>type)&0xff;
      }
    }
  }
}

static void
gen_horiz_ramp (SchroFrameData *fd, int type)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(fd->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = ((j<<4)>>type)&0xff;
      }
    }
  } else {
    int16_t *data;
    for(j=0;j<fd->height;j++){
      data = SCHRO_FRAME_DATA_GET_LINE(fd, j);
      for(i=0;i<fd->width;i++) {
        data[i] = ((j<<4)>>type)&0xff;
      }
    }
  }
}


typedef struct _Generator Generator;
struct _Generator {
  char *name;
  void (*generate)(SchroFrameData *fd, int i);
  int n;
};

Generator generators_u8[] = {
  { "random", gen_random, 1 },
  { "const", gen_const, ARRAY_SIZE(gen_const_array) },
  { "vert_lines", gen_vert_lines, 8 },
  { "horiz_lines", gen_horiz_lines, 8 },
  { "vert_bands", gen_vert_bands, 8 },
  { "horiz_bands", gen_horiz_bands, 8 },
  { "vert_edge", gen_vert_edge, 1 },
  { "horiz_edge", gen_horiz_edge, 1 },
  { "vert_ramp", gen_vert_ramp, 8 },
  { "horiz_ramp", gen_horiz_ramp, 8 },
};

int
test_pattern_get_n_generators (void)
{
  int i;
  int n;

  n = 0;
  for(i=0;i<ARRAY_SIZE(generators_u8);i++) {
    n += generators_u8[i].n;
  }
  return n;
}

void
test_pattern_generate (SchroFrameData *fd, char *name, int n)
{
  int i;

  for(i=0;i<ARRAY_SIZE(generators_u8);i++) {
    if (n < generators_u8[i].n) {
      generators_u8[i].generate (fd, n);
      if (name) sprintf(name, "%s %d", generators_u8[i].name, n);
      return;
    }
    n -= generators_u8[i].n;
  }
}

int
frame_data_compare (SchroFrameData *dest, SchroFrameData *src)
{
  int i,j;

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *d;
    uint8_t *s;

    for(j=0;j<dest->height;j++){
      d = SCHRO_FRAME_DATA_GET_LINE(dest, j);
      s = SCHRO_FRAME_DATA_GET_LINE(src, j);
      for(i=0;i<dest->width;i++){
        if (d[i] != s[i]) return FALSE;
      }
    }
  } else {
    int16_t *d;
    int16_t *s;

    for(j=0;j<dest->height;j++){
      d = SCHRO_FRAME_DATA_GET_LINE(dest, j);
      s = SCHRO_FRAME_DATA_GET_LINE(src, j);
      for(i=0;i<dest->width;i++){
        if (d[i] != s[i]) return FALSE;
      }
    }
  }
  return TRUE;
}

void
frame_data_dump_u8 (SchroFrameData *test, SchroFrameData *ref)
{
  int i;
  int j;

  printf("=====\n");
  for(j=0;j<test->height;j++){
    uint8_t *tline;
    uint8_t *rline;

    tline = SCHRO_FRAME_DATA_GET_LINE(test, j);
    rline = SCHRO_FRAME_DATA_GET_LINE(ref, j);
    for(i=0;i<test->width;i++){
      if (tline[i] == rline[i]) {
        printf("%4d ", tline[i]);
      } else {
        printf("\033[00;01;37;41m%4d\033[00m ", tline[i]);
      }
    }
    printf("\n");
  }
  printf("=====\n");
}

void
frame_data_dump_s16 (SchroFrameData *test, SchroFrameData *ref)
{
  int i;
  int j;

  printf("=====\n");
  for(j=0;j<test->height;j++){
    int16_t *tline;
    int16_t *rline;

    tline = SCHRO_FRAME_DATA_GET_LINE(test, j);
    rline = SCHRO_FRAME_DATA_GET_LINE(ref, j);
    for(i=0;i<test->width;i++){
      if (tline[i] == rline[i]) {
        printf("%4d ", tline[i]);
      } else {
        printf("\033[00;01;37;41m%4d\033[00m ", tline[i]);
      }
    }
    printf("\n");
  }
  printf("=====\n");
}

void
frame_data_dump_full (SchroFrameData *test, SchroFrameData *ref, SchroFrameData *orig)
{
  int i;
  int j;

  printf("=====\n");
  if (SCHRO_FRAME_FORMAT_DEPTH(test->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    for(j=0;j<test->height;j++){
      uint8_t *tline;
      uint8_t *rline;
      uint8_t *oline;

      tline = SCHRO_FRAME_DATA_GET_LINE(test, j);
      rline = SCHRO_FRAME_DATA_GET_LINE(ref, j);
      oline = SCHRO_FRAME_DATA_GET_LINE(orig, j);
      for(i=0;i<test->width;i++){
        printf("\033[00;35m%3d\033[00m ", oline[i]);
      }
      printf("\n");
      for(i=0;i<test->width;i++){
        if (tline[i] == rline[i]) {
          printf("%3d ", tline[i]);
        } else {
          printf("\033[00;01;37;41m%3d\033[00m ", tline[i]);
        }
      }
      printf("\n");
      for(i=0;i<test->width;i++){
        printf("\033[00;32m%3d\033[00m ", rline[i]);
      }
      printf("\n");
    }
  } else {
    for(j=0;j<test->height;j++){
      int16_t *tline;
      int16_t *rline;

      tline = SCHRO_FRAME_DATA_GET_LINE(test, j);
      rline = SCHRO_FRAME_DATA_GET_LINE(ref, j);
      for(i=0;i<test->width;i++){
        if (tline[i] == rline[i]) {
          printf("%3d ", tline[i]);
        } else {
          printf("\033[00;01;37;41m%3d\033[00m ", tline[i]);
        }
      }
      printf("\n");
    }
  }
  printf("=====\n");
}


int
frame_compare (SchroFrame *dest, SchroFrame *src)
{
  int ret;
  ret = frame_data_compare (dest->components + 0, src->components + 0);
  ret &= frame_data_compare (dest->components + 1, src->components + 1);
  ret &= frame_data_compare (dest->components + 2, src->components + 2);
  return ret;
}

void
frame_data_dump (SchroFrameData *test, SchroFrameData *ref)
{
  if (SCHRO_FRAME_FORMAT_DEPTH(test->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    frame_data_dump_u8 (test, ref);
  } else {
    frame_data_dump_s16 (test, ref);
  }
}

void
frame_dump (SchroFrame *test, SchroFrame *ref)
{
  if (SCHRO_FRAME_FORMAT_DEPTH(test->format) == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    frame_data_dump_u8 (test->components + 0, ref->components + 0);
    frame_data_dump_u8 (test->components + 1, ref->components + 1);
    frame_data_dump_u8 (test->components + 2, ref->components + 2);
  } else {
    frame_data_dump_s16 (test->components + 0, ref->components + 0);
    frame_data_dump_s16 (test->components + 1, ref->components + 1);
    frame_data_dump_s16 (test->components + 2, ref->components + 2);
  }
}

/* simple parse dirac stream */


int
parse_packet (FILE *file, void **p_data, int *p_size)
{
  unsigned char *packet;
  unsigned char header[13];
  int n;
  int size;

  n = fread (header, 1, 13, file);
  if (n == 0) {
    *p_data = NULL;
    *p_size = 0;
    return 1;
  }
  if (n < 13) {
    printf("truncated header\n");
    return 0;
  }

  if (header[0] != 'B' || header[1] != 'B' || header[2] != 'C' ||
      header[3] != 'D') {
    return 0;
  }

  size = (header[5]<<24) | (header[6]<<16) | (header[7]<<8) | (header[8]);
  if (size == 0) {
    size = 13;
  }
  if (size < 13) {
    return 0;
  }
  if (size > 16*1024*1024) {
    printf("packet too large? (%d > 16777216)\n", size);
    return 0;
  }

  packet = malloc (size);
  memcpy (packet, header, 13);
  n = fread (packet + 13, 1, size - 13, file);
  if (n < size - 13) {
    free (packet);
    return 0;
  }

  *p_data = packet;
  *p_size = size;
  return 1;
}

