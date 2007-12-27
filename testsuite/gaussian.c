
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schroutils.h>
#include <math.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboilprofile.h>

#define N 2000
#define INT 1
#define SIGMA 5.0
#define VSHIFT 10
#define SHIFT 12


uint8_t src[N];
uint8_t dest1[N];
uint8_t dest2[N];

typedef struct _KernelState KernelState;
struct _KernelState {
  double y1;
  double y2;
  double y3;

  double b0;
  double b1;
  double b2;
  double b3;
  double B;
  double b0inv;

  double c[4];

  int value_shift;
  int shift;
  int d[4];
  int y1i;
  int y2i;
  int y3i;
  int err;
};

void
create_pattern (uint8_t *data, int n)
{
  int i;

  for(i=0;i<n;i++){
    //data[i] = 255*((i>=990) && (i<1010));
    //data[i] = 255*(i==1000);
    data[i] = 255*((i>>3)&1);
    //data[i] = 100+100*((i>>4)&1);
    //data[i] = 100;
  }
}

void
kernel_init (KernelState *state, double sigma)
{
  double q;

  if (sigma >= 2.5) {
    q = 0.98711 * sigma - 0.96330;
  } else {
    q = 3.97156 - 4.41554 * sqrt (1 - 0.26891 * sigma);
  }

  state->b0 = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
  state->b0inv = 1.0/state->b0;
  state->b1 = 2.44413*q + 2.85619*q*q + 1.26661*q*q*q;
  state->b2 = -1.4281*q*q - 1.26661*q*q*q;
  state->b3 = 0.422205*q*q*q;
  state->B = 1 - (state->b1 + state->b2 + state->b3)/state->b0;

  state->c[0] = state->B;
  state->c[1] = state->b1 * state->b0inv;
  state->c[2] = state->b2 * state->b0inv;
  state->c[3] = state->b3 * state->b0inv;

  state->value_shift = VSHIFT;
  state->shift = SHIFT;
  state->d[0] = rint(state->c[0]*(1<<(state->shift + state->value_shift)));
  state->d[1] = rint(state->c[1]*(1<<state->shift));
  state->d[2] = rint(state->c[2]*(1<<state->shift));
  state->d[3] = rint(state->c[3]*(1<<state->shift));

  state->err = 0;
}

void
kernel_set_start (KernelState *state, int value)
{
  state->y1 = value;
  state->y2 = value;
  state->y3 = value;

  state->y1i = value << state->value_shift;
  state->y2i = value << state->value_shift;
  state->y3i = value << state->value_shift;
}

static int
kernel_int (KernelState *state, int value)
{
  int x;
  int y;

  y = state->d[0]*value + state->d[1]*state->y1i + state->d[2]*state->y2i +
    state->d[3]*state->y3i;
#if 1
  x = (y + (1<<(state->shift - 1))) >> state->shift;
#else
  x = (y + state->err) >> state->shift;
  state->err = y - (x << state->shift);
#endif

  state->y3i = state->y2i;
  state->y2i = state->y1i;
  state->y1i = x;

  return (x + (1<<(state->value_shift - 1))) >> state->value_shift;
}

void
filter_u8_int (uint8_t *dest, uint8_t *src, uint8_t *tmp, double sigma, int n)
{
  KernelState state;
  int i;

  kernel_init (&state, sigma);

  kernel_set_start(&state, src[0]);
  for(i=0;i<n;i++) {
    int x;
    x = kernel_int(&state, src[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    tmp[i] = x;
  }

  kernel_set_start(&state, tmp[n-1]);
  for(i=n-1;i>=0;i--) {
    int x;
    x = kernel_int(&state, tmp[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    dest[i] = x;
  }
}

static int
kernel_double (KernelState *state, int value)
{
  double x;

  x = state->c[0]*value + state->c[1]*state->y1 + state->c[2]*state->y2 +
    state->c[3]*state->y3;

  state->y3 = state->y2;
  state->y2 = state->y1;
  state->y1 = x;

  return rint(x);
}

void
filter_u8_double (uint8_t *dest, uint8_t *src, uint8_t *tmp, double sigma, int n)
{
  KernelState state;
  int i;

  kernel_init (&state, sigma);

  kernel_set_start(&state, src[0]);
  for(i=0;i<n;i++) {
    int x;
    x = kernel_double(&state, src[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    tmp[i] = x;
  }

  kernel_set_start(&state, tmp[n-1]);
  for(i=n-1;i>=0;i--) {
    int x;
    x = kernel_double(&state, tmp[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    dest[i] = x;
  }
}



void
test (void)
{
  int i;
  double sigma;
  int diff;
  OilProfile prof;
  double ave, std;
  uint8_t *tmp;

  tmp = malloc (N);

  create_pattern (src, N);

  sigma = SIGMA;

  oil_profile_init (&prof);
  for(i=0;i<10;i++){
    oil_profile_start(&prof);
    filter_u8_double (dest1, src, tmp, sigma, N);
    oil_profile_stop(&prof);
  }
  oil_profile_get_ave_std (&prof, &ave, &std);
  fprintf(stderr, "double cycles %g %g\n", ave, std);

  oil_profile_init (&prof);
  for(i=0;i<10;i++){
    oil_profile_start(&prof);
    filter_u8_int (dest2, src, tmp, sigma, N);
    oil_profile_stop(&prof);
  }
  oil_profile_get_ave_std (&prof, &ave, &std);
  fprintf(stderr, "int cycles %g %g\n", ave, std);

  diff = 0;
  for(i=0;i<N;i++){
    printf("%d %d %d %d\n", i, src[i], dest1[i], dest2[i]);
    //    255.0*exp(-(N/2-i)*(N/2-i)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma));
    diff += abs(dest1[i] - dest2[i]);
  }
  fprintf(stderr,"diff %d\n", diff);

  free(tmp);
}

int
main (int argc, char *argv[])
{
  schro_init();

  test();

  return 0;
}


