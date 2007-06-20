
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

#define N 2000
#define INT 1
#define SIGMA 3.0
#define SHIFT 11


double src[N];
double dest[N];

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
create_pattern (double *data, int n)
{
  int i;

  for(i=0;i<n;i++){
    //data[i] = 255*((i>=990) && (i<1010));
    //data[i] = 255*(i==1000);
    data[i] = 255*((i>>2)&1);
  }
}

double
kernel (KernelState *state, double value)
{
  double x;

#if 0
  x = state->B*value +state->b0inv*
    (state->y1*state->b1 + state->y2*state->b2 + state->y3*state->b3);
#endif
#ifdef INT
  value *= (1<<state->value_shift);
  x = (state->d[0]*value + state->d[1]*state->y1 + state->d[2]*state->y2 +
    state->d[3]*state->y3)/(1<<state->shift);
  x = rint(x);
#else
  x = state->c[0]*value + state->c[1]*state->y1 + state->c[2]*state->y2 +
    state->c[3]*state->y3;
#endif

  state->y3 = state->y2;
  state->y2 = state->y1;
  state->y1 = x;

  return x/(1<<state->value_shift);
}

int
kernel_int (KernelState *state, int value)
{
  int x;
  int y;

  //value <<= state->value_shift;
  y = state->d[0]*value + state->d[1]*state->y1i + state->d[2]*state->y2i +
    state->d[3]*state->y3i;
  //x = (y + (1<<(state->shift - 1)) + state->err) >> state->shift;
  x = (y + state->err) >> state->shift;
  state->err = y - (x << state->shift);

  state->y3i = state->y2i;
  state->y2i = state->y1i;
  state->y1i = x;

  return (x + (1<<(state->value_shift - 1))) >> state->value_shift;
}

void
kernel_reverse (KernelState *state)
{
  state->y3 = state->y1;
  state->y2 = state->y1;

  state->y3i = state->y1i;
  state->y2i = state->y1i;
}

void
kernel_state_init (KernelState *state, double sigma, double value)
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

  state->y1 = value;
  state->y2 = value;
  state->y3 = value;

  state->c[0] = state->B;
  state->c[1] = state->b1 * state->b0inv;
  state->c[2] = state->b2 * state->b0inv;
  state->c[3] = state->b3 * state->b0inv;

  state->value_shift = SHIFT;
  state->shift = SHIFT;
  state->d[0] = rint(state->c[0]*(1<<(state->shift + state->value_shift)));
  state->d[1] = rint(state->c[1]*(1<<state->shift));
  state->d[2] = rint(state->c[2]*(1<<state->shift));
  state->d[3] = rint(state->c[3]*(1<<state->shift));

  state->y1i = value;
  state->y2i = value;
  state->y3i = value;
  state->err = 0;
}

void
test (void)
{
  double *tmp, *tmpdata;
  int i;
  int n = N;
  KernelState state;
  double sigma;

  tmpdata = malloc ((N + 8)*sizeof(double));
  tmp = tmpdata + 4;

  create_pattern (src, N);

  sigma = SIGMA;

#if 0
  for(x=1;x<10.0;x+=0.5) {
  kernel_state_init (&state, x, src[0]);
  fprintf(stderr,"%g %g %g %g\n", state.c[0], state.c[1], state.c[2], state.c[3]);
  fprintf(stderr,"%d %d %d %d\n", state.d[0], state.d[1], state.d[2], state.d[3]);
  }
#endif
  kernel_state_init (&state, sigma, src[0]);
  fprintf(stderr,"%g %g %g %g\n", state.c[0], state.c[1], state.c[2], state.c[3]);
  fprintf(stderr,"%d %d %d %d\n", state.d[0], state.d[1], state.d[2], state.d[3]);

  for(i=0;i<n;i++) {
#ifdef INT
    int x;
    x = kernel_int(&state, src[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    tmp[i] = rint(x);
#else
    double x;
    x = kernel(&state, src[i]);
    tmp[i] = x;
#endif
  }
  kernel_reverse(&state);
  for(i=n-1;i>=0;i--) {
#ifdef INT
    int x;
    x = kernel_int(&state, tmp[i]);
    if (x < 0) x = 0;
    if (x > 255) x = 255;
    dest[i] = x;
#else
    double x;
    x = kernel(&state, tmp[i]);
    dest[i] = x;
#endif
  }

  for(i=0;i<N;i++){
    printf("%d %g %g %g %g\n", i, src[i], tmp[i], dest[i],
        255.0*exp(-(N/2-i)*(N/2-i)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma));
  }

  free(tmpdata);
}

void dump(void)
{
  int i;
  for(i=0;i<N;i++){
    printf("%d %g %g\n", i, src[i], dest[i]);
  }
}

int
main (int argc, char *argv[])
{
  schro_init();

  test();

  if(0) dump();

  return 0;
}


