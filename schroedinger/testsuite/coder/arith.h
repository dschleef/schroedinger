
#ifndef _ARITH_H_
#define _ARITH_H_

#include <liboil/liboilrandom.h>
#include <liboil/liboilprofile.h>

typedef struct _Arith Arith;
typedef struct _Context Context;

struct _Context {
  int mps;
  int state;
  int count[2];
  int next;
};

struct _Arith {
  int code;
  int range0;
  int range1;

#if 0
  int a;
  int c;
  int st;
  int ct;
  int bp;
#endif
  int cntr;
  int st;

  unsigned int output_byte;
  int output_bits;

  unsigned char *data;
  int offset;

  Context contexts[10];
};

#define DEFINE_EFFICIENCY(name) \
int efficiency_arith_ ## name (int x, unsigned char *data, int n) \
{ \
  Arith a; \
  int i; \
  int value; \
  arith_ ## name ## _init (&a); \
  a.data = data; \
  for(i=0;i<n;i++){ \
    value = (oil_rand_u8() < x); \
    arith_ ## name ## _encode (&a, 0, value); \
  } \
  arith_ ## name ## _flush(&a); \
  return a.offset*8; \
}

#define DEFINE_SPEED(name) \
double speed_arith_ ## name (int x, unsigned char *data, int n) \
{ \
  Arith a; \
  int i; \
  int j; \
  double ave, std; \
  unsigned char *indata; \
  OilProfile prof; \
  indata = malloc(n); \
  oil_profile_init (&prof); \
  for(i=0;i<n;i++){ \
    indata[i] = (oil_rand_u8() < x); \
  } \
  for(j=0;j<10;j++) { \
    arith_ ## name ## _init (&a); \
    a.data = data; \
    oil_profile_start(&prof); \
    for(i=0;i<n;i++){ \
      arith_ ## name ## _encode (&a, 0, indata[i]); \
    } \
    oil_profile_stop(&prof); \
  } \
  oil_profile_get_ave_std (&prof, &ave, &std); \
  free (indata); \
  return ave/n; \
}


#endif

