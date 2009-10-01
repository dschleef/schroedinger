
#ifndef _ARITH_H_
#define _ARITH_H_

#include "../common.h"
#include <orc-test/orcprofile.h>

typedef struct _Arith Arith;
typedef struct _Context Context;

struct _Context {
  int mps;
  int state;
  int count[2];
  int next;
  int probability;
  int n;
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
  
  int carry;

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
    value = (rand_u8() < x); \
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
  OrcProfile prof; \
  indata = malloc(n); \
  orc_profile_init (&prof); \
  for(i=0;i<n;i++){ \
    indata[i] = (rand_u8() < x); \
  } \
  for(j=0;j<20;j++) { \
    arith_ ## name ## _init (&a); \
    a.data = data; \
    orc_profile_start(&prof); \
    for(i=0;i<n;i++){ \
      arith_ ## name ## _encode (&a, 0, indata[i]); \
    } \
    orc_profile_stop(&prof); \
  } \
  orc_profile_get_ave_std (&prof, &ave, &std); \
  free (indata); \
  return ave/n; \
}

#define DEFINE_ENCODE(name) \
int encode_arith_ ## name (unsigned char *outdata, \
    unsigned char *indata, int n) \
{ \
  Arith a; \
  int i; \
  arith_ ## name ## _init (&a); \
  a.data = outdata; \
  for(i=0;i<n;i++){ \
    arith_ ## name ## _encode (&a, 0, indata[i]); \
  } \
  arith_ ## name ## _flush (&a); \
  return a.offset; \
}

#define DEFINE_DECODE(name) \
void decode_arith_ ## name (unsigned char *outdata, \
    unsigned char *indata, int n) \
{ \
  Arith a; \
  int i; \
  arith_ ## name ## _init (&a); \
  a.data = indata; \
  a.code = (indata[0]<<8) | indata[1]; \
  a.offset = 2; \
  for(i=0;i<n;i++){ \
    outdata[i] = arith_ ## name ## _decode (&a, 0); \
  } \
}

#endif

