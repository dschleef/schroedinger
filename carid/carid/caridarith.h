
#ifndef _CARID_ARITH_H_
#define _CARID_ARITH_H_

#include <carid/carid-stdint.h>


#define CARID_ARITH_N_CONTEXTS 64

typedef struct _CaridArith CaridArith;
typedef struct _CaridArithContext CaridArithContext;

struct _CaridArithContext {
  int count0;
  int count1;
};

struct _CaridArith {
  uint8_t *data;
  int length;
  int offset;
  int bit_offset;

  uint16_t code;
  uint16_t low;
  uint16_t high;

  int underflow;

  int n_contexts;
  CaridArithContext contexts[CARID_ARITH_N_CONTEXTS];
};



#endif

