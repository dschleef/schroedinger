
#ifndef _ARITH_EXP_H
#define _ARITH_EXP_H

#include <schroedinger/schroarith.h>

typedef struct _ArithExp ArithExp;

struct _ArithExp {
  int code;
  int range0;
  int range1;
  int value;

  int cntr;
  int offset;
  unsigned char *dataptr;

  SchroArithContext contexts[1];
};


void arith_exp_encode (ArithExp *arith, int i, int value);
void arith_exp_init (ArithExp *arith);
void arith_exp_flush (ArithExp *arith);

#endif

