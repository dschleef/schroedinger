
#ifndef _ARITH_QM_H_
#define _ARITH_QM_H_

typedef struct _ArithQM ArithQM;
struct _ArithQM {
  int a;
  int c;
  int st;
  int ct;
  int bp;
  unsigned char *data;

  int contexts_mps[10];
  int contexts_state[10];
};

void arith_qm_init (ArithQM *arith);
void arith_qm_encode (ArithQM *arith, int context, int value);
void arith_qm_flush (ArithQM *coder);

#endif

