
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
  int size;
  int offset;
  int bit_offset;

  uint16_t code;
  uint16_t low;
  uint16_t high;

  int cntr;

  int n_contexts;
  CaridArithContext contexts[CARID_ARITH_N_CONTEXTS];
};

CaridArith * carid_arith_new (void);
void carid_arith_free (CaridArith *arith);
void carid_arith_decode_init (CaridArith *arith);
void carid_arith_encode_init (CaridArith *arith);
void carid_arith_context_init (CaridArith *arith, int i, int count0, int count1);
void carid_arith_context_halve_counts (CaridArith *arith, int i);
void carid_arith_context_halve_all_counts (CaridArith *arith);
void carid_arith_context_update (CaridArith *arith, int i, int value);
int carid_arith_context_binary_decode (CaridArith *arith, int i);
void carid_arith_context_binary_encode (CaridArith *arith, int i, int value);
void carid_arith_flush (CaridArith *arith);
void carid_arith_input_bit (CaridArith *arith);
void carid_arith_output_bit (CaridArith *arith);

void carid_arith_context_encode_uu (CaridArith *arith, int context, int value);
void carid_arith_context_encode_su (CaridArith *arith, int context, int value);
void carid_arith_context_encode_ut (CaridArith *arith, int context, int value, int max);
void carid_arith_context_encode_uegol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_segol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_ue2gol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_se2gol (CaridArith *arith, int context, int value);

#endif


