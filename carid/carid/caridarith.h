
#ifndef _CARID_ARITH_H_
#define _CARID_ARITH_H_

#include <carid/carid-stdint.h>
#include <carid/caridbits.h>
#include <carid/caridbuffer.h>

enum {
  CARID_CTX_BLOCK_SKIP = 0,
  CARID_CTX_QUANT_MAG,
  CARID_CTX_QUANT_SIGN,
  CARID_CTX_SIGN_POS,
  CARID_CTX_SIGN_NEG,
  CARID_CTX_SIGN_ZERO,
  CARID_CTX_Z_BIN1z,
  CARID_CTX_Z_BIN1nz,
  CARID_CTX_Z_BIN2,
  CARID_CTX_Z_BIN3,
  CARID_CTX_Z_BIN4,
  CARID_CTX_Z_BIN5,
  CARID_CTX_NZ_BIN1z,
  CARID_CTX_NZ_BIN1a,
  CARID_CTX_NZ_BIN1b,
  CARID_CTX_NZ_BIN2,
  CARID_CTX_NZ_BIN3,
  CARID_CTX_NZ_BIN4,
  CARID_CTX_NZ_BIN5,
  CARID_CTX_LAST
};


typedef struct _CaridArith CaridArith;
typedef struct _CaridArithContext CaridArithContext;

struct _CaridArithContext {
  int count0;
  int count1;
};

struct _CaridArith {
  CaridBits *bits;
  CaridBuffer *buf;

#if 0
  uint8_t *data;
  int size;
#endif
#if 0
  int offset;
  int bit_offset;
#endif

  uint16_t code;
  uint16_t low;
  uint16_t high;

  int cntr;

  int n_contexts;
  CaridArithContext contexts[CARID_CTX_LAST];
};

CaridArith * carid_arith_new (void);
void carid_arith_free (CaridArith *arith);
void carid_arith_decode_init (CaridArith *arith, CaridBits *bits);
void carid_arith_encode_init (CaridArith *arith, CaridBits *bits);
void carid_arith_context_init (CaridArith *arith, int i, int count0, int count1);
void carid_arith_context_halve_counts (CaridArith *arith, int i);
void carid_arith_context_halve_all_counts (CaridArith *arith);
void carid_arith_flush (CaridArith *arith);
void carid_arith_init_contexts (CaridArith *arith);

void carid_arith_context_encode_bit (CaridArith *arith, int context, int value);
void carid_arith_context_encode_uu (CaridArith *arith, int context, int context2, int value);
void carid_arith_context_encode_su (CaridArith *arith, int context, int value);
void carid_arith_context_encode_ut (CaridArith *arith, int context, int value, int max);
void carid_arith_context_encode_uegol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_segol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_ue2gol (CaridArith *arith, int context, int value);
void carid_arith_context_encode_se2gol (CaridArith *arith, int context, int value);

int carid_arith_context_decode_bit (CaridArith *arith, int context);
int carid_arith_context_decode_bits (CaridArith *arith, int context, int max);
int carid_arith_context_decode_uu (CaridArith *arith, int context, int context2);
int carid_arith_context_decode_su (CaridArith *arith, int context);
int carid_arith_context_decode_ut (CaridArith *arith, int context, int max);
int carid_arith_context_decode_uegol (CaridArith *arith, int context);
int carid_arith_context_decode_segol (CaridArith *arith, int context);
int carid_arith_context_decode_ue2gol (CaridArith *arith, int context);
int carid_arith_context_decode_se2gol (CaridArith *arith, int context);

#endif


