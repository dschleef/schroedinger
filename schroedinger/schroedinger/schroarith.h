
#ifndef _SCHRO_ARITH_H_
#define _SCHRO_ARITH_H_

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schrobits.h>
#include <schroedinger/schrobuffer.h>

enum {
  SCHRO_CTX_ZERO_CODEBLOCK = 0,
  SCHRO_CTX_QUANTISER_CONT,
  SCHRO_CTX_QUANTISER_VALUE,
  SCHRO_CTX_QUANTISER_SIGN,
  SCHRO_CTX_Z_BIN1_0,
  SCHRO_CTX_Z_BIN1_1,
  SCHRO_CTX_Z_BIN2,
  SCHRO_CTX_Z_BIN3,
  SCHRO_CTX_Z_BIN4,
  SCHRO_CTX_Z_BIN5,
  SCHRO_CTX_Z_VALUE,
  SCHRO_CTX_Z_SIGN_0,
  SCHRO_CTX_Z_SIGN_1,
  SCHRO_CTX_Z_SIGN_2,
  SCHRO_CTX_NZ_BIN1_0,
  SCHRO_CTX_NZ_BIN1_1,
  SCHRO_CTX_NZ_BIN1_2,
  SCHRO_CTX_NZ_BIN2,
  SCHRO_CTX_NZ_BIN3,
  SCHRO_CTX_NZ_BIN4,
  SCHRO_CTX_NZ_BIN5,
  SCHRO_CTX_NZ_VALUE,
  SCHRO_CTX_NZ_SIGN_0,
  SCHRO_CTX_NZ_SIGN_1,
  SCHRO_CTX_NZ_SIGN_2,
  SCHRO_CTX_LAST
};


typedef struct _SchroArith SchroArith;
typedef struct _SchroArithContext SchroArithContext;

struct _SchroArithContext {
  int count0;
  int count1;
  int next;
};

struct _SchroArith {
  SchroBits *bits;
  SchroBuffer *buf;

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
  SchroArithContext contexts[SCHRO_CTX_LAST];
};

SchroArith * schro_arith_new (void);
void schro_arith_free (SchroArith *arith);
void schro_arith_decode_init (SchroArith *arith, SchroBits *bits);
void schro_arith_encode_init (SchroArith *arith, SchroBits *bits);
void schro_arith_context_init (SchroArith *arith, int i, int count0, int count1);
void schro_arith_context_halve_counts (SchroArith *arith, int i);
void schro_arith_context_halve_all_counts (SchroArith *arith);
void schro_arith_flush (SchroArith *arith);
void schro_arith_init_contexts (SchroArith *arith);

void schro_arith_context_encode_bit (SchroArith *arith, int context, int value);
void schro_arith_context_encode_uu (SchroArith *arith, int context, int context2, int value);
void schro_arith_context_encode_su (SchroArith *arith, int context, int value);
void schro_arith_context_encode_ut (SchroArith *arith, int context, int value, int max);
void schro_arith_context_encode_uegol (SchroArith *arith, int context, int value);
void schro_arith_context_encode_segol (SchroArith *arith, int context, int value);
void schro_arith_context_encode_ue2gol (SchroArith *arith, int context, int value);
void schro_arith_context_encode_se2gol (SchroArith *arith, int context, int value);
void schro_arith_context_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value);
void schro_arith_context_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value);

int schro_arith_context_decode_bit (SchroArith *arith, int context);
int schro_arith_context_decode_bits (SchroArith *arith, int context, int max);
int schro_arith_context_decode_uu (SchroArith *arith, int context, int context2);
int schro_arith_context_decode_su (SchroArith *arith, int context);
int schro_arith_context_decode_ut (SchroArith *arith, int context, int max);
int schro_arith_context_decode_uegol (SchroArith *arith, int context);
int schro_arith_context_decode_segol (SchroArith *arith, int context);
int schro_arith_context_decode_ue2gol (SchroArith *arith, int context);
int schro_arith_context_decode_se2gol (SchroArith *arith, int context);
int schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context);
int schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context);

#endif


