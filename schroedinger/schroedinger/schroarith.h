
#ifndef _SCHRO_ARITH_H_
#define _SCHRO_ARITH_H_

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schrobits.h>
#include <schroedinger/schrobuffer.h>

#define INTERNAL __attribute__ ((visibility ("internal")))

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

  SCHRO_CTX_SPLIT_0,
  SCHRO_CTX_SPLIT_1,
  SCHRO_CTX_COMMON,
  SCHRO_CTX_BLOCK_MODE_REF1,
  SCHRO_CTX_BLOCK_MODE_REF2,
  SCHRO_CTX_GLOBAL_BLOCK,
  SCHRO_CTX_LUMA_DC_CONT_BIN1,
  SCHRO_CTX_LUMA_DC_CONT_BIN2,
  SCHRO_CTX_LUMA_DC_VALUE,
  SCHRO_CTX_LUMA_DC_SIGN,
  SCHRO_CTX_CHROMA1_DC_CONT_BIN1,
  SCHRO_CTX_CHROMA1_DC_CONT_BIN2,
  SCHRO_CTX_CHROMA1_DC_VALUE,
  SCHRO_CTX_CHROMA1_DC_SIGN,
  SCHRO_CTX_CHROMA2_DC_CONT_BIN1,
  SCHRO_CTX_CHROMA2_DC_CONT_BIN2,
  SCHRO_CTX_CHROMA2_DC_VALUE,
  SCHRO_CTX_CHROMA2_DC_SIGN,
  SCHRO_CTX_MV_REF1_H_CONT_BIN1,
  SCHRO_CTX_MV_REF1_H_CONT_BIN2,
  SCHRO_CTX_MV_REF1_H_CONT_BIN3,
  SCHRO_CTX_MV_REF1_H_CONT_BIN4,
  SCHRO_CTX_MV_REF1_H_CONT_BIN5,
  SCHRO_CTX_MV_REF1_H_VALUE,
  SCHRO_CTX_MV_REF1_H_SIGN,
  SCHRO_CTX_MV_REF1_V_CONT_BIN1,
  SCHRO_CTX_MV_REF1_V_CONT_BIN2,
  SCHRO_CTX_MV_REF1_V_CONT_BIN3,
  SCHRO_CTX_MV_REF1_V_CONT_BIN4,
  SCHRO_CTX_MV_REF1_V_CONT_BIN5,
  SCHRO_CTX_MV_REF1_V_VALUE,
  SCHRO_CTX_MV_REF1_V_SIGN,
  SCHRO_CTX_MV_REF2_H_CONT_BIN1,
  SCHRO_CTX_MV_REF2_H_CONT_BIN2,
  SCHRO_CTX_MV_REF2_H_CONT_BIN3,
  SCHRO_CTX_MV_REF2_H_CONT_BIN4,
  SCHRO_CTX_MV_REF2_H_CONT_BIN5,
  SCHRO_CTX_MV_REF2_H_VALUE,
  SCHRO_CTX_MV_REF2_H_SIGN,
  SCHRO_CTX_MV_REF2_V_CONT_BIN1,
  SCHRO_CTX_MV_REF2_V_CONT_BIN2,
  SCHRO_CTX_MV_REF2_V_CONT_BIN3,
  SCHRO_CTX_MV_REF2_V_CONT_BIN4,
  SCHRO_CTX_MV_REF2_V_CONT_BIN5,
  SCHRO_CTX_MV_REF2_V_VALUE,
  SCHRO_CTX_MV_REF2_V_SIGN,

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
  uint16_t code;
  uint16_t low;
  uint16_t high;

  int cntr;

  int n_contexts;
  SchroArithContext contexts[SCHRO_CTX_LAST];

  SchroBuffer *buffer;
  int offset;
  uint8_t nextcode;
  int nextbits;
};

SchroArith * schro_arith_new (void) INTERNAL;
void schro_arith_free (SchroArith *arith) INTERNAL;
void schro_arith_decode_init (SchroArith *arith, SchroBuffer *buffer) INTERNAL;
void schro_arith_encode_init (SchroArith *arith, SchroBuffer *buffer) INTERNAL;
void schro_arith_context_init (SchroArith *arith, int i, int count0, int count1) INTERNAL;
void schro_arith_context_halve_counts (SchroArith *arith, int i) INTERNAL;
void schro_arith_halve_all_counts (SchroArith *arith) INTERNAL;
void schro_arith_flush (SchroArith *arith) INTERNAL;
void schro_arith_init_contexts (SchroArith *arith) INTERNAL;

void schro_arith_context_encode_bit (SchroArith *arith, int context, int value) INTERNAL;
void schro_arith_context_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value) INTERNAL;
void schro_arith_context_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value) INTERNAL;
void schro_arith_encode_mode (SchroArith *arith, int context0, int context1,
    int value);

int schro_arith_context_decode_bit (SchroArith *arith, int context) INTERNAL;
int schro_arith_context_decode_bits (SchroArith *arith, int context, int max) INTERNAL;
int schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context) INTERNAL;
int schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context) INTERNAL;
int schro_arith_decode_mode (SchroArith *arith, int context0, int context1);

#endif


