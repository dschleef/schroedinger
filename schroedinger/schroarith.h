
#ifndef _SCHRO_ARITH_H_
#define _SCHRO_ARITH_H_

#include <schroedinger/schroutils.h>
#include <schroedinger/schrobuffer.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

enum {
  SCHRO_CTX_ZERO_CODEBLOCK = 0,
  SCHRO_CTX_QUANTISER_CONT,
  SCHRO_CTX_QUANTISER_VALUE,
  SCHRO_CTX_QUANTISER_SIGN,
  SCHRO_CTX_ZPZN_F1,
  SCHRO_CTX_ZPNN_F1,
  SCHRO_CTX_ZP_F2,
  SCHRO_CTX_ZP_F3,
  SCHRO_CTX_ZP_F4,
  SCHRO_CTX_ZP_F5,
  SCHRO_CTX_ZP_F6p,
  SCHRO_CTX_NPZN_F1,
  SCHRO_CTX_NPNN_F1,
  SCHRO_CTX_NP_F2,
  SCHRO_CTX_NP_F3,
  SCHRO_CTX_NP_F4,
  SCHRO_CTX_NP_F5,
  SCHRO_CTX_NP_F6p,
  SCHRO_CTX_SIGN_POS,
  SCHRO_CTX_SIGN_NEG,
  SCHRO_CTX_SIGN_ZERO,
  SCHRO_CTX_COEFF_DATA,

  SCHRO_CTX_SB_F1,
  SCHRO_CTX_SB_F2,
  SCHRO_CTX_SB_DATA,
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
  unsigned int next;
  int stat_range;
  int n_bits;
  int n_symbols;
  int ones;
};

/* TODO: eliminate unused members */
struct _SchroArith {
  SchroBuffer *buffer;
  uint8_t *dataptr;
  uintptr_t offset;

  uint32_t range[2];
  uint32_t code;
  uint32_t range_size;
  int cntr;
  int carry;
  uint8_t shift;

  uint16_t probabilities[SCHRO_CTX_LAST];
  uint16_t lut[512];
  SchroArithContext contexts[SCHRO_CTX_LAST];
};

SchroArith * schro_arith_new (void);
void schro_arith_free (SchroArith *arith);
void schro_arith_decode_init (SchroArith *arith, SchroBuffer *buffer);
void schro_arith_encode_init (SchroArith *arith, SchroBuffer *buffer);
void schro_arith_estimate_init (SchroArith *arith);
void schro_arith_flush (SchroArith *arith);
void schro_arith_decode_flush (SchroArith *arith);

void schro_arith_encode_bit (SchroArith *arith, int context, int value);
void schro_arith_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value);
void schro_arith_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value);

int schro_arith_decode_bit (SchroArith *arith, unsigned int context);
int schro_arith_decode_uint (SchroArith *arith, unsigned int cont_context,
    unsigned int value_context);
int schro_arith_decode_sint (SchroArith *arith, unsigned int cont_context,
    unsigned int value_context, unsigned int sign_context);

void _schro_arith_encode_bit (SchroArith *arith, int context, int
    value) SCHRO_INTERNAL;
void _schro_arith_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value) SCHRO_INTERNAL;
void _schro_arith_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value) SCHRO_INTERNAL;

int _schro_arith_decode_sint (SchroArith *arith, unsigned int cont_context,
    unsigned int value_context, unsigned int sign_context) SCHRO_INTERNAL;

void schro_arith_estimate_bit (SchroArith *arith, int i, int value);
void schro_arith_estimate_uint (SchroArith *arith, int cont_context,
    int value_context, int value);
void schro_arith_estimate_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value);

#ifdef SCHRO_ARITH_DEFINE_INLINE
static int
_schro_arith_decode_bit (SchroArith *arith, unsigned int i)
{
  unsigned int range_x_prob;
  unsigned int value;
  unsigned int lut_index;
  register unsigned int range = arith->range[1];
  register unsigned int code_minus_low = arith->code;

  while (range <= 0x4000) {

    range <<= 1;
    code_minus_low <<= 1;

    if (!--arith->cntr) {
      arith->offset++;
      if (arith->offset < arith->buffer->length) {
        code_minus_low |= arith->dataptr[arith->offset] << 8;
      } else {
        code_minus_low |= 0xff00;
      }

      arith->offset++;
      if (arith->offset < arith->buffer->length) {
        code_minus_low |= arith->dataptr[arith->offset];
      } else {
        code_minus_low |= 0xff;
      }

      arith->cntr = 16;
    }
  }

  range_x_prob = (range * arith->probabilities[i]) >> 16;
  lut_index = arith->probabilities[i]>>7 & ~1;

  value = ((code_minus_low >> 16) >= range_x_prob);
  arith->probabilities[i] += arith->lut[lut_index | value];

  if (value) {
    code_minus_low -= range_x_prob << 16;
    range -= range_x_prob;
  } else {
    range = range_x_prob;
  }

  arith->range[1] = range;
  arith->code = code_minus_low;

  return value;
}

static int
_schro_arith_decode_uint (SchroArith *arith, unsigned int cont_context,
    unsigned int value_context)
{
  unsigned int bits=1;
  int count=0;

  while(!_schro_arith_decode_bit (arith, cont_context)) {
    bits <<= 1;
    bits |= _schro_arith_decode_bit (arith, value_context);
    cont_context = arith->contexts[cont_context].next;
    count++;

    /* FIXME being careful */
    if (count == 30) break;
  }
  return bits - 1;
}
#else /* SCHRO_ARITH_DEFINE_INLINE */
int _schro_arith_decode_bit (SchroArith *arith, unsigned int context);
int _schro_arith_decode_uint (SchroArith *arith, unsigned int cont_context,
    unsigned int value_context);
#endif /* SCHRO_ARITH_DEFINE_INLINE */

#endif

SCHRO_END_DECLS

#endif


