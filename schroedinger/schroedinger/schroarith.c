
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

//#include <stdlib.h>
//#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>

static int __schro_arith_context_decode_bit (SchroArith *arith, int i);

SchroArith *
schro_arith_new (void)
{
  SchroArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  /* FIXME wtf? */
  (void)&__schro_arith_context_decode_bit;

  return arith;
}

void
schro_arith_free (SchroArith *arith)
{
  free(arith);
}

void
schro_arith_decode_init (SchroArith *arith, SchroBuffer *buffer)
{
  memset(arith, 0, sizeof(SchroArith));
  arith->range[0] = 0;
  arith->range[1] = 0x10000;
  arith->code = 0;

  arith->buffer = buffer;

  arith->dataptr = arith->buffer->data;
  //arith->maxdataptr = arith->buffer->data + arith->buffer->length;
  arith->code = arith->dataptr[0] << 8;
  arith->code |= arith->dataptr[1];
  arith->offset = 2;
}

void
schro_arith_encode_init (SchroArith *arith, SchroBuffer *buffer)
{
  memset(arith, 0, sizeof(SchroArith));
  arith->range[0] = 0;
  arith->range[1] = 0x10000;
  arith->code = 0;

  arith->buffer = buffer;
  arith->offset = 0;
  arith->dataptr = arith->buffer->data;
}

void
schro_arith_flush (SchroArith *arith)
{
  while (arith->cntr < 8) {
    arith->range[0] <<= 1;
    arith->cntr++;
  }

  if (arith->range[0] >= (1<<24)) {
    arith->dataptr[arith->offset-1]++;
    while (arith->carry) {
      arith->dataptr[arith->offset] = 0x00;
      arith->carry--;
      arith->offset++;
    }
  } else {
    while (arith->carry) {
      arith->dataptr[arith->offset] = 0xff;
      arith->carry--;
      arith->offset++;
    }
  }

  arith->dataptr[arith->offset] = arith->range[0] >> 16;
  arith->offset++;
  arith->dataptr[arith->offset] = arith->range[0] >> 8;
  arith->offset++;
  arith->dataptr[arith->offset] = arith->range[0] >> 0;
  arith->offset++;
}

static const int next_list[] = {
  0,
  SCHRO_CTX_QUANTISER_CONT,
  0,
  0,
  SCHRO_CTX_ZP_F2,
  SCHRO_CTX_ZP_F2,
  SCHRO_CTX_ZP_F3,
  SCHRO_CTX_ZP_F4,
  SCHRO_CTX_ZP_F5,
  SCHRO_CTX_ZP_F6p,
  SCHRO_CTX_ZP_F6p,
  SCHRO_CTX_NP_F2,
  SCHRO_CTX_NP_F2,
  SCHRO_CTX_NP_F3,
  SCHRO_CTX_NP_F4,
  SCHRO_CTX_NP_F5,
  SCHRO_CTX_NP_F6p,
  SCHRO_CTX_NP_F6p,
  0,
  0,
  0,
  0,

  SCHRO_CTX_SB_F2,
  SCHRO_CTX_SB_F2,
  0,
  0,
  0,
  0,
  SCHRO_CTX_LUMA_DC_CONT_BIN2,
  SCHRO_CTX_LUMA_DC_CONT_BIN2,
  0,
  0,
  SCHRO_CTX_CHROMA1_DC_CONT_BIN2,
  SCHRO_CTX_CHROMA1_DC_CONT_BIN2,
  0,
  0,
  SCHRO_CTX_CHROMA2_DC_CONT_BIN2,
  SCHRO_CTX_CHROMA2_DC_CONT_BIN2,
  0,
  0,
  SCHRO_CTX_MV_REF1_H_CONT_BIN2,
  SCHRO_CTX_MV_REF1_H_CONT_BIN3,
  SCHRO_CTX_MV_REF1_H_CONT_BIN4,
  SCHRO_CTX_MV_REF1_H_CONT_BIN5,
  SCHRO_CTX_MV_REF1_H_CONT_BIN5,
  0,
  0,
  SCHRO_CTX_MV_REF1_V_CONT_BIN2,
  SCHRO_CTX_MV_REF1_V_CONT_BIN3,
  SCHRO_CTX_MV_REF1_V_CONT_BIN4,
  SCHRO_CTX_MV_REF1_V_CONT_BIN5,
  SCHRO_CTX_MV_REF1_V_CONT_BIN5,
  0,
  0,
  SCHRO_CTX_MV_REF2_H_CONT_BIN2,
  SCHRO_CTX_MV_REF2_H_CONT_BIN3,
  SCHRO_CTX_MV_REF2_H_CONT_BIN4,
  SCHRO_CTX_MV_REF2_H_CONT_BIN5,
  SCHRO_CTX_MV_REF2_H_CONT_BIN5,
  0,
  0,
  SCHRO_CTX_MV_REF2_V_CONT_BIN2,
  SCHRO_CTX_MV_REF2_V_CONT_BIN3,
  SCHRO_CTX_MV_REF2_V_CONT_BIN4,
  SCHRO_CTX_MV_REF2_V_CONT_BIN5,
  SCHRO_CTX_MV_REF2_V_CONT_BIN5,
  0,
  0,
  0,
};
  
void
schro_arith_init_contexts (SchroArith *arith)
{
  int i;
  for(i=0;i<SCHRO_CTX_LAST;i++){
    arith->contexts[i].count[0] = 1;
    arith->contexts[i].count[1] = 1;
    arith->contexts[i].next = next_list[i];
    arith->contexts[i].n = 1;
    arith->contexts[i].probability = 0x8000;
  }
}

void
schro_arith_halve_all_counts (SchroArith *arith)
{
  int i;
  for(i=0;i<SCHRO_CTX_LAST;i++) {
#if DIRAC_COMPAT
    if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 16) {
      arith->contexts[i].count[0] >>= 1;
      arith->contexts[i].count[0]++;
      arith->contexts[i].count[1] >>= 1;
      arith->contexts[i].count[1]++;
    }
#else
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
#endif
  }
}

int
_schro_arith_context_decode_bit (SchroArith *arith, int i)
{
  return __schro_arith_context_decode_bit (arith, i);
}

static int
__schro_arith_context_decode_bit (SchroArith *arith, int i)
{
  unsigned int range;
  unsigned int probability0;
  unsigned int range_x_prob;
  unsigned int count;
  int value;

  probability0 = arith->contexts[i].probability;
  count = arith->code - arith->range[0] + 1;
  range = arith->range[1];
  range_x_prob = (range * probability0) >> 16;

  value = (count > range_x_prob);
  if (value) {
    arith->range[0] = arith->range[0] + range_x_prob;
    arith->range[1] -= range_x_prob;
  } else {
    arith->range[1] = range_x_prob;
  }
  arith->contexts[i].count[value]++;
  arith->contexts[i].n--;
  if (arith->contexts[i].n == 0) {
    unsigned int scaler;
    unsigned int weight;

    if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
      arith->contexts[i].count[0] >>= 1;
      arith->contexts[i].count[0]++;
      arith->contexts[i].count[1] >>= 1;
      arith->contexts[i].count[1]++;
    }
    weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
    scaler = schro_table_division_factor[weight];
    arith->contexts[i].probability = arith->contexts[i].count[0] * scaler;

    if (weight > 16) {
      arith->contexts[i].n = 16;
    } else {
      arith->contexts[i].n = 1;
    }
  }

  while (arith->range[1] < 0x1000) {
    arith->range[0] <<= 1;
    arith->range[1] <<= 1;

    arith->code <<= 1;
    arith->code |= (arith->dataptr[arith->offset] >> (7-arith->cntr))&1;

    arith->cntr++;

    if (arith->cntr == 8) {
      arith->offset++;
      arith->range[0] &= 0xffff;
      arith->code &= 0xffff;

      if (arith->code < arith->range[0]) {
        arith->code |= (1<<16);
      }
      arith->cntr = 0;
    }
  }

  return value;
}

void
_schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  unsigned int range;
  unsigned int probability0;
  unsigned int range_x_prob;

  probability0 = arith->contexts[i].probability;
  range = arith->range[1];
  range_x_prob = (range * probability0) >> 16;

  if (value) {
    arith->range[0] = arith->range[0] + range_x_prob;
    arith->range[1] -= range_x_prob;
  } else {
    arith->range[1] = range_x_prob;
  }
  arith->contexts[i].count[value]++;
  arith->contexts[i].n--;
  if (arith->contexts[i].n == 0) {
    unsigned int scaler;
    unsigned int weight;

    if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
      arith->contexts[i].count[0] >>= 1;
      arith->contexts[i].count[0]++;
      arith->contexts[i].count[1] >>= 1;
      arith->contexts[i].count[1]++;
    }
    weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
    scaler = schro_table_division_factor[weight];
    arith->contexts[i].probability = arith->contexts[i].count[0] * scaler;

    if (weight > 16) {
      arith->contexts[i].n = 16;
    } else {
      arith->contexts[i].n = 1;
    }
  }

  while (arith->range[1] < 0x1000) {
    arith->range[0] <<= 1;
    arith->range[1] <<= 1;
    arith->cntr++;

    if (arith->cntr == 8) {
      if (arith->range[0] < (1<<24) &&
          (arith->range[0] + arith->range[1]) >= (1<<24)) {
        arith->carry++;
      } else {
        if (arith->range[0] >= (1<<24)) {
          arith->dataptr[arith->offset-1]++;
          while (arith->carry) {
            arith->dataptr[arith->offset] = 0x00;
            arith->carry--;
            arith->offset++;
          }
        } else {
          while (arith->carry) {
            arith->dataptr[arith->offset] = 0xff;
            arith->carry--;
            arith->offset++;
          }
        }
        arith->dataptr[arith->offset] = arith->range[0] >> 16;
        arith->offset++;
      }

      arith->range[0] &= 0xffff;
      arith->cntr = 0;
    }
  }
}

static int
maxbit (unsigned int x)
{
  int i;
  for(i=0;x;i++){
    x >>= 1;
  }
  return i;
}

void
_schro_arith_context_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    _schro_arith_context_encode_bit (arith, cont_context, 0);
    _schro_arith_context_encode_bit (arith, value_context,
        (value>>(n_bits - 2 - i))&1);
    cont_context = arith->contexts[cont_context].next;
  }
  _schro_arith_context_encode_bit (arith, cont_context, 1);
}

void
_schro_arith_context_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value)
{
  int sign;

  if (value < 0) {
    sign = 1;
    value = -value;
  } else {
    sign = 0;
  }
  _schro_arith_context_encode_uint (arith, cont_context, value_context, value);
  if (value) {
    _schro_arith_context_encode_bit (arith, sign_context, sign);
  }
}

int
_schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context)
{
  int bits;
  int count=0;

  bits = 0;
  while(!__schro_arith_context_decode_bit (arith, cont_context)) {
    bits <<= 1;
    bits |= __schro_arith_context_decode_bit (arith, value_context);
    cont_context = arith->contexts[cont_context].next;
    count++;

    /* FIXME being careful */
    if (count == 30) break;
  }
  return (1<<count) - 1 + bits;
}

int
_schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context)
{
  int bits;
  int count=0;
  int value;

  bits = 0;
  while(!__schro_arith_context_decode_bit (arith, cont_context)) {
    bits <<= 1;
    bits |= __schro_arith_context_decode_bit (arith, value_context);
    cont_context = arith->contexts[cont_context].next;
    count++;

    /* FIXME being careful */
    if (count == 30) break;
  }
  value = (1<<count) - 1 + bits;

  if (value) {
    if (__schro_arith_context_decode_bit (arith, sign_context)) {
      value = -value;
    }
  }

  return value;
}

/* wrappers */

void
schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  _schro_arith_context_encode_bit (arith, i, value);
}

void
schro_arith_context_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value)
{
  _schro_arith_context_encode_uint (arith, cont_context, value_context, value);
}

void
schro_arith_context_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value)
{
  _schro_arith_context_encode_sint (arith, cont_context, value_context,
      sign_context, value);
}

int
schro_arith_context_decode_bit (SchroArith *arith, int context)
{
  return __schro_arith_context_decode_bit (arith, context);
}

int
schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context)
{
  return _schro_arith_context_decode_uint (arith, cont_context, value_context);
}

int
schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context)
{
  return _schro_arith_context_decode_sint (arith, cont_context,
      value_context, sign_context);
}

