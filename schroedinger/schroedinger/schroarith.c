
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>


static void _schro_arith_output_bit (SchroArith *arith);

SchroArith *
schro_arith_new (void)
{
  SchroArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  memcpy (arith->division_factor, schro_table_division_factor,
      sizeof(arith->division_factor));
  memcpy (arith->fixup_shift, schro_table_arith_shift,
      sizeof(arith->fixup_shift));

  return arith;
}

void
schro_arith_free (SchroArith *arith)
{
  free(arith);
}

static void
schro_arith_reload_nextcode (SchroArith *arith)
{
  while(arith->nextbits <= 24) {
    if (arith->offset < arith->buffer->length) {
      arith->nextcode |= 
        arith->buffer->data[arith->offset] << (24-arith->nextbits);
    } else {
      arith->nextcode |= 0xff << (24-arith->nextbits);
    }
    arith->nextbits+=8;
    arith->offset++;
  }
}

void
schro_arith_decode_init (SchroArith *arith, SchroBuffer *buffer)
{
  arith->range[0] = 0;
  arith->range[1] = 0xffff;
  arith->code = 0;

  arith->buffer = buffer;

  arith->dataptr = arith->buffer->data;
  arith->maxdataptr = arith->buffer->data + arith->buffer->length;
  arith->code = arith->dataptr[0] << 8;
  arith->code |= arith->dataptr[1];
  arith->offset = 2;
  schro_arith_reload_nextcode(arith);
}

void
schro_arith_encode_init (SchroArith *arith, SchroBuffer *buffer)
{
  arith->range[0] = 0;
  arith->range[1] = 0xffff;
  arith->code = 0;

  arith->buffer = buffer;
  arith->offset = 0;
  arith->nextbits = 0;
  arith->nextcode = 0;
}

static void
_schro_arith_push_bit (SchroArith *arith, int value)
{
  arith->nextcode <<= 1;
  arith->nextcode |= value;
  arith->nextbits++;
  if (arith->nextbits == 8) {
    arith->buffer->data[arith->offset] = arith->nextcode;
    arith->nextbits = 0;
    arith->nextcode = 0;
    arith->offset++;
  }
}

static void
_schro_arith_output_bit (SchroArith *arith)
{
  int value;

  value = arith->range[0] >> 15;

  _schro_arith_push_bit (arith, value);
  
  value = !value;
  while(arith->cntr) {
    _schro_arith_push_bit (arith, value);
    arith->cntr--;
  }
}

void
schro_arith_flush (SchroArith *arith)
{
  int i;
  for(i=0;i<16;i++){
    _schro_arith_output_bit (arith);
    arith->range[0] <<= 1;
  }
  while(arith->nextbits < 8) {
    arith->nextcode <<= 1;
    arith->nextbits++;
  }
  arith->buffer->data[arith->offset] = arith->nextcode;
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

  0,
  0,
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
  unsigned int count;
  unsigned int value;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;

  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = arith->division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  count = arith->code - arith->range[0] + 1;
  range = arith->range[1] - arith->range[0] + 1;
  range_x_prob = (range * probability0) >> 16;
  value = (count > range_x_prob);

  arith->range[1 - value] = arith->range[0] + range_x_prob - 1 + value;
  arith->contexts[i].count[value]++;

  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }

  do {
    if ((arith->range[1] & (1<<15)) == (arith->range[0] & (1<<15))) {
      /* do nothing */
    } else if ((arith->range[0] & (1<<14)) && !(arith->range[1] & (1<<14))) {
      arith->code ^= (1<<14);
      arith->range[0] ^= (1<<14);
      arith->range[1] ^= (1<<14);
    } else {
      break;
    }

    arith->range[0] <<= 1;
    arith->range[1] <<= 1;
    arith->range[1]++;

    arith->code <<= 1;
    arith->code |= (arith->nextcode >> 31);
    arith->nextcode <<= 1;
    arith->nextbits--;
    if (arith->nextbits == 0) {
      schro_arith_reload_nextcode(arith);
    }
  } while (1);

  return value;
}


void
_schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  unsigned int count;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;

  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = arith->division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  count = arith->code - arith->range[0] + 1;
  range = arith->range[1] - arith->range[0] + 1;
  range_x_prob = (range * probability0) >> 16;
//SCHRO_ERROR("scaler %d count0 %d prob0 %d range_x_prob %d", scaler, arith->contexts[i].count[0], probability0, range_x_prob);
  //value = (count > range_x_prob);

  arith->range[1 - value] = arith->range[0] + range_x_prob - 1 + value;
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }

  do {
    if ((arith->range[1] & (1<<15)) == (arith->range[0] & (1<<15))) {
      _schro_arith_output_bit(arith);

      arith->range[0] <<= 1;
      arith->range[1] <<= 1;
      arith->range[1]++;
    } else if ((arith->range[0] & (1<<14)) && !(arith->range[1] & (1<<14))) {
      arith->range[0] ^= (1<<14);
      arith->range[1] ^= (1<<14);

      arith->range[0] <<= 1;
      arith->range[1] <<= 1;
      arith->range[1]++;
      arith->cntr++;
    } else {
      break;
    }

  } while (1);
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

void
_schro_arith_encode_mode (SchroArith *arith, int context0, int context1,
    int value)
{
  switch (value) {
    case 0:
      _schro_arith_context_encode_bit (arith, context0, 1);
      break;
    case 1:
      _schro_arith_context_encode_bit (arith, context0, 0);
      _schro_arith_context_encode_bit (arith, context1, 1);
      break;
    case 2:
      _schro_arith_context_encode_bit (arith, context0, 0);
      _schro_arith_context_encode_bit (arith, context1, 0);
      break;
  }
}


int
_schro_arith_context_decode_bits (SchroArith *arith, int context, int n)
{
  int value = 0;
  int i;
  
  for(i=0;i<n;i++){
    value = (value << 1) | _schro_arith_context_decode_bit (arith, context);
  } 
  
  return value;
}

int
_schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context)
{
  int bits;
  int count=0;

  bits = 0;
  while(!_schro_arith_context_decode_bit (arith, cont_context)) {
    bits <<= 1;
    bits |= _schro_arith_context_decode_bit (arith, value_context);
    cont_context = arith->contexts[cont_context].next;
    count++;
  }
  return (1<<count) - 1 + bits;
}

int
_schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context)
{
  int value;

  value = _schro_arith_context_decode_uint (arith, cont_context, value_context);
  if (value) {
    if (_schro_arith_context_decode_bit (arith, sign_context)) {
      value = -value;
    }
  }

  return value;
}

int
_schro_arith_decode_mode (SchroArith *arith, int context0, int context1)
{
  if (_schro_arith_context_decode_bit (arith, context0)) {
    return 0;
  }
  if (_schro_arith_context_decode_bit (arith, context1)) {
    return 1;
  }
  return 2;
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

void
schro_arith_encode_mode (SchroArith *arith, int context0, int context1,
    int value)
{
  _schro_arith_encode_mode (arith, context0, context1, value);
}

int
schro_arith_context_decode_bit (SchroArith *arith, int context)
{
  return _schro_arith_context_decode_bit (arith, context);
}

int
schro_arith_context_decode_bits (SchroArith *arith, int context, int n)
{
  return _schro_arith_context_decode_bits (arith, context, n);
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

int
schro_arith_decode_mode (SchroArith *arith, int context0, int context1)
{
  return _schro_arith_decode_mode (arith, context0, context1);
}

