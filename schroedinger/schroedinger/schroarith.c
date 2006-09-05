
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>


static void _schro_arith_input_bit (SchroArith *arith);
static void _schro_arith_output_bit (SchroArith *arith);

SchroArith *
schro_arith_new (void)
{
  SchroArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  arith->division_factor = schro_table_division_factor;

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
  arith->range[0] = 0;
  arith->range[1] = 0xffff;
  arith->code = 0;

  arith->buffer = buffer;

  arith->code = arith->buffer->data[0] << 8;
  arith->code |= arith->buffer->data[1];
  arith->nextcode = arith->buffer->data[2];
  arith->nextbits = 8;
  arith->offset=3;
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
_schro_arith_input_bit (SchroArith *arith)
{
  arith->code <<= 1;
  arith->code |= (arith->nextcode >> 7);
  arith->nextcode <<= 1;
  arith->nextbits--;
  if (arith->nextbits == 0) {
    arith->nextbits = 8;
    arith->nextcode = arith->buffer->data[arith->offset];
    arith->offset++;
  }
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
  _schro_arith_push_bit (arith, 1);
  while(arith->nextbits < 8) {
    arith->nextcode <<= 1;
    arith->nextbits++;
  }
  arith->buffer->data[arith->offset] = arith->nextcode;
  arith->nextbits = 0;
  arith->nextcode = 0;
  arith->offset++;
  arith->buffer->data[arith->offset] = arith->nextcode;
  arith->offset++;
  arith->buffer->data[arith->offset] = arith->nextcode;
  arith->offset++;
}

static const int next_list[] = {
  0,
  SCHRO_CTX_QUANTISER_CONT,
  0,
  0,
  SCHRO_CTX_Z_BIN2,
  SCHRO_CTX_Z_BIN2,
  SCHRO_CTX_Z_BIN3,
  SCHRO_CTX_Z_BIN4,
  SCHRO_CTX_Z_BIN5,
  SCHRO_CTX_Z_BIN5,
  0,
  0,
  0,
  0,
  SCHRO_CTX_NZ_BIN2,
  SCHRO_CTX_NZ_BIN2,
  SCHRO_CTX_NZ_BIN2,
  SCHRO_CTX_NZ_BIN3,
  SCHRO_CTX_NZ_BIN4,
  SCHRO_CTX_NZ_BIN5,
  SCHRO_CTX_NZ_BIN5,
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

static void
_schro_arith_context_halve_counts (SchroArith *arith, int i)
{
  arith->contexts[i].count[0] >>= 1;
  arith->contexts[i].count[0]++;
  arith->contexts[i].count[1] >>= 1;
  arith->contexts[i].count[1]++;
}

void
schro_arith_halve_all_counts (SchroArith *arith)
{
  int i;
  for(i=0;i<arith->n_contexts;i++) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }
}

static void
_schro_arith_context_update (SchroArith *arith, int i, int value)
{
  arith->contexts[i].count[value]++;
  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    _schro_arith_context_halve_counts (arith, i);
  }
}



int
_schro_arith_context_decode_bit (SchroArith *arith, int i)
{
  unsigned int count;
  int value;
  int range;
  int scaled_count0;
  int weight;

  count = ((arith->code - arith->range[0] + 1)<<10) - 1;
  range = arith->range[1] - arith->range[0] + 1;
  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaled_count0 = ((unsigned int)arith->contexts[i].count[0] *
      arith->division_factor[weight - 1]) >> 21;
  value = (count >= range * scaled_count0);

  arith->contexts[i].count[value]++;
  if (arith->contexts[i].count[0] + arith->contexts[i].count[1] > 255) {
    arith->contexts[i].count[0] >>= 1;
    arith->contexts[i].count[0]++;
    arith->contexts[i].count[1] >>= 1;
    arith->contexts[i].count[1]++;
  }
  
  {
    int newval = arith->range[0] + ((range * scaled_count0)>>10);
    arith->range[1 - value] = newval - 1 + value;
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

    _schro_arith_input_bit(arith);
  } while (1);

  return value;
}


void
_schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  int range;
  int scaled_count0;
  int weight;

  range = arith->range[1] - arith->range[0] + 1;
  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaled_count0 = ((unsigned int)arith->contexts[i].count[0] *
      arith->division_factor[weight - 1]) >> 21;

  _schro_arith_context_update (arith, i, value);
  
  arith->range[1-value] = arith->range[0] +
    ((range * scaled_count0)>>10) - 1 + value;

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
    sign = 0;
    value = -value;
  } else {
    sign = 1;
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
    if (!_schro_arith_context_decode_bit (arith, sign_context)) {
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

