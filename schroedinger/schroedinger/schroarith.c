
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>



static unsigned int division_factor[256];

static void schro_arith_input_bit (SchroArith *arith);
static void schro_arith_output_bit (SchroArith *arith);

static void
_schro_arith_division_factor_init (void)
{
  static int inited = 0;

  if (!inited) {
    int i;
    for(i=0;i<256;i++){
      division_factor[i] = (1U<<31)/(i+1);
    }
  }
}

SchroArith *
schro_arith_new (void)
{
  SchroArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  _schro_arith_division_factor_init();

  return arith;
}

void
schro_arith_free (SchroArith *arith)
{
  free(arith);
}

void
schro_arith_decode_init (SchroArith *arith, SchroBits *bits)
{
  int i;

  arith->low = 0;
  arith->high = 0xffff;
  arith->code = 0;

  arith->bits = bits;

  for(i=0;i<16;i++){
    schro_arith_input_bit(arith);
  }
}

void
schro_arith_encode_init (SchroArith *arith, SchroBits *bits)
{
  arith->low = 0;
  arith->high = 0xffff;
  arith->code = 0;

  arith->bits = bits;
}

static void
schro_arith_input_bit (SchroArith *arith)
{
  arith->code <<= 1;
  arith->code += schro_bits_decode_bit(arith->bits);
}

static void
schro_arith_push_bit (SchroArith *arith, int value)
{
  schro_bits_encode_bit (arith->bits, value);
}

static void
schro_arith_output_bit (SchroArith *arith)
{
  int value;

  value = arith->low >> 15;

  schro_arith_push_bit (arith, value);
  
  value = !value;
  while(arith->cntr) {
    schro_arith_push_bit (arith, value);
    arith->cntr--;
  }
}

void
schro_arith_flush (SchroArith *arith)
{
  schro_arith_push_bit (arith, 1);
  schro_bits_sync (arith->bits);
  arith->bits->offset += 16;
}

void
schro_arith_init_contexts (SchroArith *arith)
{
  int i;
  for(i=0;i<SCHRO_CTX_LAST;i++){
    schro_arith_context_init (arith, i, 1, 1);
  }
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
schro_arith_context_init (SchroArith *arith, int i, int count0, int count1)
{
  arith->contexts[i].count0 = count0;
  arith->contexts[i].count1 = count1;
  arith->contexts[i].next = next_list[i];
}


void
schro_arith_context_halve_counts (SchroArith *arith, int i)
{
  arith->contexts[i].count0 >>= 1;
  arith->contexts[i].count0++;
  arith->contexts[i].count1 >>= 1;
  arith->contexts[i].count1++;
}

void
schro_arith_context_halve_all_counts (SchroArith *arith)
{
  int i;
  for(i=0;i<arith->n_contexts;i++) {
    arith->contexts[i].count0 >>= 1;
    arith->contexts[i].count0++;
    arith->contexts[i].count1 >>= 1;
    arith->contexts[i].count1++;
  }
}

static void
schro_arith_context_update (SchroArith *arith, int i, int value)
{
  if (value) {
    arith->contexts[i].count1++;
  } else {
    arith->contexts[i].count0++;
  }
  if (arith->contexts[i].count0 + arith->contexts[i].count1 > 255) {
    schro_arith_context_halve_counts (arith, i);
  }
}




static int
schro_arith_context_binary_decode (SchroArith *arith, int i)
{
  unsigned int count;
  int value;
  int range;
  int scaled_count0;
  int weight;

  count = ((arith->code - arith->low + 1)<<10) - 1;
  range = arith->high - arith->low + 1;
  weight = arith->contexts[i].count0 + arith->contexts[i].count1;
  scaled_count0 = ((unsigned int)arith->contexts[i].count0 * division_factor[weight - 1]) >> 21;
  if (count < range * scaled_count0) {
    value = 0;
  } else {
    value = 1;
  }

  schro_arith_context_update (arith, i, value);
  
  if (value == 0) {
    arith->high = arith->low + ((range * scaled_count0)>>10) - 1;
  } else {
    arith->low = arith->low + ((range * scaled_count0)>>10);
  }

  do {
    if ((arith->high & (1<<15)) == (arith->low & (1<<15))) {
      /* do nothing */
    } else if ((arith->low & (1<<14)) && !(arith->high & (1<<14))) {
      arith->code ^= (1<<14);
      arith->low ^= (1<<14);
      arith->high ^= (1<<14);
    } else {
      break;
    }

    arith->low <<= 1;
    arith->high <<= 1;
    arith->high++;

    schro_arith_input_bit(arith);
  } while (1);

  return value;
}


static void
schro_arith_context_binary_encode (SchroArith *arith, int i, int value)
{
  int range;
  int scaled_count0;
  int weight;

  range = arith->high - arith->low + 1;
  weight = arith->contexts[i].count0 + arith->contexts[i].count1;
  scaled_count0 = ((unsigned int)arith->contexts[i].count0 * division_factor[weight - 1]) >> 21;

  schro_arith_context_update (arith, i, value);
  
  if (value == 0) {
    arith->high = arith->low + ((range * scaled_count0)>>10) - 1;
  } else {
    arith->low = arith->low + ((range * scaled_count0)>>10);
  }

  do {
    if ((arith->high & (1<<15)) == (arith->low & (1<<15))) {
      schro_arith_output_bit(arith);

      arith->low <<= 1;
      arith->high <<= 1;
      arith->high++;
    } else if ((arith->low & (1<<14)) && !(arith->high & (1<<14))) {
      arith->low ^= (1<<14);
      arith->high ^= (1<<14);

      arith->low <<= 1;
      arith->high <<= 1;
      arith->high++;
      arith->cntr++;
    } else {
      break;
    }

  } while (1);
}




void
schro_arith_context_encode_bit (SchroArith *arith, int context, int value)
{
  schro_arith_context_binary_encode (arith, context, value);
}

void
schro_arith_context_encode_uu (SchroArith *arith, int context, int context2, int value)
{
  switch (value) {
    case 0:
      schro_arith_context_binary_encode (arith, context, 1);
      break;
    case 1:
      schro_arith_context_binary_encode (arith, context, 0);
      schro_arith_context_binary_encode (arith, context2, 1);
      break;
    case 2:
      schro_arith_context_binary_encode (arith, context, 0);
      schro_arith_context_binary_encode (arith, context2, 0);
      schro_arith_context_binary_encode (arith, context2 + 1, 1);
      break;
    case 3:
      schro_arith_context_binary_encode (arith, context, 0);
      schro_arith_context_binary_encode (arith, context2, 0);
      schro_arith_context_binary_encode (arith, context2 + 1, 0);
      schro_arith_context_binary_encode (arith, context2 + 2, 1);
      break;
    default:
      schro_arith_context_binary_encode (arith, context, 0);
      schro_arith_context_binary_encode (arith, context2, 0);
      schro_arith_context_binary_encode (arith, context2 + 1, 0);
      schro_arith_context_binary_encode (arith, context2 + 2, 0);
      value -= 4;
      while (value > 0) {
        schro_arith_context_binary_encode (arith, context2 + 3, 0);
        value--;
      }
      schro_arith_context_binary_encode (arith, context2 + 3, 1);
      break;
  }
}

void
schro_arith_context_encode_su (SchroArith *arith, int context, int value)
{
  int i;
  int sign;

  if (value==0) {
    schro_arith_context_binary_encode (arith, context, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  schro_arith_context_binary_encode (arith, context, 1);
  schro_arith_context_binary_encode (arith, context, sign);
}

void
schro_arith_context_encode_ut (SchroArith *arith, int context, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  if (value < max) {
    schro_arith_context_binary_encode (arith, context, 1);
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
schro_arith_context_encode_uegol (SchroArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, 0);
  }
  schro_arith_context_binary_encode (arith, context, 1);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_arith_context_encode_segol (SchroArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_arith_context_encode_uegol (arith, context, value);
  if (value) {
    schro_arith_context_binary_encode (arith, context, sign);
  }
}

void
schro_arith_context_encode_ue2gol (SchroArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  schro_arith_context_encode_uegol (arith, context, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_arith_context_encode_se2gol (SchroArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_arith_context_encode_ue2gol (arith, context, value);
  if (value) {
    schro_arith_context_binary_encode (arith, context, sign);
  }
}

void
schro_arith_context_encode_uint (SchroArith *arith, int cont_context,
    int value_context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    schro_arith_context_binary_encode (arith, cont_context, 0);
    schro_arith_context_binary_encode (arith, value_context,
        (value>>(n_bits - 2 - i))&1);
    cont_context = arith->contexts[cont_context].next;
  }
  schro_arith_context_binary_encode (arith, cont_context, 1);
}

void
schro_arith_context_encode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_arith_context_encode_uint (arith, cont_context, value_context, value);
  if (value) {
    schro_arith_context_binary_encode (arith, sign_context, sign);
  }
}




int schro_arith_context_decode_bit (SchroArith *arith, int context)
{
  return schro_arith_context_binary_decode (arith, context);
}

int schro_arith_context_decode_bits (SchroArith *arith, int context, int n)
{
  int value = 0;
  int i;
  
  for(i=0;i<n;i++){
    value = (value << 1) | schro_arith_context_binary_decode (arith, context);
  } 
  
  return value;
}

int schro_arith_context_decode_uu (SchroArith *arith, int context, int context2)
{
  int value;

  if (schro_arith_context_binary_decode (arith, context)) return 0;
  if (schro_arith_context_binary_decode (arith, context2)) return 1;
  if (schro_arith_context_binary_decode (arith, context2 + 1)) return 2;
  if (schro_arith_context_binary_decode (arith, context2 + 2)) return 3;
  value = 4;
  while (schro_arith_context_binary_decode (arith, context2 + 3) == 0) {
    value++;
  }
  return value;
}

int schro_arith_context_decode_su (SchroArith *arith, int context)
{
  int value = 0;

  if (schro_arith_context_binary_decode (arith, context) == 1) {
    return 0;
  }
  value = 1;
  while (schro_arith_context_binary_decode (arith, context) == 0) {
    value++;
  }
  if (schro_arith_context_binary_decode (arith, context) == 0) {
    value = -value;
  }

  return value;
}

int schro_arith_context_decode_ut (SchroArith *arith, int context, int max)
{
  int value;

  for(value=0;value<max;value++){
    if (schro_arith_context_binary_decode (arith, context)) {
      return value;
    }
  }
  return value;
}

int schro_arith_context_decode_uegol (SchroArith *arith, int context)
{
  int count;
  int value;

  count = 0;
  while(!schro_arith_context_binary_decode (arith, context)) {
    count++;
  }
  value = (1<<count) - 1 + schro_arith_context_decode_bits (arith, context, count);

  return value;
}

int schro_arith_context_decode_segol (SchroArith *arith, int context)
{
  int value;

  value = schro_arith_context_decode_uegol (arith, context);
  if (value) {
    if (!schro_arith_context_binary_decode (arith, context)) {
      value = -value;
    }
  }

  return value;
}

int schro_arith_context_decode_ue2gol (SchroArith *arith, int context)
{
  int count;
  int value;

  count = schro_arith_context_decode_uegol (arith, context);
  value = (1<<count) - 1 + schro_arith_context_decode_bits (arith, context, count);

  return value;
}

int schro_arith_context_decode_se2gol (SchroArith *arith, int context)
{
  int value;

  value = schro_arith_context_decode_ue2gol (arith, context);
  if (value) {
    if (!schro_arith_context_binary_decode (arith, context)) {
      value = -value;
    }
  }

  return value;
}

int schro_arith_context_decode_uint (SchroArith *arith, int cont_context,
    int value_context)
{
  int bits;
  int count=0;

  bits = 0;
  while(!schro_arith_context_binary_decode (arith, cont_context)) {
    bits <<= 1;
    bits |= schro_arith_context_binary_decode (arith, value_context);
    cont_context = arith->contexts[cont_context].next;
    count++;
  }
  return (1<<count) - 1 + bits;
}

int
schro_arith_context_decode_sint (SchroArith *arith, int cont_context,
    int value_context, int sign_context)
{
  int value;

  value = schro_arith_context_decode_uint (arith, cont_context, value_context);
  if (value) {
    if (!schro_arith_context_binary_decode (arith, sign_context)) {
      value = -value;
    }
  }

  return value;
}

