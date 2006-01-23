
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <carid/caridarith.h>

#pragma GCC visibility push(hidden)


static unsigned int division_factor[1024];

static void carid_arith_input_bit (CaridArith *arith);
static void carid_arith_output_bit (CaridArith *arith);

static void
_carid_arith_division_factor_init (void)
{
  static int inited = 0;

  if (!inited) {
    int i;
    for(i=0;i<1024;i++){
      division_factor[i] = (1U<<31)/(i+1);
    }
  }
}

CaridArith *
carid_arith_new (void)
{
  CaridArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  _carid_arith_division_factor_init();

  return arith;
}

void
carid_arith_free (CaridArith *arith)
{
  free(arith);
}

void
carid_arith_decode_init (CaridArith *arith, CaridBits *bits)
{
  int i;

  arith->low = 0;
  arith->high = 0xffff;
  arith->code = 0;

  arith->bits = bits;

  for(i=0;i<16;i++){
    carid_arith_input_bit(arith);
  }
}

void
carid_arith_encode_init (CaridArith *arith, CaridBits *bits)
{
  arith->low = 0;
  arith->high = 0xffff;
  arith->code = 0;

  arith->bits = bits;
}

static void
carid_arith_input_bit (CaridArith *arith)
{
  arith->code <<= 1;
  arith->code += carid_bits_decode_bit(arith->bits);
}

static void
carid_arith_push_bit (CaridArith *arith, int value)
{
  carid_bits_encode_bit (arith->bits, value);
}

static void
carid_arith_output_bit (CaridArith *arith)
{
  int value;

  value = arith->low >> 15;

  carid_arith_push_bit (arith, value);
  
  value = !value;
  while(arith->cntr) {
    carid_arith_push_bit (arith, value);
    arith->cntr--;
  }
}

void
carid_arith_flush (CaridArith *arith)
{
  carid_arith_push_bit (arith, 1);
  carid_bits_sync (arith->bits);
  arith->bits->offset += 16;
}

void
carid_arith_init_contexts (CaridArith *arith)
{
  int i;
  for(i=0;i<CARID_CTX_LAST;i++){
    carid_arith_context_init (arith, i, 1, 1);
  }
}
void
carid_arith_context_init (CaridArith *arith, int i, int count0, int count1)
{
  arith->contexts[i].count0 = count0;
  arith->contexts[i].count1 = count1;
}


void
carid_arith_context_halve_counts (CaridArith *arith, int i)
{
  arith->contexts[i].count0 >>= 1;
  arith->contexts[i].count0++;
  arith->contexts[i].count1 >>= 1;
  arith->contexts[i].count1++;
}

void
carid_arith_context_halve_all_counts (CaridArith *arith)
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
carid_arith_context_update (CaridArith *arith, int i, int value)
{
  if (value) {
    arith->contexts[i].count1++;
  } else {
    arith->contexts[i].count0++;
  }
  if (arith->contexts[i].count0 + arith->contexts[i].count1 >= 1024) {
    carid_arith_context_halve_counts (arith, i);
  }
}




static int
carid_arith_context_binary_decode (CaridArith *arith, int i)
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

  carid_arith_context_update (arith, i, value);
  
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

    carid_arith_input_bit(arith);
  } while (1);

  return value;
}


static void
carid_arith_context_binary_encode (CaridArith *arith, int i, int value)
{
  int range;
  int scaled_count0;
  int weight;

  range = arith->high - arith->low + 1;
  weight = arith->contexts[i].count0 + arith->contexts[i].count1;
  scaled_count0 = ((unsigned int)arith->contexts[i].count0 * division_factor[weight - 1]) >> 21;

  carid_arith_context_update (arith, i, value);
  
  if (value == 0) {
    arith->high = arith->low + ((range * scaled_count0)>>10) - 1;
  } else {
    arith->low = arith->low + ((range * scaled_count0)>>10);
  }

  do {
    if ((arith->high & (1<<15)) == (arith->low & (1<<15))) {
      carid_arith_output_bit(arith);

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
carid_arith_context_encode_bit (CaridArith *arith, int context, int value)
{
  carid_arith_context_binary_encode (arith, context, value);
}

void
carid_arith_context_encode_uu (CaridArith *arith, int context, int context2, int value)
{
  switch (value) {
    case 0:
      carid_arith_context_binary_encode (arith, context, 1);
      break;
    case 1:
      carid_arith_context_binary_encode (arith, context, 0);
      carid_arith_context_binary_encode (arith, context2, 1);
      break;
    case 2:
      carid_arith_context_binary_encode (arith, context, 0);
      carid_arith_context_binary_encode (arith, context2, 0);
      carid_arith_context_binary_encode (arith, context2 + 1, 1);
      break;
    case 3:
      carid_arith_context_binary_encode (arith, context, 0);
      carid_arith_context_binary_encode (arith, context2, 0);
      carid_arith_context_binary_encode (arith, context2 + 1, 0);
      carid_arith_context_binary_encode (arith, context2 + 2, 1);
      break;
    default:
      carid_arith_context_binary_encode (arith, context, 0);
      carid_arith_context_binary_encode (arith, context2, 0);
      carid_arith_context_binary_encode (arith, context2 + 1, 0);
      carid_arith_context_binary_encode (arith, context2 + 2, 0);
      value -= 4;
      while (value > 0) {
        carid_arith_context_binary_encode (arith, context2 + 3, 0);
        value--;
      }
      carid_arith_context_binary_encode (arith, context2 + 3, 1);
      break;
  }
}

void
carid_arith_context_encode_su (CaridArith *arith, int context, int value)
{
  int i;
  int sign;

  if (value==0) {
    carid_arith_context_binary_encode (arith, context, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  carid_arith_context_binary_encode (arith, context, 1);
  carid_arith_context_binary_encode (arith, context, sign);
}

void
carid_arith_context_encode_ut (CaridArith *arith, int context, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  if (value < max) {
    carid_arith_context_binary_encode (arith, context, 1);
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
carid_arith_context_encode_uegol (CaridArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, 0);
  }
  carid_arith_context_binary_encode (arith, context, 1);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_arith_context_encode_segol (CaridArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_arith_context_encode_uegol (arith, context, value);
  if (value) {
    carid_arith_context_binary_encode (arith, context, sign);
  }
}

void
carid_arith_context_encode_ue2gol (CaridArith *arith, int context, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  carid_arith_context_encode_uegol (arith, context, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    carid_arith_context_binary_encode (arith, context, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_arith_context_encode_se2gol (CaridArith *arith, int context, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_arith_context_encode_ue2gol (arith, context, value);
  if (value) {
    carid_arith_context_binary_encode (arith, context, sign);
  }
}



int carid_arith_context_decode_bit (CaridArith *arith, int context)
{
  return carid_arith_context_binary_decode (arith, context);
}

int carid_arith_context_decode_bits (CaridArith *arith, int context, int n)
{
  int value = 0;
  int i;
  
  for(i=0;i<n;i++){
    value = (value << 1) | carid_arith_context_binary_decode (arith, context);
  } 
  
  return value;
}

int carid_arith_context_decode_uu (CaridArith *arith, int context, int context2)
{
  int value;

  if (carid_arith_context_binary_decode (arith, context)) return 0;
  if (carid_arith_context_binary_decode (arith, context2)) return 1;
  if (carid_arith_context_binary_decode (arith, context2 + 1)) return 2;
  if (carid_arith_context_binary_decode (arith, context2 + 2)) return 3;
  value = 4;
  while (carid_arith_context_binary_decode (arith, context2 + 3) == 0) {
    value++;
  }
  return value;
}

int carid_arith_context_decode_su (CaridArith *arith, int context)
{
  int value = 0;

  if (carid_arith_context_binary_decode (arith, context) == 1) {
    return 0;
  }
  value = 1;
  while (carid_arith_context_binary_decode (arith, context) == 0) {
    value++;
  }
  if (carid_arith_context_binary_decode (arith, context) == 0) {
    value = -value;
  }

  return value;
}

int carid_arith_context_decode_ut (CaridArith *arith, int context, int max)
{
  int value;

  for(value=0;value<max;value++){
    if (carid_arith_context_binary_decode (arith, context)) {
      return value;
    }
  }
  return value;
}

int carid_arith_context_decode_uegol (CaridArith *arith, int context)
{
  int count;
  int value;

  count = 0;
  while(!carid_arith_context_binary_decode (arith, context)) {
    count++;
  }
  value = (1<<count) - 1 + carid_arith_context_decode_bits (arith, context, count);

  return value;
}

int carid_arith_context_decode_segol (CaridArith *arith, int context)
{
  int value;

  value = carid_arith_context_decode_uegol (arith, context);
  if (value) {
    if (!carid_arith_context_binary_decode (arith, context)) {
      value = -value;
    }
  }

  return value;
}

int carid_arith_context_decode_ue2gol (CaridArith *arith, int context)
{
  int count;
  int value;

  count = carid_arith_context_decode_uegol (arith, context);
  value = (1<<count) - 1 + carid_arith_context_decode_bits (arith, context, count);

  return value;
}

int carid_arith_context_decode_se2gol (CaridArith *arith, int context)
{
  int value;

  value = carid_arith_context_decode_ue2gol (arith, context);
  if (value) {
    if (!carid_arith_context_binary_decode (arith, context)) {
      value = -value;
    }
  }

  return value;
}

