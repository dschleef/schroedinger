
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>

#ifndef USE_NEW_CODER
static void _schro_arith_output_bit (SchroArith *arith);
#endif
static int __schro_arith_context_decode_bit (SchroArith *arith, int i);

SchroArith *
schro_arith_new (void)
{
  SchroArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

#ifndef USE_NEW_CODER
  memcpy (arith->division_factor, schro_table_division_factor,
      sizeof(arith->division_factor));
  memcpy (arith->fixup_shift, schro_table_arith_shift,
      sizeof(arith->fixup_shift));
#endif

  (void)&__schro_arith_context_decode_bit;

  return arith;
}

void
schro_arith_free (SchroArith *arith)
{
  free(arith);
}

#ifndef USE_NEW_CODER
static void
schro_arith_reload_nextcode (SchroArith *arith)
{
  while(arith->nextbits <= 24) {
    if (arith->dataptr < arith->maxdataptr) {
      arith->nextcode |= arith->dataptr[0] << (24-arith->nextbits);
    } else {
      arith->nextcode |= 0xff << (24-arith->nextbits);
    }
    arith->nextbits+=8;
    arith->dataptr++;
    arith->offset++;
  }
}
#endif

void
schro_arith_decode_init (SchroArith *arith, SchroBuffer *buffer)
{
#ifdef USE_NEW_CODER
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
#else
  arith->range[0] = 0;
  arith->range[1] = 0xffff;
  arith->code = 0;

  arith->buffer = buffer;

  arith->dataptr = arith->buffer->data;
  arith->maxdataptr = arith->buffer->data + arith->buffer->length;
  arith->code = arith->dataptr[0] << 8;
  arith->code |= arith->dataptr[1];
  arith->dataptr+=2;
  arith->offset = 2;
  schro_arith_reload_nextcode(arith);
#endif
}

void
schro_arith_encode_init (SchroArith *arith, SchroBuffer *buffer)
{
#ifdef USE_NEW_CODER
  memset(arith, 0, sizeof(SchroArith));
  arith->range[0] = 0;
  arith->range[1] = 0x10000;
  arith->code = 0;

  arith->buffer = buffer;
  arith->offset = 0;
  arith->dataptr = arith->buffer->data;
  //arith->nextbits = 0;
  //arith->nextcode = 0;
#else
  arith->range[0] = 0;
  arith->range[1] = 0xffff;
  arith->code = 0;

  arith->buffer = buffer;
  arith->offset = 0;
  arith->nextbits = 0;
  arith->nextcode = 0;
#endif
}

#ifndef USE_NEW_CODER
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
#endif

int hist[11];

void
schro_arith_flush (SchroArith *arith)
{
#ifdef USE_NEW_CODER
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
#else
  int i;

#define ENABLE_ARITH_REAPING
#ifdef ENABLE_ARITH_REAPING
  {
    int n;
    for(n=0;n<16;n++){
      if ((arith->range[0] | ((1<<(n+1))-1)) > arith->range[1]) {
        break;
      }
    }
    arith->range[0] |= (1<<n)-1;
  }
#endif

  for(i=0;i<16;i++){
    _schro_arith_output_bit (arith);
    arith->range[0] <<= 1;
  }
  while(arith->nextbits < 8) {
    arith->nextcode <<= 1;
    arith->nextcode |= 1;
    arith->nextbits++;
  }
  arith->buffer->data[arith->offset] = arith->nextcode;
  arith->offset++;

#ifdef ENABLE_ARITH_REAPING
#if 1
  if (arith->offset > 1 && arith->buffer->data[arith->offset - 1] == 0xff) {
    arith->offset--;
  }
  if (arith->offset > 1 && arith->buffer->data[arith->offset - 1] == 0xff) {
    arith->offset--;
  }
  if (arith->offset > 1 && arith->buffer->data[arith->offset - 1] == 0xff) {
    arith->offset--;
  }
#else
  while (arith->offset > 1 && arith->buffer->data[arith->offset - 1] == 0xff) {
    arith->offset--;
  }
#endif
#endif

#if 0
  for(i=0;i<SCHRO_CTX_LAST;i++) {
    int a,b;
    int x;
    a = arith->contexts[i].count[0];
    b = arith->contexts[i].count[1];
    if (a < b) {
      x = (a * 10)/b;
    } else {
      x = (b * 10)/a;
    }
    if (x > 9) x = 9;
#if 1
    SCHRO_INFO("%d %d %d %d %d", i,
        arith->contexts[i].used,
        arith->contexts[i].count[0], arith->contexts[i].count[1], x);
#endif
    hist[x] += arith->contexts[i].used;
  }
#endif
#endif
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

#ifdef USE_NEW_CODER
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
#else
#ifdef __i386_lKJSDLKfjsf__
#include "schroarith-i386.c"
#else
static int
__schro_arith_context_decode_bit (SchroArith *arith, int i)
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
#endif


void
_schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  //unsigned int count;
  unsigned int range;
  unsigned int scaler;
  unsigned int weight;
  unsigned int probability0;
  unsigned int range_x_prob;

  weight = arith->contexts[i].count[0] + arith->contexts[i].count[1];
  scaler = arith->division_factor[weight];
  probability0 = arith->contexts[i].count[0] * scaler;
  //count = arith->code - arith->range[0] + 1;
  range = arith->range[1] - arith->range[0] + 1;
  range_x_prob = (range * probability0) >> 16;
//SCHRO_ERROR("scaler %d count0 %d prob0 %d range_x_prob %d", scaler, arith->contexts[i].count[0], probability0, range_x_prob);
  //value = (count > range_x_prob);

  arith->range[1 - value] = arith->range[0] + range_x_prob - 1 + value;
  arith->contexts[i].count[value]++;
  //arith->contexts[i].used++;
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
#endif

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

