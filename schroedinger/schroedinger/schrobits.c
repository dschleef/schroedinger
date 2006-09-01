
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <liboil/liboil.h>

#include <schroedinger/schrobits.h>
#include <schroedinger/schrointernal.h>


SchroBits *
schro_bits_new (void)
{
  SchroBits *bits;
  
  bits = malloc (sizeof(*bits));
  memset (bits, 0, sizeof(*bits));

  return bits;
}

void
schro_bits_free (SchroBits *bits)
{
  free(bits);
}

void
schro_bits_decode_init (SchroBits *bits, SchroBuffer *buffer)
{
  bits->buffer = buffer;
  bits->offset = 0;
}

void
schro_bits_encode_init (SchroBits *bits, SchroBuffer *buffer)
{
  uint8_t value = 0;

  bits->buffer = buffer;
  bits->offset = 0;

  /* FIXME this should be done incrementally */
  oil_splat_u8_ns (bits->buffer->data, &value, bits->buffer->length);
}

void
schro_bits_sync (SchroBits *bits)
{
  bits->offset = (bits->offset + 7) & (~0x7);
}

void
schro_bits_dumpbits (SchroBits *bits)
{
  char s[101];
  SchroBits mybits;
  int i;

  oil_memcpy (&mybits, bits, sizeof(*bits));

  for(i=0;i<100;i++){
    int bit = schro_bits_decode_bit (&mybits);
    s[i] = bit ? '1' : '0';
  }
  s[100] = 0;

  SCHRO_DEBUG ("dump bits %s", s);
}

void
schro_bits_append (SchroBits *bits, uint8_t *data, int len)
{
  if (bits->offset & 7) {
    SCHRO_ERROR ("appending to unsyncronized bits");
  }

  oil_memcpy (bits->buffer->data + (bits->offset>>3), data, len);
  bits->offset += len*8;
}


void
schro_bits_encode_bit (SchroBits *bits, int value)
{
  value &= 1;
  value <<= 7 - (bits->offset & 7);
  bits->buffer->data[(bits->offset>>3)] |= value;
  bits->offset++;
}

void
schro_bits_encode_bits (SchroBits *bits, int n, unsigned int value)
{
  int i;
  for(i=0;i<n;i++){
    schro_bits_encode_bit (bits, (value>>(n - 1 - i)) & 1);
  }
}

void
schro_bits_encode_uu (SchroBits *bits, int value)
{
  int i;

  for(i=0;i<value;i++){
    schro_bits_encode_bit (bits, 0);
  }
  schro_bits_encode_bit (bits, 1);
}

void
schro_bits_encode_su (SchroBits *bits, int value)
{
  int i;
  int sign;

  if (value==0) {
    schro_bits_encode_bit (bits, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    schro_bits_encode_bit (bits, 0);
  }
  schro_bits_encode_bit (bits, 1);
  schro_bits_encode_bit (bits, sign);
}

void
schro_bits_encode_ut (SchroBits *bits, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    schro_bits_encode_bit (bits, 0);
  }
  if (value < max) {
    schro_bits_encode_bit (bits, 1);
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
schro_bits_encode_uint (SchroBits *bits, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    schro_bits_encode_bit (bits, 0);
    schro_bits_encode_bit (bits, (value>>(n_bits - 2 - i))&1);
  }
  schro_bits_encode_bit (bits, 1);
}

void
schro_bits_encode_sint (SchroBits *bits, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_bits_encode_uint (bits, value);
  if (value) {
    schro_bits_encode_bit (bits, sign);
  }
}

void
schro_bits_encode_uegol (SchroBits *bits, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    schro_bits_encode_bit (bits, 0);
  }
  schro_bits_encode_bit (bits, 1);
  for(i=0;i<n_bits - 1;i++){
    schro_bits_encode_bit (bits, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_bits_encode_segol (SchroBits *bits, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_bits_encode_uegol (bits, value);
  if (value) {
    schro_bits_encode_bit (bits, sign);
  }
}

void
schro_bits_encode_ue2gol (SchroBits *bits, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  schro_bits_encode_uegol (bits, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    schro_bits_encode_bit (bits, (value>>(n_bits - 2 - i))&1);
  }
}

void
schro_bits_encode_se2gol (SchroBits *bits, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  schro_bits_encode_ue2gol (bits, value);
  if (value) {
    schro_bits_encode_bit (bits, sign);
  }
}




int
schro_bits_decode_bit (SchroBits *bits)
{
  int value;
  value = bits->buffer->data[(bits->offset>>3)];
  value >>= 7 - (bits->offset & 7);
  value &= 1;
  bits->offset++;
  return value;
}

int
schro_bits_decode_bits (SchroBits *bits, int n)
{
  int value = 0;
  int i;

  for(i=0;i<n;i++){
    value = (value << 1) | schro_bits_decode_bit (bits);
  }

  return value;
}

int schro_bits_decode_uu (SchroBits *bits)
{
  int value = 0;

  while (schro_bits_decode_bit (bits) == 0) {
    value++;
  }

  return value;
}

int schro_bits_decode_su (SchroBits *bits)
{
  int value = 0;

  if (schro_bits_decode_bit (bits) == 1) {
    return 0;
  }
  value = 1;
  while (schro_bits_decode_bit (bits) == 0) {
    value++;
  }
  if (schro_bits_decode_bit (bits) == 0) {
    value = -value;
  }

  return value;
}

int schro_bits_decode_ut (SchroBits *bits, int max)
{
  int value;

  for(value=0;value<max;value++){
    if (schro_bits_decode_bit (bits)) {
      return value;
    }
  }
  return value;
}

int schro_bits_decode_uegol (SchroBits *bits)
{
  int count;
  int value;

  count = 0;
  while(!schro_bits_decode_bit (bits)) {
    count++;
  }
  value = (1<<count) - 1 + schro_bits_decode_bits (bits, count);

  return value;
}

int schro_bits_decode_segol (SchroBits *bits)
{
  int value;

  value = schro_bits_decode_uegol (bits);
  if (value) {
    if (!schro_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

int schro_bits_decode_uint (SchroBits *bits)
{
  int count;
  int value;

  count = 0;
  value = 0;
  while(!schro_bits_decode_bit (bits)) {
    count++;
    value <<= 1;
    value |= schro_bits_decode_bit (bits);
  }

  return (1<<count) - 1 + value;
}

int schro_bits_decode_sint (SchroBits *bits)
{
  int value;

  value = schro_bits_decode_uint (bits);
  if (value) {
    if (!schro_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

int schro_bits_decode_ue2gol (SchroBits *bits)
{
  int count;
  int value;

  count = schro_bits_decode_uegol (bits);
  value = (1<<count) - 1 + schro_bits_decode_bits (bits, count);

  return value;
}

int schro_bits_decode_se2gol (SchroBits *bits)
{
  int value;

  value = schro_bits_decode_ue2gol (bits);
  if (value) {
    if (!schro_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

