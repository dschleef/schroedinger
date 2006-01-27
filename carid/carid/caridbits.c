
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <liboil/liboil.h>

#include <carid/caridbits.h>
#include <carid/carid.h>


CaridBits *
carid_bits_new (void)
{
  CaridBits *bits;
  
  bits = malloc (sizeof(*bits));
  memset (bits, 0, sizeof(*bits));

  return bits;
}

void
carid_bits_free (CaridBits *bits)
{
  free(bits);
}

void
carid_bits_decode_init (CaridBits *bits, CaridBuffer *buffer)
{
  bits->buffer = buffer;
  bits->offset = 0;
}

void
carid_bits_encode_init (CaridBits *bits, CaridBuffer *buffer)
{
  uint8_t value = 0;

  bits->buffer = buffer;
  bits->offset = 0;

  /* FIXME this should be done incrementally */
  oil_splat_u8_ns (bits->buffer->data, &value, bits->buffer->length);
}

void
carid_bits_sync (CaridBits *bits)
{
  bits->offset = (bits->offset + 7) & (~0x7);
}

void
carid_bits_dumpbits (CaridBits *bits)
{
  char s[101];
  CaridBits mybits;
  int i;

  oil_memcpy (&mybits, bits, sizeof(*bits));

  for(i=0;i<100;i++){
    int bit = carid_bits_decode_bit (&mybits);
    s[i] = bit ? '1' : '0';
  }
  s[101] = 0;

  CARID_DEBUG ("dump bits %s", s);
}

void
carid_bits_append (CaridBits *bits, CaridBits *bits2)
{
  if (bits->offset & 7) {
    CARID_ERROR ("appending to unsyncronized bits");
  }
  if (bits2->offset & 7) {
    CARID_ERROR ("appending unsyncronized bits");
  }

  oil_memcpy (bits->buffer->data + (bits->offset>>3), bits2->buffer->data,
      (bits2->offset>>3));
  bits->offset += bits2->offset;
}


void
carid_bits_encode_bit (CaridBits *bits, int value)
{
  value &= 1;
  value <<= 7 - (bits->offset & 7);
  bits->buffer->data[(bits->offset>>3)] |= value;
  bits->offset++;
}

void
carid_bits_encode_bits (CaridBits *bits, int value, int n)
{
  int i;
  for(i=0;i<n;i++){
    carid_bits_encode_bit (bits, (value>>(n - 1 - i)) & 1);
  }
}

void
carid_bits_encode_uu (CaridBits *bits, int value)
{
  int i;

  for(i=0;i<value;i++){
    carid_bits_encode_bit (bits, 0);
  }
  carid_bits_encode_bit (bits, 1);
}

void
carid_bits_encode_su (CaridBits *bits, int value)
{
  int i;
  int sign;

  if (value==0) {
    carid_bits_encode_bit (bits, 1);
    return;
  }
  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  for(i=0;i<value;i++){
    carid_bits_encode_bit (bits, 0);
  }
  carid_bits_encode_bit (bits, 1);
  carid_bits_encode_bit (bits, sign);
}

void
carid_bits_encode_ut (CaridBits *bits, int value, int max)
{
  int i;

  for(i=0;i<value;i++){
    carid_bits_encode_bit (bits, 0);
  }
  if (value < max) {
    carid_bits_encode_bit (bits, 1);
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
carid_bits_encode_uegol (CaridBits *bits, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  for(i=0;i<n_bits - 1;i++){
    carid_bits_encode_bit (bits, 0);
  }
  carid_bits_encode_bit (bits, 1);
  for(i=0;i<n_bits - 1;i++){
    carid_bits_encode_bit (bits, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_bits_encode_segol (CaridBits *bits, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_bits_encode_uegol (bits, value);
  if (value) {
    carid_bits_encode_bit (bits, sign);
  }
}

void
carid_bits_encode_ue2gol (CaridBits *bits, int value)
{
  int i;
  int n_bits;

  value++;
  n_bits = maxbit(value);
  carid_bits_encode_uegol (bits, n_bits - 1);
  for(i=0;i<n_bits - 1;i++){
    carid_bits_encode_bit (bits, (value>>(n_bits - 2 - i))&1);
  }
}

void
carid_bits_encode_se2gol (CaridBits *bits, int value)
{
  int sign;

  if (value < 0) {
    sign = 0;
    value = -value;
  } else {
    sign = 1;
  }
  carid_bits_encode_ue2gol (bits, value);
  if (value) {
    carid_bits_encode_bit (bits, sign);
  }
}




int
carid_bits_decode_bit (CaridBits *bits)
{
  int value;
  value = bits->buffer->data[(bits->offset>>3)];
  value >>= 7 - (bits->offset & 7);
  value &= 1;
  bits->offset++;
  return value;
}

int
carid_bits_decode_bits (CaridBits *bits, int n)
{
  int value = 0;
  int i;

  for(i=0;i<n;i++){
    value = (value << 1) | carid_bits_decode_bit (bits);
  }

  return value;
}

int carid_bits_decode_uu (CaridBits *bits)
{
  int value = 0;

  while (carid_bits_decode_bit (bits) == 0) {
    value++;
  }

  return value;
}

int carid_bits_decode_su (CaridBits *bits)
{
  int value = 0;

  if (carid_bits_decode_bit (bits) == 1) {
    return 0;
  }
  value = 1;
  while (carid_bits_decode_bit (bits) == 0) {
    value++;
  }
  if (carid_bits_decode_bit (bits) == 0) {
    value = -value;
  }

  return value;
}

int carid_bits_decode_ut (CaridBits *bits, int max)
{
  int value;

  for(value=0;value<max;value++){
    if (carid_bits_decode_bit (bits)) {
      return value;
    }
  }
  return value;
}

int carid_bits_decode_uegol (CaridBits *bits)
{
  int count;
  int value;

  count = 0;
  while(!carid_bits_decode_bit (bits)) {
    count++;
  }
  value = (1<<count) - 1 + carid_bits_decode_bits (bits, count);

  return value;
}

int carid_bits_decode_segol (CaridBits *bits)
{
  int value;

  value = carid_bits_decode_uegol (bits);
  if (value) {
    if (!carid_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

int carid_bits_decode_ue2gol (CaridBits *bits)
{
  int count;
  int value;

  count = carid_bits_decode_uegol (bits);
  value = (1<<count) - 1 + carid_bits_decode_bits (bits, count);

  return value;
}

int carid_bits_decode_se2gol (CaridBits *bits)
{
  int value;

  value = carid_bits_decode_ue2gol (bits);
  if (value) {
    if (!carid_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

