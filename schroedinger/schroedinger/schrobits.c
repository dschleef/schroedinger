
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include <liboil/liboil.h>

#include <schroedinger/schrobits.h>
#include <schroedinger/schro.h>


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
schro_bits_copy (SchroBits *dest, SchroBits *src)
{
  memcpy (dest, src, sizeof(SchroBits));
}

static void
schro_bits_shift_in (SchroBits *bits)
{
  if (bits->n_bits >= 8) {
    bits->value = bits->buffer->data[bits->n];
    bits->n++;
    bits->shift = 7;
    bits->n_bits-=8;
    return;
  }
  if (bits->n_bits > 0) {
    bits->value = bits->buffer->data[bits->n] >> (8-bits->n_bits);
    bits->n++;
    bits->shift = 8 - bits->n_bits;
    bits->n_bits = 0;
    return;
  }
  bits->value = 0xff;
  bits->shift = 7;
  bits->error = TRUE;
}

static void
schro_bits_shift_out (SchroBits *bits)
{
  if (bits->n < bits->buffer->length) {
    bits->buffer->data[bits->n] = bits->value;
    bits->n++;
    bits->shift = 7;
    bits->value = 0;
    return;
  }
  if (bits->error == FALSE) {
    SCHRO_ERROR("buffer overrun");
  }
  bits->error = TRUE;
  bits->shift = 7;
  bits->value = 0;
}

void
schro_bits_set_length (SchroBits *bits, int n_bits)
{
  bits->n_bits = n_bits;
}

void
schro_bits_decode_init (SchroBits *bits, SchroBuffer *buffer)
{
  bits->buffer = buffer;
  bits->n = 0;
  bits->shift = -1;
  bits->type = SCHRO_BITS_DECODE;
  bits->n_bits = buffer->length * 8;
}

void
schro_bits_encode_init (SchroBits *bits, SchroBuffer *buffer)
{
  bits->buffer = buffer;
  bits->n = 0;
  bits->type = SCHRO_BITS_ENCODE;

  bits->value = 0;
  bits->shift = 7;
}

int
schro_bits_get_offset (SchroBits *bits)
{
  return bits->n;
}

void
schro_bits_flush (SchroBits *bits)
{
  schro_bits_sync (bits);
}

void
schro_bits_sync (SchroBits *bits)
{
  if (bits->type == SCHRO_BITS_DECODE) {
    bits->shift = -1;
  } else {
    if (bits->shift != 7) {
      schro_bits_shift_out (bits);
    }
  }
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
  if (bits->shift != 7) {
    SCHRO_ERROR ("appending to unsyncronized bits");
  }

  SCHRO_ASSERT(bits->n + len <= bits->buffer->length);

  oil_memcpy (bits->buffer->data + bits->n, data, len);
  bits->n += len;
}

void
schro_bits_skip (SchroBits *bits, int n_bytes)
{
  if (bits->shift != -1) {
    SCHRO_ERROR ("skipping on unsyncronized bits");
  }

  bits->n += n_bytes;
}

void
schro_bits_skip_bits (SchroBits *bits, int n_bits)
{
  if (bits->shift >= 0) {
    if (n_bits <= bits->shift + 1) {
      bits->shift -= n_bits;
      return;
    }
    n_bits -= bits->shift + 1;
    bits->shift = -1;
  }
  if (n_bits >= 8) {
    bits->n += (n_bits>>3);
    bits->n_bits -= (n_bits & ~7);
    if (bits->n_bits < 0) {
      bits->error = TRUE;
    }
  }
  schro_bits_shift_in (bits);

  bits->shift -= n_bits;
}


void
schro_bits_encode_bit (SchroBits *bits, int value)
{
  value &= 1;
  bits->value |= (value << bits->shift);
  bits->shift--;
  if (bits->shift < 0) {
    schro_bits_shift_out (bits);
  }
}

void
schro_bits_encode_bits (SchroBits *bits, int n, unsigned int value)
{
  int i;
  for(i=0;i<n;i++){
    schro_bits_encode_bit (bits, (value>>(n - 1 - i)) & 1);
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
    sign = 1;
    value = -value;
  } else {
    sign = 0;
  }
  schro_bits_encode_uint (bits, value);
  if (value) {
    schro_bits_encode_bit (bits, sign);
  }
}

int
schro_bits_decode_bit (SchroBits *bits)
{
  int value;

  if (bits->shift < 0) {
    schro_bits_shift_in (bits);
  }
  value = (bits->value >> bits->shift) & 1;
  bits->shift--;
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
    if (schro_bits_decode_bit (bits)) {
      value = -value;
    }
  }

  return value;
}

int
schro_bits_estimate_uint (int value)
{
  int n_bits;

  value++;
  n_bits = maxbit(value);
  return n_bits + n_bits - 1;
}

int
schro_bits_estimate_sint (int value)
{
  int n_bits;

  if (value < 0) {
    value = -value;
  }
  n_bits = schro_bits_estimate_uint (value);
  if (value) n_bits++;
  return n_bits;
}

