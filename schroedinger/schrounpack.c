
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>
#include <schroedinger/schrotables.h>

#include <string.h>

static void _schro_unpack_shift_in (SchroUnpack *unpack);
static unsigned int _schro_unpack_shift_out (SchroUnpack *unpack, int n);

void
schro_unpack_init_with_data (SchroUnpack *unpack, uint8_t *data,
    int n_bytes, unsigned int guard_bit)
{
  memset (unpack, 0, sizeof(SchroUnpack));

  unpack->data = data;
  unpack->n_bits_left = 8*n_bytes;
  unpack->guard_bit = guard_bit;
}

void
schro_unpack_copy (SchroUnpack *dest, SchroUnpack *src)
{
  memcpy (dest, src, sizeof(SchroUnpack));
}

int
schro_unpack_get_bits_read (SchroUnpack *unpack)
{
  return unpack->n_bits_read;
}

#ifdef unused
int
schro_unpack_get_bits_remaining (SchroUnpack *unpack)
{
  if (unpack->overrun) {
    return 0;
  }
  return unpack->n_bits_left + unpack->n_bits_in_shift_register;
}
#endif

void
schro_unpack_limit_bits_remaining (SchroUnpack *unpack, int n_bits)
{
  if (n_bits <= unpack->n_bits_in_shift_register) {
    unpack->n_bits_in_shift_register = n_bits;
    unpack->shift_register &= ~(0xffffffff >> n_bits);
    unpack->n_bits_left = 0;
    return;
  }

  unpack->n_bits_left = n_bits - unpack->n_bits_in_shift_register;
}

#if 0
void
schro_unpack_dumpbits (SchroUnpack *unpack)
{

}
#endif

static void
_schro_unpack_shift_in (SchroUnpack *unpack)
{
  if (unpack->n_bits_left >= 32) {
    /* the fast path */
    if (unpack->n_bits_in_shift_register == 0) {
      unpack->shift_register = (unpack->data[0]<<24) | (unpack->data[1]<<16) |
        (unpack->data[2]<<8) | (unpack->data[3]);
      unpack->data += 4;
      unpack->n_bits_left -= 32;
      unpack->n_bits_in_shift_register = 32;
    } else {
      while (unpack->n_bits_in_shift_register <= 24) {
        unpack->shift_register |=
          unpack->data[0]<<(24 - unpack->n_bits_in_shift_register);
        unpack->data++;
        unpack->n_bits_left -= 8;
        unpack->n_bits_in_shift_register += 8;
      }
    }
    return;
  }

  if (unpack->n_bits_left == 0) {
    unsigned int value = (unpack->guard_bit)?0xffffffff:0;

    unpack->overrun += 32 - unpack->n_bits_in_shift_register;
    unpack->shift_register |= (value >> unpack->n_bits_in_shift_register);
    unpack->n_bits_in_shift_register = 32;
    return;
  }

  while (unpack->n_bits_left >= 8 && unpack->n_bits_in_shift_register <= 24) {
    unpack->shift_register |=
      unpack->data[0]<<(24 - unpack->n_bits_in_shift_register);
    unpack->data++;
    unpack->n_bits_left -= 8;
    unpack->n_bits_in_shift_register += 8;
  }

  if (unpack->n_bits_left > 0 &&
      unpack->n_bits_in_shift_register + unpack->n_bits_left <= 32) {
    unsigned int value;

    value = unpack->data[0]>>(8-unpack->n_bits_left);
    unpack->shift_register |=
      value<<(32 - unpack->n_bits_in_shift_register - unpack->n_bits_left);
    unpack->data++;
    unpack->n_bits_in_shift_register += unpack->n_bits_left;
    unpack->n_bits_left = 0;
  }
}

static unsigned int
_schro_unpack_shift_out (SchroUnpack *unpack, int n)
{
  unsigned int value;

  if (n == 0) return 0;

  value = unpack->shift_register >> (32 - n);
  unpack->shift_register <<= n;
  unpack->n_bits_in_shift_register -= n;
  unpack->n_bits_read += n;

  return value;
}


void
schro_unpack_skip_bits (SchroUnpack *unpack, int n_bits)
{
  int n_bytes;

  if (n_bits <= unpack->n_bits_in_shift_register) {
    _schro_unpack_shift_out (unpack, n_bits);
    return;
  }

  n_bits -= unpack->n_bits_in_shift_register;
  _schro_unpack_shift_out (unpack, unpack->n_bits_in_shift_register);

  n_bytes = MIN(n_bits>>3, unpack->n_bits_left>>3);
  unpack->data += n_bytes;
  unpack->n_bits_read += n_bytes*8;
  unpack->n_bits_left -= n_bytes*8;
  n_bits -= n_bytes*8;

  if (n_bits == 0) return;

  _schro_unpack_shift_in (unpack);

  if (n_bits <= unpack->n_bits_in_shift_register) {
    _schro_unpack_shift_out (unpack, n_bits);
    return;
  }

  unpack->n_bits_in_shift_register = 0;
  unpack->shift_register = 0;
  unpack->overrun += n_bits;
  unpack->n_bits_read += n_bits;
}

void
schro_unpack_byte_sync (SchroUnpack *unpack)
{
  if (unpack->n_bits_read & 7) {
    schro_unpack_skip_bits (unpack, 8 - (unpack->n_bits_read & 7));
  }
}

unsigned int
schro_unpack_decode_bit (SchroUnpack *unpack)
{
  if (unpack->n_bits_in_shift_register < 1) {
    _schro_unpack_shift_in (unpack);
  }

  return _schro_unpack_shift_out (unpack, 1);
}

unsigned int
schro_unpack_decode_bits (SchroUnpack *unpack, int n)
{
  unsigned int value;
  int m;

  m = MIN(n,unpack->n_bits_in_shift_register);
  value = _schro_unpack_shift_out (unpack, m) << (n-m);
  n -= m;

  while(n>0) {
    _schro_unpack_shift_in (unpack);
    m = MIN(n,unpack->n_bits_in_shift_register);
    value |= _schro_unpack_shift_out (unpack, m) << (n-m);
    n -= m;
  }

  return value;
}

unsigned int
schro_unpack_decode_uint (SchroUnpack *unpack)
{
  int count;
  int value;

  count = 0;
  value = 0;
  while(!schro_unpack_decode_bit (unpack)) {
    count++;
    value <<= 1;
    value |= schro_unpack_decode_bit (unpack);
  }

  return (1<<count) - 1 + value;
}

int
schro_unpack_decode_sint_slow (SchroUnpack *unpack)
{
  int value;

  value = schro_unpack_decode_uint (unpack);
  if (value) {
    if (schro_unpack_decode_bit (unpack)) {
      value = -value;
    }
  }

  return value;
}

#define SHIFT 8

int
schro_unpack_decode_sint (SchroUnpack *unpack)
{
  int value;
  int i;

  if (unpack->n_bits_in_shift_register < SHIFT) {
    _schro_unpack_shift_in (unpack);
  }
  if (unpack->n_bits_in_shift_register >= SHIFT) {
    i = unpack->shift_register >> (32-SHIFT);
    if (schro_table_unpack_sint[i][0] > 0) {
      value = schro_table_unpack_sint[i][1];
      _schro_unpack_shift_out (unpack, schro_table_unpack_sint[i][2]);
      return value;
    }
  }

  return schro_unpack_decode_sint_slow (unpack);
}

void
schro_unpack_decode_sint_s16 (int16_t *dest, SchroUnpack *unpack, int n)
{
  int i;
  int m;
  int j;

  while (n > 0) {
    while (unpack->n_bits_in_shift_register < SHIFT) {
      _schro_unpack_shift_in (unpack);
    }
    i = unpack->shift_register >> (32-SHIFT);
    if (schro_table_unpack_sint[i][0] == 0) {
      dest[0] = schro_unpack_decode_sint_slow (unpack);
      dest++;
      n--;
    } else {
      m = MIN(n, schro_table_unpack_sint[i][0]);
      for(j=0;j<m;j++){
        dest[j] = schro_table_unpack_sint[i][1+2*j];
      }
      _schro_unpack_shift_out (unpack, schro_table_unpack_sint[i][2*m]);
      dest += m;
      n -= m;
    }
  }
}

