
#ifndef _SCHRO_BITS_H_
#define _SCHRO_BITS_H_

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schrobuffer.h>

SCHRO_BEGIN_DECLS

#define SCHRO_ARITH_N_CONTEXTS 64

typedef struct _SchroBits SchroBits;

#define SCHRO_BITS_DECODE 0
#define SCHRO_BITS_ENCODE 1

struct _SchroBits {
  int type;
  SchroBuffer *buffer;

  int n;
  int shift;

  uint32_t value;

  int error;
};

SchroBits * schro_bits_new (void);
void schro_bits_free (SchroBits *bits);

void schro_bits_decode_init (SchroBits *bits, SchroBuffer *buffer);
void schro_bits_encode_init (SchroBits *bits, SchroBuffer *buffer);

void schro_bits_sync (SchroBits *bits);
void schro_bits_flush (SchroBits *bits);
int schro_bits_get_offset (SchroBits *bits);
void schro_bits_dumpbits (SchroBits *bits);

void schro_bits_append (SchroBits *bits, uint8_t *data, int len);
void schro_bits_skip (SchroBits *bits, int n_bytes);

void schro_bits_encode_bit (SchroBits *bits, int value);
void schro_bits_encode_bits (SchroBits *bits, int n, unsigned int value);
void schro_bits_encode_uint (SchroBits *bits, int value);
void schro_bits_encode_sint (SchroBits *bits, int value);

int schro_bits_decode_bit (SchroBits *bits);
int schro_bits_decode_bits (SchroBits *bits, int n);
int schro_bits_decode_uint (SchroBits *bits);
int schro_bits_decode_sint (SchroBits *bits);

SCHRO_END_DECLS

#endif


