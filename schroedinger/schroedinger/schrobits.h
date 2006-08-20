
#ifndef _SCHRO_BITS_H_
#define _SCHRO_BITS_H_

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schrobuffer.h>


#define SCHRO_ARITH_N_CONTEXTS 64

typedef struct _SchroBits SchroBits;

struct _SchroBits {
  SchroBuffer *buffer;

  int offset;
  int bit_offset;

  int cntr;

  int n_contexts;
};

SchroBits * schro_bits_new (void);
void schro_bits_free (SchroBits *bits);

void schro_bits_decode_init (SchroBits *bits, SchroBuffer *buffer);
void schro_bits_encode_init (SchroBits *bits, SchroBuffer *buffer);

void schro_bits_sync (SchroBits *bits);
void schro_bits_dumpbits (SchroBits *bits);

void schro_bits_append (SchroBits *bits, uint8_t *data, int len);

void schro_bits_encode_bit (SchroBits *bits, int value);
void schro_bits_encode_bits (SchroBits *bits, int n, unsigned int value);
void schro_bits_encode_uu (SchroBits *bits, int value);
void schro_bits_encode_su (SchroBits *bits, int value);
void schro_bits_encode_ut (SchroBits *bits, int value, int max);
void schro_bits_encode_uegol (SchroBits *bits, int value);
void schro_bits_encode_segol (SchroBits *bits, int value);
void schro_bits_encode_uint (SchroBits *bits, int value);
void schro_bits_encode_sint (SchroBits *bits, int value);
void schro_bits_encode_ue2gol (SchroBits *bits, int value);
void schro_bits_encode_se2gol (SchroBits *bits, int value);

int schro_bits_decode_bit (SchroBits *bits);
int schro_bits_decode_bits (SchroBits *bits, int n);
int schro_bits_decode_uu (SchroBits *bits);
int schro_bits_decode_su (SchroBits *bits);
int schro_bits_decode_ut (SchroBits *bits, int max);
int schro_bits_decode_uegol (SchroBits *bits);
int schro_bits_decode_segol (SchroBits *bits);
int schro_bits_decode_uint (SchroBits *bits);
int schro_bits_decode_sint (SchroBits *bits);
int schro_bits_decode_ue2gol (SchroBits *bits);
int schro_bits_decode_se2gol (SchroBits *bits);

#endif


