
#ifndef _CARID_BITS_H_
#define _CARID_BITS_H_

#include <carid/carid-stdint.h>
#include <carid/caridbuffer.h>


#define CARID_ARITH_N_CONTEXTS 64

typedef struct _CaridBits CaridBits;

struct _CaridBits {
  CaridBuffer *buffer;

  int offset;
  int bit_offset;

  int cntr;

  int n_contexts;
};

CaridBits * carid_bits_new (void);
void carid_bits_free (CaridBits *bits);

void carid_bits_decode_init (CaridBits *bits, CaridBuffer *buffer);
void carid_bits_encode_init (CaridBits *bits, CaridBuffer *buffer);

void carid_bits_sync (CaridBits *bits);
void carid_bits_dumpbits (CaridBits *bits);

void carid_bits_encode_bit (CaridBits *bits, int value);
void carid_bits_encode_bits (CaridBits *bits, int value, int n);
void carid_bits_encode_uu (CaridBits *bits, int value);
void carid_bits_encode_su (CaridBits *bits, int value);
void carid_bits_encode_ut (CaridBits *bits, int value, int max);
void carid_bits_encode_uegol (CaridBits *bits, int value);
void carid_bits_encode_segol (CaridBits *bits, int value);
void carid_bits_encode_ue2gol (CaridBits *bits, int value);
void carid_bits_encode_se2gol (CaridBits *bits, int value);

int carid_bits_decode_bit (CaridBits *bits);
int carid_bits_decode_bits (CaridBits *bits, int n);
int carid_bits_decode_uu (CaridBits *bits);
int carid_bits_decode_su (CaridBits *bits);
int carid_bits_decode_ut (CaridBits *bits, int max);
int carid_bits_decode_uegol (CaridBits *bits);
int carid_bits_decode_segol (CaridBits *bits);
int carid_bits_decode_ue2gol (CaridBits *bits);
int carid_bits_decode_se2gol (CaridBits *bits);

#endif


