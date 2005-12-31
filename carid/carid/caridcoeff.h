
#ifndef _CARID_BITS_H_
#define _CARID_BITS_H_

#include <carid/caridbits.h>

typedef struct _CaridDecoder CaridDecoder;

struct _CaridDecoder {
  CaridBits *bits;
};

void carid_coeff_decode_transform_parameters (CaridDecoder *decoder);
void carid_coeff_decode_transform_data (CaridDecoder *decoder);



#endif


