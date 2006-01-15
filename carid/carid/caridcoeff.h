
#ifndef _CARID_COEFF_H_
#define _CARID_COEFF_H_

#include <carid/cariddecoder.h>
#include <carid/caridencoder.h>


void carid_coeff_decode_transform_parameters (CaridDecoder *decoder);
void carid_coeff_decode_transform_data (CaridDecoder *decoder);
void carid_coeff_decode_subband (CaridDecoder *decoder, int x, int y, int w, int h);

void carid_coeff_encode_transform_parameters (CaridEncoder *encoder);
void carid_coeff_encode_transform_data (CaridEncoder *encoder);
void carid_coeff_encode_subband (CaridEncoder *encoder, int x, int y, int w, int h);


#endif


