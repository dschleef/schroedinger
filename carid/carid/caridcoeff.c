
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <carid/carid.h>


void
carid_coeff_decode_transform_parameters (CaridDecoder *decoder)
{
  int bit;
  CaridParams *params = &decoder->params;

  /* transform */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    params->wavelet_filter_index = carid_bits_decode_uegol (decoder->bits);
    if (params->wavelet_filter_index > 4) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->wavelet_filter_index = CARID_WAVELET_DAUB97;
  }
  CARID_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    params->transform_depth = carid_bits_decode_uegol (decoder->bits);
    if (params->transform_depth > 6) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->transform_depth = 4;
  }
  CARID_DEBUG ("transform depth %d", params->transform_depth);

  /* spatial partitioning */
  params->spatial_partition = carid_bits_decode_bit (decoder->bits);
  CARID_DEBUG ("spatial_partitioning %d", params->spatial_partition);
  if (params->spatial_partition) {
    params->partition_index = carid_bits_decode_uegol (decoder->bits);
    if (params->partition_index > 1) {
      /* FIXME: ? */
      params->non_spec_input = TRUE;
    }
    if (params->partition_index == 0) {
      params->max_xblocks = carid_bits_decode_uegol (decoder->bits);
      params->max_yblocks = carid_bits_decode_uegol (decoder->bits);
    }
    params->multi_quant = carid_bits_decode_bit (decoder->bits);
  }

  carid_params_calculate_iwt_sizes (params);

  carid_bits_sync(decoder->bits);
}

void
carid_coeff_decode_transform_data (CaridDecoder *decoder)
{
  int i;
  int w,h;
  CaridParams *params = &decoder->params;

  carid_bits_dumpbits (decoder->bits);

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  carid_coeff_decode_subband (decoder, 0, 0, w, h);
  for(i=0; i < params->transform_depth; i++) {
    carid_coeff_decode_subband (decoder, w, h, w, h);
    carid_coeff_decode_subband (decoder, 0, h, w, h);
    carid_coeff_decode_subband (decoder, w, 0, w, h);
    w <<= 1;
    h <<= 1;
  }
}

void
carid_coeff_decode_subband (CaridDecoder *decoder, int x, int y, int w, int h)
{
  CaridParams *params = &decoder->params;
  int subband_zero_flag;
  int stride;
  int16_t *data;
  int quant_index = 0;
  int quant_factor = 0;
  int quant_offset = 0;
  int subband_length;

  CARID_DEBUG("subband %d x %d at %d, %d", w, h, x, y);

  stride = params->iwt_chroma_width;
  data = (int16_t *)decoder->frame_buffer->data;
  subband_zero_flag = carid_bits_decode_bit (decoder->bits);
  if (!subband_zero_flag) {
    quant_index = carid_bits_decode_uegol (decoder->bits);
    CARID_DEBUG("quant index %d", quant_index);
    if ((unsigned int)quant_index > 60) {
      CARID_ERROR("quant_index too big (%u > 60)", quant_index);
      params->non_spec_input = TRUE;
      return;
    }
if (quant_index != 10) {
  CARID_ERROR("quant_index wrong");
  return;
}
    quant_factor = carid_table_quant[quant_index];
    quant_offset = carid_table_offset[quant_index];
    CARID_DEBUG("quant factor %d offset %d", quant_factor, quant_offset);

    subband_length = carid_bits_decode_uegol (decoder->bits);
    CARID_DEBUG("subband length %d", subband_length);
  }
  carid_bits_sync (decoder->bits);

  if (subband_zero_flag) {
    int i,j;
    CARID_DEBUG("subband is zero");
    for(j=y;j<y+h;j++){
      for(i=x;i<x+w;i++){
        data[j*stride + i] = 0;
      }
    }
  } else {
    int i,j;

    for(j=y;j<y+h;j++){
      for(i=x;i<x+w;i++){
        int sign = 0;
        int v;
        v = carid_bits_decode_uegol (decoder->bits);
//        CARID_DEBUG("decoded %d", v);
        if (v) {
          sign = carid_bits_decode_bit (decoder->bits);
//          CARID_DEBUG("decoded sign %d", sign);
        }
        v *= quant_factor;
        if (v > 0) {
          v += quant_offset;
        }
        if (v > 0) {
          if (!sign) {
            v = -v;
          }
        }
        data[j*stride + i] = v;
      }
    }
    carid_bits_sync (decoder->bits);
  }
}



void
carid_coeff_encode_transform_parameters (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;

  /* transform */
  if (params->wavelet_filter_index == CARID_WAVELET_DAUB97) {
    carid_bits_encode_bit (encoder->bits, 0);
  } else {
    carid_bits_encode_bit (encoder->bits, 1);
    carid_bits_encode_uegol (encoder->bits, params->wavelet_filter_index);
  }

  /* transform depth */
  if (params->transform_depth == 4) {
    carid_bits_encode_bit (encoder->bits, 0);
  } else {
    carid_bits_encode_bit (encoder->bits, 1);
    carid_bits_encode_uegol (encoder->bits, params->transform_depth);
  }

  /* spatial partitioning */
  carid_bits_encode_bit (encoder->bits, 0);

  carid_bits_sync(encoder->bits);
}

void
carid_coeff_encode_transform_data (CaridEncoder *encoder)
{
  int i;
  int w,h;
  CaridParams *params = &encoder->params;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  carid_coeff_encode_subband (encoder, 0, 0, w, h);
  for(i=0; i < params->transform_depth; i++) {
    carid_coeff_encode_subband (encoder, w, h, w, h);
    carid_coeff_encode_subband (encoder, 0, h, w, h);
    carid_coeff_encode_subband (encoder, w, 0, w, h);
    w <<= 1;
    h <<= 1;
  }
}

void
carid_coeff_encode_subband (CaridEncoder *encoder, int x, int y, int w, int h)
{
  CaridParams *params = &encoder->params;
  int stride;
  int16_t *data;
  int i,j;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int subband_zero_flag;

  CARID_DEBUG("subband %d x %d at %d, %d", w, h, x, y);
  quant_index = 10;
  stride = params->iwt_chroma_width;
  data = (int16_t *)encoder->frame_buffer->data;
  quant_factor = carid_table_quant[quant_index];
  quant_offset = carid_table_offset[quant_index];
  //subband_zero_flag = 1;
  subband_zero_flag = 0;
  for(j=y;j<y+h;j++){
    for(i=x;i<x+w;i++){
      int v = data[j*stride + i];
      int sign;
      if (data[j*stride + i] < 0) {
        sign = 0;
        v = -v;
      } else {
        sign = 1;
      }
      v += quant_factor/2 - quant_offset;
      v /= quant_factor;
      if (sign == 0){
        v = -v;
      }
      data[j*stride + i] = v;
      if (v != 0) {
        subband_zero_flag = 0;
      }
    }
  }
  carid_bits_encode_bit (encoder->bits, subband_zero_flag);
  if (!subband_zero_flag) {
    int subband_length;

    carid_bits_encode_uegol (encoder->bits, quant_index);
    subband_length = 100;
    carid_bits_encode_uegol (encoder->bits, subband_length);
  } else {
    CARID_DEBUG ("subband is zero");
  }
  carid_bits_sync (encoder->bits);

  if (!subband_zero_flag) {
    for(j=y;j<y+h;j++){
      for(i=x;i<x+w;i++){
        int v = data[j*stride + i];
        int sign;
        if (v > 0) {
          sign = 1;
        } else {
          sign = 0;
          v = -v;
        }
        carid_bits_encode_uegol (encoder->bits, v);
        if (v) {
          carid_bits_encode_bit (encoder->bits, sign);
        }
      }
    }
    carid_bits_sync (encoder->bits);
  }
}

