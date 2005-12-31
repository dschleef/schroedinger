
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <carid/caridcoeff.h>


static unsigned int
round_up_pow2 (unsigned int x, int pow)
{
  unsigned int mask = (1<<pow) - 1;
  return (x + mask) & (~mask);
}

void
carid_coeff_decode_transform_parameters (CaridDecoder *decoder)
{
  int bit;

  /* transform */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    wavelet_filter_index = carid_bits_decode_uegol (decoder->bits);
    if (wavelet_filter_index > 4) {
      decoder->non_spec_input = TRUE;
    }
  } else {
    wavelet_filter_index = CARID_WAVELET_DAUB97;
  }

  /* transform depth */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    transform_depth = carid_bits_decode_uegol (decoder->bits);
    if (transform_depth > 6) {
      decoder->non_spec_input = TRUE;
    }
  } else {
    transform_depth = 4;
  }

  /* spatial partitioning */
  spatial_partition = carid_bits_decode_bit (decoder->bits);
  if (spatial_partition) {
    partition_index = carid_bits_decode_uegol (decoder->bits);
    if (partition_index > 1) {
      /* FIXME: ? */
      decoder->non_spec_input = TRUE;
    }
    if (partition_index == 0) {
      max_xblocks = carid_bits_decode_uegol (decoder->bits);
      max_yblocks = carid_bits_decode_uegol (decoder->bits);
    }
    multi_quant = carid_bits_decode_bit (decoder->bits);
  }

  if (is_intra) {
    iwt_chroma_width = round_up_pow2(chroma_width, transform_depth);
    iwt_chroma_height = round_up_pow2(chroma_height, transform_depth);
  } else {
    iwt_chroma_width = round_up_pow2(mc_chroma_width, transform_depth);
    iwt_chroma_height = round_up_pow2(mc_chroma_height, transform_depth);
  }
  iwt_luma_width = iwt_chroma_width * chroma_h_scale;
  iwt_luma_height = iwt_chroma_height * chroma_v_scale;


}

void
carid_coeff_decode_transform_data (CaridDecoder *decoder)
{
  int i;

  for(i=n_subbands; i > 0; i--) {
    carid_coeff_decode_subband (decoder, i);
  }
}

void
carid_coeff_decode_subband (CaridDecoder *decoder, int i)
{

}

