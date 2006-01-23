
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>



CaridEncoder *
carid_encoder_new (void)
{
  CaridEncoder *encoder;
  CaridParams *params;

  encoder = malloc(sizeof(CaridEncoder));
  memset (encoder, 0, sizeof(CaridEncoder));

  encoder->tmpbuf = malloc(1024 * 2);
  encoder->tmpbuf2 = malloc(1024 * 2);

  params = &encoder->params;
  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;
  params->transform_depth = 6;

  encoder->encoder_params.quant_index_dc = 6;
  encoder->encoder_params.quant_index[0] = 6;
  encoder->encoder_params.quant_index[1] = 8;
  encoder->encoder_params.quant_index[2] = 12;
  encoder->encoder_params.quant_index[3] = 14;
  encoder->encoder_params.quant_index[4] = 16;
  encoder->encoder_params.quant_index[5] = 18;

  return encoder;
}

void
carid_encoder_free (CaridEncoder *encoder)
{
  if (encoder->frame_buffer) {
    carid_buffer_unref (encoder->frame_buffer);
  }

  free (encoder->tmpbuf);
  free (encoder->tmpbuf2);
  free (encoder);
}


void
carid_encoder_set_wavelet_type (CaridEncoder *encoder, int wavelet_type)
{
  encoder->params.wavelet_filter_index = wavelet_type;
  CARID_DEBUG("set wavelet %d", wavelet_type);
}

void
carid_encoder_set_size (CaridEncoder *encoder, int width, int height)
{
  CaridParams *params = &encoder->params;

  if (params->width == width && params->height == height) return;

  params->width = width;
  params->height = height;
  params->chroma_width =
    (width + params->chroma_h_scale - 1) / params->chroma_h_scale;
  params->chroma_height =
    (height + params->chroma_v_scale - 1) / params->chroma_v_scale;

  carid_params_calculate_iwt_sizes (params);

  if (encoder->frame_buffer) {
    carid_buffer_unref (encoder->frame_buffer);
    encoder->frame_buffer = NULL;
  }
}

#if 0
static int
round_up_pow2 (int x, int pow)
{
  x += (1<<pow) - 1;
  x &= ~((1<<pow) - 1);
  return x;
}
#endif

CaridBuffer *
carid_encoder_encode (CaridEncoder *encoder, CaridBuffer *buffer)
{
  int16_t *tmp = encoder->tmpbuf;
  int level;
  uint8_t *data;
  int16_t *frame_data;
  CaridParams *params = &encoder->params;
  CaridBuffer *outbuffer;
  
  if (encoder->frame_buffer == NULL) {
    encoder->frame_buffer = carid_buffer_new_and_alloc (params->iwt_luma_width *
        params->iwt_luma_height * sizeof(int16_t));
  }

  outbuffer = carid_buffer_new_and_alloc (params->iwt_luma_width *
      params->iwt_luma_height * 2);

  encoder->bits = carid_bits_new ();
  carid_bits_encode_init (encoder->bits, outbuffer);

  carid_encoder_encode_rap (encoder);
  carid_encoder_encode_frame_header (encoder);

  data = (uint8_t *)buffer->data;
  frame_data = (int16_t *)encoder->frame_buffer->data;

  carid_encoder_copy_to_frame_buffer (encoder, buffer);

  for(level=0;level<params->transform_depth;level++) {
    int w;
    int h;
    int stride;

    w = params->iwt_luma_width >> level;
    h = params->iwt_luma_height >> level;
    stride = params->iwt_luma_width << level;

    CARID_DEBUG("wavelet transform %dx%d stride %d", w, h, stride);
    carid_wavelet_transform_2d (params->wavelet_filter_index,
        frame_data, stride*2, w, h, tmp);
  }

  carid_encoder_encode_transform_parameters (encoder);
  carid_encoder_encode_transform_data (encoder);

  CARID_ERROR("encoded %d bits", encoder->bits->offset);
  carid_bits_free (encoder->bits);

  return outbuffer;
}

void
carid_encoder_copy_to_frame_buffer (CaridEncoder *encoder, CaridBuffer *buffer)
{
  CaridParams *params = &encoder->params;
  uint8_t *data;
  int16_t *frame_data;
  int i;

  data = (uint8_t *)buffer->data;
  frame_data = (int16_t *)encoder->frame_buffer->data;
  for(i = 0; i<params->height; i++) {
    oil_convert_s16_u8 (frame_data + i*params->iwt_luma_width,
        data + i*params->width, params->width);
    oil_splat_u16_ns ((uint16_t *)frame_data +
        i*params->iwt_luma_width + params->width,
        (uint16_t *)frame_data + i*params->iwt_luma_width + params->width - 1,
        params->iwt_luma_width - params->width);
  }
  for (i = params->height; i < params->iwt_luma_height; i++) {
    oil_memcpy (frame_data + i*params->iwt_luma_width,
        frame_data + (params->height - 1)*params->iwt_luma_width,
        params->iwt_luma_width*2);
  }
}

void
carid_encoder_encode_rap (CaridEncoder *encoder)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'C', 8);
  carid_bits_encode_bits (encoder->bits, 'D', 8);

  carid_bits_encode_bits (encoder->bits, 0xd7, 8);

  /* offsets */
  /* FIXME */
  carid_bits_encode_bits (encoder->bits, 0, 24);
  carid_bits_encode_bits (encoder->bits, 0, 24);

  /* rap frame number */
  /* FIXME */
  carid_bits_encode_ue2gol (encoder->bits, 0);

  /* major/minor version */
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 0);

  /* profile */
  carid_bits_encode_uegol (encoder->bits, 0);
  /* level */
  carid_bits_encode_uegol (encoder->bits, 0);


  /* sequence parameters */
  /* video format */
  carid_bits_encode_uegol (encoder->bits, 5);
  /* custom dimensions */
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, encoder->params.width);
  carid_bits_encode_uegol (encoder->bits, encoder->params.height);

  /* chroma format */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* signal range */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* display parameters */
  /* interlace */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* frame rate */
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 24);
  carid_bits_encode_uegol (encoder->bits, 1);

  /* pixel aspect ratio */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* clean area flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* colour matrix flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* signal_range flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* colour spec flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* transfer characteristic flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  carid_bits_sync (encoder->bits);
}

void
carid_encoder_encode_frame_header (CaridEncoder *encoder)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'C', 8);
  carid_bits_encode_bits (encoder->bits, 'D', 8);
  carid_bits_encode_bits (encoder->bits, 0xd2, 8);

  /* offsets */
  /* FIXME */
  carid_bits_encode_bits (encoder->bits, 0, 24);
  carid_bits_encode_bits (encoder->bits, 0, 24);

  /* frame number offset */
  /* FIXME */
  carid_bits_encode_se2gol (encoder->bits, 1);

  /* list */
  carid_bits_encode_uegol (encoder->bits, 0);
  //carid_bits_encode_se2gol (encoder->bits, -1);

  carid_bits_sync (encoder->bits);
}


void
carid_encoder_encode_transform_parameters (CaridEncoder *encoder)
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
carid_encoder_encode_transform_data (CaridEncoder *encoder)
{
  int i;
  int w,h;
  int stride;
  int index;
  CaridParams *params = &encoder->params;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = 2*(params->iwt_luma_width << params->transform_depth);
  index = 0;
  carid_encoder_encode_subband (encoder, index, w, h, stride,
      encoder->encoder_params.quant_index_dc);
  index++;
  for(i=0; i < params->transform_depth; i++) {
    carid_encoder_encode_subband (encoder, index, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    index++;
    carid_encoder_encode_subband (encoder, index, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    index++;
    carid_encoder_encode_subband (encoder, index, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    index++;
    w <<= 1;
    h <<= 1;
    stride >>= 1;
  }
}

struct subband_struct {
  int x;
  int y;
  int has_parent;
  int scale_factor_shift;
  int horizontally_oriented;
  int vertically_oriented;
};

struct subband_struct carid_encoder_subband_info[] = {
  { 0, 0, 0, 0, 0, 0 },
  { 1, 1, 0, 0, 0, 0 },
  { 0, 1, 0, 0, 0, 1 },
  { 1, 0, 0, 0, 1, 0 },
  { 1, 1, 1, 1, 0, 0 },
  { 0, 1, 1, 1, 0, 1 },
  { 1, 0, 1, 1, 1, 0 },
  { 1, 1, 1, 2, 0, 0 },
  { 0, 1, 1, 2, 0, 1 },
  { 1, 0, 1, 2, 1, 0 },
  { 1, 1, 1, 3, 0, 0 },
  { 0, 1, 1, 3, 0, 1 },
  { 1, 0, 1, 3, 1, 0 },
  { 1, 1, 1, 4, 0, 0 },
  { 0, 1, 1, 4, 0, 1 },
  { 1, 0, 1, 4, 1, 0 },
  { 1, 1, 1, 5, 0, 0 },
  { 0, 1, 1, 5, 0, 1 },
  { 1, 0, 1, 5, 1, 0 }
};


void
carid_encoder_encode_subband (CaridEncoder *encoder, int subband_index, int w,
    int h, int stride, int quant_index)
{
  CaridParams *params = &encoder->params;
  struct subband_struct *info = carid_encoder_subband_info + subband_index;
  CaridArith *arith;
  CaridBuffer *buffer;
  CaridBits *bits;
  int16_t *data;
  int16_t *parent_data;
  int i,j;
  int quant_factor;
  int quant_offset;
  int scale_factor;
  int subband_zero_flag;
  int ntop;

  CARID_DEBUG("subband index=%d %d x %d at %d, %d with stride %d", subband_index, w, h, stride);
  stride >>= 1;
  data = (int16_t *)encoder->frame_buffer->data;
  data += info->x * w;
  data += info->y * (stride/2);
  parent_data = (int16_t *)encoder->frame_buffer->data;
  parent_data += info->x * (w>>1);
  parent_data += info->y * ((stride<<1)/2);
  quant_factor = carid_table_quant[quant_index];
  quant_offset = carid_table_offset[quant_index];
  subband_zero_flag = 1;
  scale_factor = 1<<(params->transform_depth - quant_index);

  ntop = (scale_factor>>1) * quant_factor;

  buffer = carid_buffer_new_and_alloc (10000);
  bits = carid_bits_new ();
  carid_bits_encode_init (bits, buffer);
  arith = carid_arith_new ();
  carid_arith_encode_init (arith, bits);
  carid_arith_init_contexts (arith);

  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
      int v = data[j*stride + i];
      int sign;
      int parent_zero;
      int context;
      int nhood_sum;
      int previous_value;
      int sign_context;

      nhood_sum = 0;
      if (j>0) {
        nhood_sum += abs(data[(j-1)*stride + i]);
      }
      if (i>0) {
        nhood_sum += abs(data[j*stride + i - 1]);
      }
      if (i>0 && j>0) {
        nhood_sum += abs(data[(j-1)*stride + i - 1]);
      }

      if (data[j*stride + i] < 0) {
        sign = 0;
        v = -v;
      } else {
        sign = 1;
      }
      v += quant_factor/2 - quant_offset;
      v /= quant_factor;
      if (v != 0) {
        subband_zero_flag = 0;
      }
      if (v) {
        if (sign) {
          data[j*stride + i] = quant_offset + quant_factor * v;
        } else {
          data[j*stride + i] = -(quant_offset + quant_factor * v);
        }
      } else {
        data[j*stride + i] = 0;
      }
      
      if (info->has_parent) {
        if (parent_data[(j>>1)*(stride<<1) + (i>>1)]==0) {
          parent_zero = 1;
        } else {
          parent_zero = 0;
        }
      } else {
        if (info->x == 0 && info->y == 0) {
          parent_zero = 0;
        } else {
          parent_zero = 1;
        }
      }
      if (parent_zero) {
        if (nhood_sum == 0) {
          context = CARID_CTX_Z_BIN1z;
        } else {
          context = CARID_CTX_Z_BIN1nz;
        }
      } else {
        if (nhood_sum == 0) {
          context = CARID_CTX_NZ_BIN1z;
        } else {
          if (nhood_sum <= ntop) {
            context = CARID_CTX_NZ_BIN1a;
          } else {
            context = CARID_CTX_NZ_BIN1b;
          }
        }
      }

      previous_value = 0;
      if (info->horizontally_oriented) {
        if (i > 0) {
          previous_value = data[j*stride + i - 1];
        }
      } else if (info->vertically_oriented) {
        if (j > 0) {
          previous_value = data[(j-1)*stride + i];
        }
      }

      if (previous_value > 0) {
        sign_context = CARID_CTX_SIGN_POS;
      } else if (previous_value < 0) {
        sign_context = CARID_CTX_SIGN_NEG;
      } else {
        sign_context = CARID_CTX_SIGN_ZERO;
      }
      
      carid_arith_context_encode_uu (arith, context, v);
      if (v) {
        carid_arith_context_binary_encode (arith, sign_context, sign);
      }
    }
  }

  carid_arith_flush (arith);
  CARID_DEBUG("encoded %d bits", bits->offset);
  carid_arith_free (arith);
  carid_bits_sync (bits);

#if 0
if (x != 0 || y != 1) {
  subband_zero_flag = 1;
}
#endif
  carid_bits_encode_bit (encoder->bits, subband_zero_flag);
  if (!subband_zero_flag) {
    carid_bits_encode_uegol (encoder->bits, quant_index);
    carid_bits_encode_uegol (encoder->bits, bits->offset/8);

    carid_bits_sync (encoder->bits);

    carid_bits_append (encoder->bits, bits);
  } else {
    CARID_DEBUG ("subband is zero");
    carid_bits_sync (encoder->bits);
  }
  //carid_bits_sync (encoder->bits);

#if 0
  if (!subband_zero_flag) {
    CaridArith *arith;
    int offset;

    offset = encoder->bits->offset;

    arith = carid_arith_new ();
    carid_arith_encode_init (arith, encoder->bits);
    carid_arith_context_init (arith, 0, 1, 1);

    for(j=0;j<h;j++){
      for(i=0;i<w;i++){
        int v = data[j*stride + i];
        int sign;
        if (v > 0) {
          sign = 1;
        } else {
          sign = 0;
          v = -v;
        }
        carid_arith_context_encode_uegol (arith, 0, v);
        if (v) {
          carid_arith_context_binary_encode (arith, 0, sign);
        }
      }
    }
    carid_arith_flush (arith);
    CARID_ERROR("encoded %d bits", encoder->bits->offset - offset);
    carid_arith_free (arith);
    carid_bits_sync (encoder->bits);
  }
#endif

  carid_bits_free (bits);
  carid_buffer_unref (buffer);
}


