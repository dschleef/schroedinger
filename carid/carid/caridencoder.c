
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

static void
notoil_splat_u16 (uint16_t *dest, uint16_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = src[0];
  }
}

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

  CARID_ERROR("encoded %d bytes", encoder->bits->offset/8);
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
    notoil_splat_u16 (frame_data + i*params->iwt_luma_width + params->width,
        frame_data + i*params->iwt_luma_width + params->width - 1,
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
  CaridParams *params = &encoder->params;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = 2*(params->iwt_luma_width << params->transform_depth);
  carid_encoder_encode_subband (encoder, 0, 0, w, h, stride,
      encoder->encoder_params.quant_index_dc);
  for(i=0; i < params->transform_depth; i++) {
    carid_encoder_encode_subband (encoder, 1, 1, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    carid_encoder_encode_subband (encoder, 0, 1, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    carid_encoder_encode_subband (encoder, 1, 0, w, h, stride,
        encoder->encoder_params.quant_index[i]);
    w <<= 1;
    h <<= 1;
    stride >>= 1;
  }
}

void
carid_encoder_encode_subband (CaridEncoder *encoder, int x, int y, int w, int h, int stride, int quant_index)
{
  //CaridParams *params = &encoder->params;
  int16_t *data;
  int i,j;
  int quant_factor;
  int quant_offset;
  int subband_zero_flag;

  CARID_DEBUG("subband %d x %d at %d, %d with stride %d", w, h, x, y, stride);
  stride >>= 1;
  data = (int16_t *)encoder->frame_buffer->data;
  data += x * w;
  data += y * (stride/2);
  quant_factor = carid_table_quant[quant_index];
  quant_offset = carid_table_offset[quant_index];
  subband_zero_flag = 1;
  //subband_zero_flag = 0;
  for(j=0;j<h;j++){
    for(i=0;i<w;i++){
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
#if 0
if (x != 0 || y != 1) {
  subband_zero_flag = 1;
}
#endif
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
        carid_bits_encode_uegol (encoder->bits, v);
        if (v) {
          carid_bits_encode_bit (encoder->bits, sign);
        }
      }
    }
    carid_bits_sync (encoder->bits);
  }
}


