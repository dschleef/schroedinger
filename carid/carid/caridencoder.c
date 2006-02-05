
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
  int base_quant_index;
  int i;

  encoder = malloc(sizeof(CaridEncoder));
  memset (encoder, 0, sizeof(CaridEncoder));

  encoder->tmpbuf = malloc(1024 * 2);
  encoder->tmpbuf2 = malloc(1024 * 2);

  encoder->subband_buffer = carid_buffer_new_and_alloc (100000);

  params = &encoder->params;
  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;
  params->transform_depth = 4;

  base_quant_index = 18;
  encoder->encoder_params.quant_index_dc = 6;
  for(i=0;i<params->transform_depth;i++){
    encoder->encoder_params.quant_index[i] =
      base_quant_index - 2 * (params->transform_depth - 1 - i);
  }

  return encoder;
}

void
carid_encoder_free (CaridEncoder *encoder)
{
  if (encoder->frame_buffer) {
    carid_buffer_unref (encoder->frame_buffer[0]);
    carid_buffer_unref (encoder->frame_buffer[1]);
    carid_buffer_unref (encoder->frame_buffer[2]);
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

  if (encoder->frame_buffer[0]) {
    carid_buffer_unref (encoder->frame_buffer[0]);
    encoder->frame_buffer[0] = NULL;
  }
  if (encoder->frame_buffer[1]) {
    carid_buffer_unref (encoder->frame_buffer[1]);
    encoder->frame_buffer[1] = NULL;
  }
  if (encoder->frame_buffer[2]) {
    carid_buffer_unref (encoder->frame_buffer[2]);
    encoder->frame_buffer[2] = NULL;
  }

  encoder->need_rap = TRUE;
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

void
carid_encoder_push_buffer (CaridEncoder *encoder, CaridBuffer *buffer)
{
  encoder->frames[0] = buffer;
}

CaridBuffer *
carid_encoder_encode (CaridEncoder *encoder)
{
  CaridBuffer *outbuffer;
  CaridBuffer *subbuffer;
  
  outbuffer = carid_buffer_new_and_alloc (0x40000);

  encoder->bits = carid_bits_new ();
  carid_bits_encode_init (encoder->bits, outbuffer);

  if (encoder->need_rap) {
    carid_encoder_encode_rap (encoder);
    encoder->need_rap = FALSE;
  } else {
    if (encoder->frames[0]) {
      carid_encoder_encode_intra (encoder);
    }
  }

  CARID_DEBUG("encoded %d bits", encoder->bits->offset);
  carid_bits_free (encoder->bits);

  if (encoder->bits->offset > 0) {
    subbuffer = carid_buffer_new_subbuffer (outbuffer, 0,
        encoder->bits->offset/8);
  } else {
    subbuffer = NULL;
  }
  carid_buffer_unref (outbuffer);

  return subbuffer;
}

void
carid_encoder_iwt_transform (CaridEncoder *encoder, int component)
{
  int16_t *frame_data;
  CaridParams *params = &encoder->params;
  int16_t *tmp = encoder->tmpbuf;
  int width;
  int height;
  int level;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }
  
  frame_data = (int16_t *)encoder->frame_buffer[component]->data;
  for(level=0;level<params->transform_depth;level++) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = width << level;

    CARID_DEBUG("wavelet transform %dx%d stride %d", w, h, stride);
    carid_wavelet_transform_2d (params->wavelet_filter_index,
        frame_data, stride*2, w, h, tmp);
  }
}

void
carid_encoder_encode_intra (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;

  if (encoder->frame_buffer[0] == NULL) {
    encoder->frame_buffer[0] = carid_buffer_new_and_alloc (params->iwt_luma_width *
        params->iwt_luma_height * sizeof(int16_t));
    encoder->frame_buffer[1] = carid_buffer_new_and_alloc (params->iwt_chroma_width *
        params->iwt_chroma_height * sizeof(int16_t));
    encoder->frame_buffer[2] = carid_buffer_new_and_alloc (params->iwt_chroma_width *
        params->iwt_chroma_height * sizeof(int16_t));
  }

  carid_encoder_encode_frame_header (encoder);

  carid_encoder_copy_to_frame_buffer (encoder, encoder->frames[0]);

  carid_buffer_unref (encoder->frames[0]);
  encoder->frames[0] = NULL;

  carid_encoder_encode_transform_parameters (encoder);

  carid_encoder_iwt_transform (encoder, 0);
  carid_encoder_encode_transform_data (encoder, 0);

  carid_encoder_iwt_transform (encoder, 1);
  carid_encoder_encode_transform_data (encoder, 1);

  carid_encoder_iwt_transform (encoder, 2);
  carid_encoder_encode_transform_data (encoder, 2);

}

static void
carid_encoder_copy_and_extend (int16_t *dest, int iwt_width, int iwt_height,
    uint8_t *src, int width, int height)
{
  int i;

  for(i = 0; i<height; i++) {
    oil_convert_s16_u8 (dest, src, width);
    oil_splat_u16_ns ((uint16_t *)dest + width, (uint16_t *)dest + width - 1,
        iwt_width - width);
    dest += iwt_width;
    src += width;
  }
  for (i = 0; i < iwt_height - height; i++) {
    oil_memcpy (dest + i*iwt_width, dest - iwt_width, iwt_width);
  }
}

void
carid_encoder_copy_to_frame_buffer (CaridEncoder *encoder, CaridBuffer *buffer)
{
  CaridParams *params = &encoder->params;
  uint8_t *data;

  data = buffer->data;
  carid_encoder_copy_and_extend ((int16_t *)encoder->frame_buffer[0]->data,
      params->iwt_luma_width, params->iwt_luma_height,
      data, params->width, params->height);
  data += params->width * params->height;
  /* FIXME this isn't right for I420 */
  carid_encoder_copy_and_extend ((int16_t *)encoder->frame_buffer[1]->data,
      params->iwt_chroma_width, params->iwt_chroma_height,
      data, params->width/2, params->height/2);
  data += (params->width/2) * (params->height/2);
  carid_encoder_copy_and_extend ((int16_t *)encoder->frame_buffer[2]->data,
      params->iwt_chroma_width, params->iwt_chroma_height,
      data, params->width/2, params->height/2);

#if 0
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
#endif
}

void
carid_encoder_encode_rap (CaridEncoder *encoder)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'C', 8);
  carid_bits_encode_bits (encoder->bits, 'D', 8);

  carid_bits_encode_bits (encoder->bits, CARID_PARSE_CODE_RAP, 8);

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
  carid_bits_encode_bit (encoder->bits, TRUE);
  carid_bits_encode_uegol (encoder->bits, 0);
  carid_bits_encode_uegol (encoder->bits, 1);
  carid_bits_encode_uegol (encoder->bits, 1);

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
carid_encoder_init_subbands (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i;
  int w;
  int h;
  int stride;
  int chroma_w;
  int chroma_h;
  int chroma_stride;

  w = params->iwt_luma_width >> params->transform_depth;
  h = params->iwt_luma_height >> params->transform_depth;
  stride = sizeof(int16_t)*(params->iwt_luma_width << params->transform_depth);
  chroma_w = params->iwt_chroma_width >> params->transform_depth;
  chroma_h = params->iwt_chroma_height >> params->transform_depth;
  chroma_stride = sizeof(int16_t)*(params->iwt_chroma_width << params->transform_depth);

  encoder->subbands[0].x = 0;
  encoder->subbands[0].y = 0;
  encoder->subbands[0].w = w;
  encoder->subbands[0].h = h;
  encoder->subbands[0].offset = 0;
  encoder->subbands[0].stride = stride;
  encoder->subbands[0].chroma_w = chroma_w;
  encoder->subbands[0].chroma_h = chroma_h;
  encoder->subbands[0].chroma_offset = 0;
  encoder->subbands[0].chroma_stride = chroma_stride;
  encoder->subbands[0].has_parent = 0;
  encoder->subbands[0].scale_factor_shift = 0;
  encoder->subbands[0].horizontally_oriented = 0;
  encoder->subbands[0].vertically_oriented = 0;
  encoder->subbands[0].quant_index = encoder->encoder_params.quant_index_dc;

  for(i=0; i<params->transform_depth; i++) {
    encoder->subbands[1+3*i].x = 1;
    encoder->subbands[1+3*i].y = 1;
    encoder->subbands[1+3*i].w = w;
    encoder->subbands[1+3*i].h = h;
    encoder->subbands[1+3*i].offset = w + (stride/2/sizeof(int16_t));
    encoder->subbands[1+3*i].stride = stride;
    encoder->subbands[1+3*i].chroma_w = chroma_w;
    encoder->subbands[1+3*i].chroma_h = chroma_h;
    encoder->subbands[1+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    encoder->subbands[1+3*i].chroma_stride = chroma_stride;
    encoder->subbands[1+3*i].has_parent = (i>0);
    encoder->subbands[1+3*i].scale_factor_shift = i;
    encoder->subbands[1+3*i].horizontally_oriented = 0;
    encoder->subbands[1+3*i].vertically_oriented = 0;
    encoder->subbands[1+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    encoder->subbands[2+3*i].x = 0;
    encoder->subbands[2+3*i].y = 1;
    encoder->subbands[2+3*i].w = w;
    encoder->subbands[2+3*i].h = h;
    encoder->subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    encoder->subbands[2+3*i].stride = stride;
    encoder->subbands[2+3*i].chroma_w = chroma_w;
    encoder->subbands[2+3*i].chroma_h = chroma_h;
    encoder->subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    encoder->subbands[2+3*i].chroma_stride = chroma_stride;
    encoder->subbands[2+3*i].has_parent = (i>0);
    encoder->subbands[2+3*i].scale_factor_shift = i;
    encoder->subbands[2+3*i].horizontally_oriented = 0;
    encoder->subbands[2+3*i].vertically_oriented = 1;
    encoder->subbands[2+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    encoder->subbands[3+3*i].x = 1;
    encoder->subbands[3+3*i].y = 0;
    encoder->subbands[3+3*i].w = w;
    encoder->subbands[3+3*i].h = h;
    encoder->subbands[3+3*i].offset = w;
    encoder->subbands[3+3*i].stride = stride;
    encoder->subbands[3+3*i].chroma_w = chroma_w;
    encoder->subbands[3+3*i].chroma_h = chroma_h;
    encoder->subbands[3+3*i].chroma_offset = chroma_w;
    encoder->subbands[3+3*i].chroma_stride = chroma_stride;
    encoder->subbands[3+3*i].has_parent = (i>0);
    encoder->subbands[3+3*i].scale_factor_shift = i;
    encoder->subbands[3+3*i].horizontally_oriented = 1;
    encoder->subbands[3+3*i].vertically_oriented = 0;
    encoder->subbands[3+3*i].quant_index =
      encoder->encoder_params.quant_index[i];

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
carid_encoder_encode_transform_data (CaridEncoder *encoder, int component)
{
  int i;
  CaridParams *params = &encoder->params;

  carid_encoder_init_subbands (encoder);

  for (i=0;i < 1 + 3*params->transform_depth; i++) {
    carid_encoder_encode_subband (encoder, component, i);
  }
}


void
carid_encoder_encode_subband (CaridEncoder *encoder, int component, int index)
{
  CaridParams *params = &encoder->params;
  CaridSubband *subband = encoder->subbands + index;
  CaridSubband *parent_subband = NULL;
  CaridArith *arith;
  CaridBits *bits;
  int16_t *data;
  int16_t *parent_data = NULL;
  int i,j;
  int quant_factor;
  int quant_offset;
  int scale_factor;
  int subband_zero_flag;
  int ntop;
  int stride;
  int width;
  int height;
  int offset;

  if (component == 0) {
    stride = subband->stride >> 1;
    width = subband->w;
    height = subband->h;
    offset = subband->offset;
  } else {
    stride = subband->chroma_stride >> 1;
    width = subband->chroma_w;
    height = subband->chroma_h;
    offset = subband->chroma_offset;
  }

  CARID_DEBUG("subband index=%d %d x %d at offset %d with stride %d", index,
      width, height, offset, stride);

  data = (int16_t *)encoder->frame_buffer[component]->data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)encoder->frame_buffer[component]->data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)encoder->frame_buffer[component]->data +
        parent_subband->chroma_offset;
    }
  }
  quant_factor = carid_table_quant[subband->quant_index];
  quant_offset = carid_table_offset[subband->quant_index];

  scale_factor = 1<<(params->transform_depth - subband->quant_index);
  ntop = (scale_factor>>1) * quant_factor;

  bits = carid_bits_new ();
  carid_bits_encode_init (bits, encoder->subband_buffer);
  arith = carid_arith_new ();
  carid_arith_encode_init (arith, bits);
  carid_arith_init_contexts (arith);

  subband_zero_flag = 1;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      int v = data[j*stride + i];
      int sign;
      int parent_zero;
      int context;
      int context2;
      int nhood_sum;
      int previous_value;
      int sign_context;
      int pred_value;

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
//nhood_sum = 0;
      
      if (index == 0) {
        if (j>0) {
          if (i>0) {
            pred_value = (data[j*stride + i - 1] +
                data[(j-1)*stride + i] + data[(j-1)*stride + i - 1])/3;
          } else {
            pred_value = data[(j-1)*stride + i];
          }
        } else {
          if (i>0) {
            pred_value = data[j*stride + i - 1];
          } else {
            pred_value = 0;
          }
        }
      } else {
        pred_value = 0;
      }
//pred_value = 0;

      if (subband->has_parent) {
        if (parent_data[(j>>1)*(stride<<1) + (i>>1)]==0) {
          parent_zero = 1;
        } else {
          parent_zero = 0;
        }
      } else {
        if (subband->x == 0 && subband->y == 0) {
          parent_zero = 0;
        } else {
          parent_zero = 1;
        }
      }
//parent_zero = 0;

      previous_value = 0;
      if (subband->horizontally_oriented) {
        if (i > 0) {
          previous_value = data[j*stride + i - 1];
        }
      } else if (subband->vertically_oriented) {
        if (j > 0) {
          previous_value = data[(j-1)*stride + i];
        }
      }
//previous_value = 0;

      if (parent_zero) {
        if (nhood_sum == 0) {
          context = CARID_CTX_Z_BIN1z;
        } else {
          context = CARID_CTX_Z_BIN1nz;
        }
        context2 = CARID_CTX_Z_BIN2;
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
        context2 = CARID_CTX_NZ_BIN2;
      }

      if (previous_value > 0) {
        sign_context = CARID_CTX_SIGN_POS;
      } else if (previous_value < 0) {
        sign_context = CARID_CTX_SIGN_NEG;
      } else {
        sign_context = CARID_CTX_SIGN_ZERO;
      }
      
      v -= pred_value;
      if (v < 0) {
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

      carid_arith_context_encode_uu (arith, context, context2, v);
      if (v) {
        carid_arith_context_encode_bit (arith, sign_context, sign);
      }

      if (v) {
        if (sign) {
          data[j*stride + i] = pred_value + quant_offset + quant_factor * v;
        } else {
          data[j*stride + i] = pred_value - (quant_offset + quant_factor * v);
        }
      } else {
        data[j*stride + i] = pred_value;
      }
      
    }
  }

  carid_arith_flush (arith);
  CARID_DEBUG("encoded %d bits", bits->offset);
  carid_arith_free (arith);
  carid_bits_sync (bits);

  carid_bits_encode_bit (encoder->bits, subband_zero_flag);
  if (!subband_zero_flag) {
    carid_bits_encode_uegol (encoder->bits, subband->quant_index);
    carid_bits_encode_uegol (encoder->bits, bits->offset/8);

    carid_bits_sync (encoder->bits);

    carid_bits_append (encoder->bits, bits);
  } else {
    CARID_DEBUG ("subband is zero");
    carid_bits_sync (encoder->bits);
  }

  carid_bits_free (bits);
}


