
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
  if (encoder->frame) {
    carid_frame_free (encoder->frame);
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

  if (encoder->frame) {
    carid_frame_free (encoder->frame);
    encoder->frame = NULL;
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
carid_encoder_push_frame (CaridEncoder *encoder, CaridFrame *frame)
{
  encoder->frame_queue[0] = frame;
}

CaridBuffer *
carid_encoder_encode (CaridEncoder *encoder)
{
  CaridBuffer *outbuffer;
  CaridBuffer *subbuffer;
  
  if (!encoder->need_rap && encoder->frame_queue[0] == NULL) {
    /* nothing to do */
    return NULL;
  }

  outbuffer = carid_buffer_new_and_alloc (0x10000);

  encoder->bits = carid_bits_new ();
  carid_bits_encode_init (encoder->bits, outbuffer);

  if (encoder->need_rap) {
    carid_encoder_encode_rap (encoder);
    encoder->need_rap = FALSE;
  } else {
    CARID_DEBUG("frame number %d", encoder->frame_number);
    if ((encoder->frame_number & 7) == 0) {
      carid_encoder_encode_intra (encoder);
    } else {
      carid_encoder_encode_inter (encoder);
    }

    encoder->frame_number++;
  }

  CARID_ERROR("encoded %d bits", encoder->bits->offset);

  if (encoder->bits->offset > 0) {
    subbuffer = carid_buffer_new_subbuffer (outbuffer, 0,
        encoder->bits->offset/8);
  } else {
    subbuffer = NULL;
  }
  carid_bits_free (encoder->bits);
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
  
  frame_data = (int16_t *)encoder->frame->components[component].data;
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
carid_encoder_inverse_iwt_transform (CaridEncoder *encoder, int component)
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
  
  frame_data = (int16_t *)encoder->frame->components[component].data;
  for(level=params->transform_depth-1; level >=0;level--) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = width << level;

    carid_wavelet_inverse_transform_2d (params->wavelet_filter_index,
        frame_data, stride*2, w, h, tmp);
  }
}

void
carid_encoder_encode_intra (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int is_ref = 1;

  if (encoder->frame == NULL) {
    encoder->frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  carid_encoder_encode_frame_header (encoder, CARID_PARSE_CODE_INTRA_REF);

  carid_frame_convert (encoder->frame, encoder->frame_queue[0]);

  carid_frame_free (encoder->frame_queue[0]);
  encoder->frame_queue[0] = NULL;

  carid_encoder_encode_transform_parameters (encoder);

  carid_encoder_iwt_transform (encoder, 0);
  carid_encoder_encode_transform_data (encoder, 0);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 0);
  }

  carid_encoder_iwt_transform (encoder, 1);
  carid_encoder_encode_transform_data (encoder, 1);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 1);
  }

  carid_encoder_iwt_transform (encoder, 2);
  carid_encoder_encode_transform_data (encoder, 2);
  if (is_ref) {
    CaridFrame *ref_frame;

    carid_encoder_inverse_iwt_transform (encoder, 2);

    ref_frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    carid_frame_convert (ref_frame, encoder->frame);

    if (encoder->reference_frames[0]) {
      carid_frame_free (encoder->reference_frames[0]);
    }
    encoder->reference_frames[0] = ref_frame;
  }

}

void
carid_encoder_encode_inter (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  //int is_ref = 0;

  if (encoder->frame == NULL) {
    encoder->frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_S16,
        params->iwt_luma_width, params->iwt_luma_height, 2, 2);
  }

  carid_encoder_encode_frame_header (encoder, CARID_PARSE_CODE_INTER_NON_REF);

  carid_encoder_motion_predict (encoder);

  carid_encoder_encode_frame_prediction (encoder);

  carid_frame_free (encoder->frame_queue[0]);
  encoder->frame_queue[0] = NULL;
#if 0
  carid_frame_convert (encoder->frame, encoder->frame_queue[0]);

  carid_frame_free (encoder->frame_queue[0]);
  encoder->frame_queue[0] = NULL;

  carid_encoder_encode_transform_parameters (encoder);

  carid_encoder_iwt_transform (encoder, 0);
  carid_encoder_encode_transform_data (encoder, 0);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 0);
  }

  carid_encoder_iwt_transform (encoder, 1);
  carid_encoder_encode_transform_data (encoder, 1);
  if (is_ref) {
    carid_encoder_inverse_iwt_transform (encoder, 1);
  }

  carid_encoder_iwt_transform (encoder, 2);
  carid_encoder_encode_transform_data (encoder, 2);
  if (is_ref) {
    CaridFrame *ref_frame;

    carid_encoder_inverse_iwt_transform (encoder, 2);

    ref_frame = carid_frame_new_and_alloc (CARID_FRAME_FORMAT_U8,
        params->width, params->height, 2, 2);
    carid_frame_convert (ref_frame, encoder->frame);

    if (encoder->reference_frames[0]) {
      carid_frame_free (encoder->reference_frames[0]);
    }
    encoder->reference_frames[0] = ref_frame;
  }
#endif
}

void
carid_encoder_encode_frame_prediction (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i,j;

  /* block params flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* mv precision flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* global motion flag */
  carid_bits_encode_bit (encoder->bits, FALSE);

  /* block data length */
  carid_bits_encode_uegol (encoder->bits, 100);

  carid_bits_sync (encoder->bits);

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      carid_bits_encode_segol(encoder->bits,
          encoder->motion_x[j*(4*params->x_num_mb) + i]);
      carid_bits_encode_segol(encoder->bits,
          encoder->motion_y[j*(4*params->x_num_mb) + i]);
    }
  }

  carid_bits_sync (encoder->bits);
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
carid_encoder_encode_frame_header (CaridEncoder *encoder,
    int parse_code)
{
  
  /* parse parameters */
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'B', 8);
  carid_bits_encode_bits (encoder->bits, 'C', 8);
  carid_bits_encode_bits (encoder->bits, 'D', 8);
  carid_bits_encode_bits (encoder->bits, parse_code, 8);

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

  data = (int16_t *)encoder->frame->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband - 3;
    if (component == 0) {
      parent_data = (int16_t *)encoder->frame->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)encoder->frame->components[component].data +
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


static void predict_motion (CaridFrame *frame, CaridFrame *reference_frame,
    int x, int y, int w, int h, int *pred_x, int *pred_y);

void
carid_encoder_motion_predict (CaridEncoder *encoder)
{
  CaridParams *params = &encoder->params;
  int i;
  int j;
  CaridFrame *ref_frame;
  CaridFrame *frame;
  int sum_pred_x;
  int sum_pred_y;
  double pan_x, pan_y;
  double mag_x, mag_y;
  double skew_x, skew_y;
  double sum_x, sum_y;

  params->xbsep_luma = 8;
  params->ybsep_luma = 8;

  params->x_num_mb = encoder->params.width / (4*params->xbsep_luma);
#if 0
  if (encoder->width % (4*xbsep_luma)) {
    x_num_mb++;
  }
#endif
  params->y_num_mb = encoder->params.height / (4*params->ybsep_luma);
#if 0
  if (encoder->height % (4*ybsep_luma)) {
    y_num_mb++;
  }
#endif

  if (encoder->motion_x == NULL) {
    encoder->motion_x = malloc(sizeof(int16_t)*params->x_num_mb*params->y_num_mb*16);
    encoder->motion_y = malloc(sizeof(int16_t)*params->x_num_mb*params->y_num_mb*16);
  }

  ref_frame = encoder->reference_frames[0];
  if (!ref_frame) {
    CARID_ERROR("no reference frame");
  }
  frame = encoder->frame_queue[0];

  sum_pred_x = 0;
  sum_pred_y = 0;
  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      int x,y;
      int pred_x, pred_y;

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      predict_motion (frame, ref_frame, x, y, params->xbsep_luma, params->ybsep_luma,
          &pred_x, &pred_y);

      encoder->motion_x[j*(4*params->x_num_mb) + i] = pred_x;
      encoder->motion_y[j*(4*params->x_num_mb) + i] = pred_y;

      sum_pred_x += pred_x;
      sum_pred_y += pred_y;
    }
  }

  pan_x = ((double)sum_pred_x)/(16*params->x_num_mb*params->y_num_mb);
  pan_y = ((double)sum_pred_y)/(16*params->x_num_mb*params->y_num_mb);

  mag_x = 0;
  mag_y = 0;
  skew_x = 0;
  skew_y = 0;
  sum_x = 0;
  sum_y = 0;
  for(j=0;j<4*params->y_num_mb;j++) {
    for(i=0;i<4*params->x_num_mb;i++) {
      double x;
      double y;

      x = i*params->xbsep_luma - (2*params->x_num_mb - 0.5);
      y = j*params->ybsep_luma - (2*params->y_num_mb - 0.5);

      mag_x += encoder->motion_x[j*(4*params->x_num_mb) + i] * x;
      mag_y += encoder->motion_y[j*(4*params->x_num_mb) + i] * y;

      skew_x += encoder->motion_x[j*(4*params->x_num_mb) + i] * y;
      skew_y += encoder->motion_y[j*(4*params->x_num_mb) + i] * x;

      sum_x += x * x;
      sum_y += y * y;
    }
  }
  mag_x = mag_x/sum_x;
  mag_y = mag_y/sum_y;
  skew_x = skew_x/sum_x;
  skew_y = skew_y/sum_y;

  CARID_ERROR("pan %6.3f %6.3f mag %6.3f %6.3f skew %6.3f %6.3f",
      pan_x, pan_y, mag_x, mag_y, skew_x, skew_y);

}

static int
calculate_metric (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;

  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      metric += abs (a[j*a_stride + i] - b[j*b_stride + i]);
    }
  }

  return metric;
}

static void
predict_motion (CaridFrame *frame, CaridFrame *reference_frame,
    int x, int y, int w, int h, int *pred_x, int *pred_y)
{
  int dx, dy;
  uint8_t *data = frame->components[0].data;
  int stride = frame->components[0].stride;
  uint8_t *ref_data = reference_frame->components[0].data;
  int ref_stride = reference_frame->components[0].stride;
  int metric;
  int min_metric;
  int step_size;

#if 0
  min_metric = calculate_metric (data + y * stride + x, stride,
      ref_data + y * ref_stride + x, ref_stride, w, h);
  *pred_x = 0;
  *pred_y = 0;

  printf("mp %d %d metric %d\n", x, y, min_metric);
#endif

  dx = 0;
  dy = 0;
  step_size = 8;
  while (step_size > 0) {
    static const int hx[5] = { 0, 0, -1, 0, 1 };
    static const int hy[5] = { 0, -1, 0, 1, 0 };
    int px, py;
    int min_index;
    int i;

    min_index = 0;
    min_metric = calculate_metric (data + y * stride + x, stride,
          ref_data + (y + dy) * ref_stride + x + dx, ref_stride,
          w, h);
    for(i=1;i<5;i++){
      px = x + dx + hx[i] * step_size;
      py = y + dy + hy[i] * step_size;
      if (px < 0) px = 0;
      if (py < 0) py = 0;
      if (px + w > reference_frame->components[0].width) {
        px = reference_frame->components[0].width - w;
      }
      if (py + h > reference_frame->components[0].height) {
        py = reference_frame->components[0].height - h;
      }

      metric = calculate_metric (data + y * stride + x, stride,
          ref_data + py * ref_stride + px, ref_stride, w, h);

      if (metric < min_metric) {
        min_metric = metric;
        min_index = i;
      }
    }

    if (min_index == 0) {
      step_size >>= 1;
    } else {
      dx += hx[min_index] * step_size;
      dy += hy[min_index] * step_size;
    }
  }
  *pred_x = dx;
  *pred_y = dy;

#if 0
  for(dy = -4; dy <= 4; dy++) {
    for(dx = -4; dx <= 4; dx++) {
      if (y + dy < 0) continue;
      if (x + dx < 0) continue;
      if (y + dy + h > reference_frame->components[0].height) continue;
      if (x + dx + w > reference_frame->components[0].width) continue;

      metric = calculate_metric (data + y * stride + x, stride,
          ref_data + (y + dy) * ref_stride + x + dx, ref_stride,
          w, h);

      printf(" %d", metric);
      if (metric < min_metric) {
        min_metric = metric;
        *pred_x = dx;
        *pred_y = dy;
      }

    }
    printf("\n");
  }
#endif

}


