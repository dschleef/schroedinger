
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
  params->transform_depth = 4;

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
  int i;
  uint8_t *data;
  int16_t *frame_data;
  CaridParams *params = &encoder->params;
  CaridBuffer *outbuffer;
  
  if (encoder->frame_buffer == NULL) {
    encoder->frame_buffer = carid_buffer_new_and_alloc (params->iwt_luma_width *
        params->iwt_luma_height * sizeof(int16_t));
  }

  outbuffer = carid_buffer_new_and_alloc (params->chroma_width *
      params->chroma_height * 2);

  encoder->bits = carid_bits_new ();
  carid_bits_encode_init (encoder->bits, outbuffer);

  carid_encoder_encode_rap (encoder);
  carid_encoder_encode_frame_header (encoder);

  data = (uint8_t *)buffer->data;
  frame_data = (int16_t *)encoder->frame_buffer->data;

  for(i = 0; i<params->height; i++) {
    oil_convert_s16_u8 (frame_data + i*params->chroma_width,
        data + i*params->width, params->width);
    notoil_splat_u16 (frame_data + i*params->chroma_width + params->width,
        frame_data + i*params->chroma_width + params->width - 1,
        params->chroma_width - params->width);
  }
  for (i = params->height; i < params->chroma_height; i++) {
    oil_memcpy (frame_data + i*params->chroma_width,
        frame_data + (params->height - 1)*params->chroma_width,
        params->chroma_width*2);
  }

  for(level=0;level<params->transform_depth;level++) {
    int w;
    int h;

    w = params->chroma_width >> level;
    h = params->chroma_height >> level;

    for(i=0;i<h;i++) {
      carid_lift_split (params->wavelet_filter_index, tmp,
          frame_data + i*params->chroma_width, w);
      carid_deinterleave (frame_data + i*params->chroma_width, tmp, w);
    }
    for(i=0;i<w;i++) {
      carid_lift_split_str (params->wavelet_filter_index, tmp,
          frame_data + i, params->chroma_width*2, h);
      carid_deinterleave_str (frame_data + i, params->chroma_width*2,
          tmp, h);
    }
  }

  carid_coeff_encode_transform_parameters (encoder);
  carid_coeff_encode_transform_data (encoder);

  carid_bits_free (encoder->bits);

  return outbuffer;
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

