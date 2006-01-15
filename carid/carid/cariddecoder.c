
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>



CaridDecoder *
carid_decoder_new (void)
{
  CaridDecoder *decoder;
  CaridParams *params;

  decoder = malloc(sizeof(CaridDecoder));
  memset (decoder, 0, sizeof(CaridDecoder));

  decoder->tmpbuf = malloc(1024 * 2);
  decoder->tmpbuf2 = malloc(1024 * 2);

  params = &decoder->params;

  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;

  return decoder;
}

void
carid_decoder_free (CaridDecoder *decoder)
{
  if (decoder->frame_buffer) {
    carid_buffer_unref (decoder->frame_buffer);
  }
  if (decoder->output_buffer) {
    carid_buffer_unref (decoder->output_buffer);
  }

  free (decoder->tmpbuf);
  free (decoder->tmpbuf2);
  free (decoder);
}

void
carid_decoder_set_output_buffer (CaridDecoder *decoder, CaridBuffer *buffer)
{
  if (decoder->output_buffer) {
    carid_buffer_unref (decoder->output_buffer);
  }
  decoder->output_buffer = buffer;
}


void
carid_decoder_set_size (CaridDecoder *decoder, int width, int height)
{
  CaridParams *params = &decoder->params;

  if (params->width == width && params->height == height) return;

  params->width = width;
  params->height = height;
  params->chroma_width =
    (width + params->chroma_h_scale - 1) / params->chroma_h_scale;
  params->chroma_height =
    (width + params->chroma_v_scale - 1) / params->chroma_v_scale;
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
carid_decoder_decode (CaridDecoder *decoder, CaridBuffer *buffer)
{
  int16_t *tmp = decoder->tmpbuf;
  int level;
  int i;
  int16_t *frame_data;
  uint8_t *dec_data;
  CaridParams *params = &decoder->params;
  
  dec_data = (uint8_t *)decoder->output_buffer->data;

  decoder->bits = carid_bits_new ();
  carid_bits_decode_init (decoder->bits, buffer);

  carid_decoder_decode_parse_header(decoder);
  carid_decoder_decode_rap(decoder);
  carid_decoder_decode_parse_header(decoder);
  carid_decoder_decode_frame_header(decoder);

  carid_coeff_decode_transform_parameters (decoder);

  if (decoder->frame_buffer == NULL) {
    decoder->frame_buffer = carid_buffer_new_and_alloc (params->iwt_luma_width * params->iwt_luma_height * 2);
  }

  carid_coeff_decode_transform_data (decoder);

  frame_data = (int16_t *)decoder->frame_buffer->data;
  for(level=params->transform_depth-1;level>=0;level--) {
    int w;
    int h;

    w = params->chroma_width >> level;
    h = params->chroma_height >> level;

    for(i=0;i<w;i++) {
      carid_interleave_str (tmp, frame_data + i, params->chroma_width*2, h);
      carid_lift_synth_str (params->wavelet_filter_index, frame_data + i,
          params->chroma_width*2, tmp, h);
    }
    for(i=0;i<h;i++) {
      carid_interleave (tmp, frame_data + i*params->chroma_width, w);
      carid_lift_synth (params->wavelet_filter_index,
          frame_data + i*params->chroma_width,
          tmp, w);
    }
  }

  for(i=0;i<params->height;i++){
    oil_convert_u8_s16 (dec_data + i*params->width,
        frame_data + i*params->chroma_width,
        params->width);
  }

  carid_buffer_unref (buffer);

  carid_bits_free (decoder->bits);
}

void
carid_decoder_decode_parse_header (CaridDecoder *decoder)
{
  int v1, v2, v3, v4;
  
  v1 = carid_bits_decode_bits (decoder->bits, 8);
  v2 = carid_bits_decode_bits (decoder->bits, 8);
  v3 = carid_bits_decode_bits (decoder->bits, 8);
  v4 = carid_bits_decode_bits (decoder->bits, 8);
  CARID_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    CARID_ERROR ("expected parse header");
    return;
  }


  decoder->code = carid_bits_decode_bits (decoder->bits, 8);
  CARID_DEBUG ("parse code %02x", decoder->code);

  decoder->next_parse_offset = carid_bits_decode_bits (decoder->bits, 24);
  CARID_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = carid_bits_decode_bits (decoder->bits, 24);
  CARID_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

void
carid_decoder_decode_rap (CaridDecoder *decoder)
{
  int bit;

  CARID_DEBUG("decoding RAP");
  /* parse parameters */
  decoder->rap_frame_number = carid_bits_decode_ue2gol (decoder->bits);
  CARID_DEBUG("rap frame number = %d", decoder->rap_frame_number);
  decoder->params.major_version = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("major_version = %d", decoder->params.major_version);
  decoder->params.minor_version = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("minor_version = %d", decoder->params.minor_version);
  decoder->params.profile = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("profile = %d", decoder->params.profile);
  decoder->params.level = carid_bits_decode_uegol (decoder->bits);
  CARID_DEBUG("level = %d", decoder->params.level);

  /* sequence parameters */
  decoder->params.video_format_index = carid_bits_decode_uegol (decoder->bits);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.width = carid_bits_decode_uegol (decoder->bits);
    decoder->params.height = carid_bits_decode_uegol (decoder->bits);
  } else {
    /* FIXME */
  }
  CARID_DEBUG("size = %d x %d", decoder->params.width,
      decoder->params.height);

  /* chroma format */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.chroma_format_index =
      carid_bits_decode_bits (decoder->bits, 3);
  } else {
    decoder->params.chroma_format_index = 0;
  }
  CARID_DEBUG("chroma_format_index %d",
      decoder->params.chroma_format_index);
  decoder->params.chroma_width = (decoder->params.width + 1)/2;
  decoder->params.chroma_height = (decoder->params.height + 1)/2;

  /* signal range */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.signal_range_index =
      carid_bits_decode_uegol (decoder->bits);
    if (decoder->params.signal_range_index == 0) {
      decoder->params.luma_offset = carid_bits_decode_uegol (decoder->bits);
      decoder->params.luma_excursion = carid_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_offset = carid_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_excursion =
        carid_bits_decode_uegol (decoder->bits);
    }
  } else {
    decoder->params.chroma_format_index = 0;
  }
  CARID_DEBUG("luma offset %d excursion %d",
      decoder->params.luma_offset, decoder->params.luma_excursion);

  /* display parameters */
  /* interlace */
  decoder->params.interlace = carid_bits_decode_bit (decoder->bits);
  if (decoder->params.interlace) {
    decoder->params.top_field_first = carid_bits_decode_bit (decoder->bits);
  }
  CARID_DEBUG("interlace %d top_field_first %d",
      decoder->params.interlace, decoder->params.top_field_first);

  /* frame rate */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.frame_rate_index = carid_bits_decode_uegol (decoder->bits);
    if (decoder->params.frame_rate_index == 0) {
      decoder->params.frame_rate_numerator = carid_bits_decode_uegol (decoder->bits);
      decoder->params.frame_rate_denominator = carid_bits_decode_uegol (decoder->bits);
    }
  }
  CARID_DEBUG("frame rate %d/%d", decoder->params.frame_rate_numerator,
      decoder->params.frame_rate_denominator);

  /* pixel aspect ratio */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.aspect_ratio_index = carid_bits_decode_uegol (decoder->bits);
    if (decoder->params.aspect_ratio_index == 0) {
      decoder->params.aspect_ratio_numerator =
        carid_bits_decode_uegol (decoder->bits);
      decoder->params.aspect_ratio_denominator =
        carid_bits_decode_uegol (decoder->bits);
    }
  }
  CARID_DEBUG("aspect ratio %d/%d", decoder->params.aspect_ratio_numerator,
      decoder->params.aspect_ratio_denominator);

  /* clean area */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.clean_tl_x = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_tl_y = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_width = carid_bits_decode_uegol (decoder->bits);
    decoder->params.clean_height = carid_bits_decode_uegol (decoder->bits);
  }
  CARID_DEBUG("clean offset %d %d",
      decoder->params.clean_tl_x,
      decoder->params.clean_tl_y);
  CARID_DEBUG("clean size %d %d",
      decoder->params.clean_width,
      decoder->params.clean_height);

  /* colur matrix */
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }

  /* signal range */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }

  /* colour spec */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }
  CARID_DEBUG("bit %d",bit);

  /* transfer characteristic */
  CARID_DEBUG("bit %d",bit);
  bit = carid_bits_decode_bit (decoder->bits);
  if (bit) {
    CARID_ERROR ("unimplemented");
    /* FIXME */
  }
  CARID_DEBUG("bit %d",bit);

  carid_bits_sync (decoder->bits);
}

void
carid_decoder_decode_frame_header (CaridDecoder *decoder)
{
  int n;
  int i;

  decoder->frame_number_offset = carid_bits_decode_se2gol (decoder->bits);
  CARID_DEBUG("frame number offset %d", decoder->frame_number_offset);

  n = carid_bits_decode_uegol (decoder->bits);
  for(i=0;i<n;i++){
    carid_bits_decode_se2gol (decoder->bits);
    CARID_DEBUG("retire %d", decoder->frame_number_offset);
  }

  carid_bits_sync (decoder->bits);
}

