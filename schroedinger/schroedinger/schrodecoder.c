
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schro/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


static void schro_decoder_decode_macroblock(SchroDecoder *decoder, int i,
    int j);
static void schro_decoder_decode_prediction_unit(SchroDecoder *decoder,
    SchroMotionVector *mv);
static void schro_decoder_predict (SchroDecoder *decoder);


SchroDecoder *
schro_decoder_new (void)
{
  SchroDecoder *decoder;
  SchroParams *params;

  decoder = malloc(sizeof(SchroDecoder));
  memset (decoder, 0, sizeof(SchroDecoder));

  decoder->tmpbuf = malloc(1024 * 2);
  decoder->tmpbuf2 = malloc(1024 * 2);

  params = &decoder->params;

  params->is_intra = TRUE;
  params->chroma_h_scale = 2;
  params->chroma_v_scale = 2;

  return decoder;
}

void
schro_decoder_free (SchroDecoder *decoder)
{
  if (decoder->frame) {
    schro_frame_free (decoder->frame);
  }
  if (decoder->output_frame) {
    schro_frame_free (decoder->output_frame);
  }

  free (decoder->tmpbuf);
  free (decoder->tmpbuf2);
  free (decoder);
}

void
schro_decoder_set_output_frame (SchroDecoder *decoder, SchroFrame *frame)
{
  if (decoder->output_frame) {
    schro_frame_free (decoder->output_frame);
  }
  decoder->output_frame = frame;
}

int
schro_decoder_is_parse_header (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  return 1;
}

int
schro_decoder_is_rap (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (data[4] == SCHRO_PARSE_CODE_RAP) return 1;

  return 0;
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
schro_decoder_decode (SchroDecoder *decoder, SchroBuffer *buffer)
{
  SchroParams *params = &decoder->params;
  
  decoder->bits = schro_bits_new ();
  schro_bits_decode_init (decoder->bits, buffer);

  if (schro_decoder_is_rap (buffer)) {
    schro_decoder_decode_parse_header(decoder);
    schro_decoder_decode_rap(decoder);

    schro_buffer_unref (buffer);
    schro_bits_free (decoder->bits);
    return;
  }

  schro_decoder_decode_parse_header(decoder);
  schro_decoder_decode_frame_header(decoder);

  if (decoder->code == SCHRO_PARSE_CODE_INTRA_REF) {
    SCHRO_ERROR("intra ref");
    schro_decoder_decode_transform_parameters (decoder);

    if (decoder->frame == NULL) {
      decoder->frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
          params->iwt_luma_width, params->iwt_luma_height, 2, 2);
    }

    schro_decoder_decode_transform_data (decoder, 0);
    schro_decoder_iwt_transform (decoder, 0);

    schro_decoder_decode_transform_data (decoder, 1);
    schro_decoder_iwt_transform (decoder, 1);

    schro_decoder_decode_transform_data (decoder, 2);
    schro_decoder_iwt_transform (decoder, 2);

    schro_frame_convert (decoder->output_frame, decoder->frame);

    if (decoder->reference_frames[0] == NULL) {
      decoder->reference_frames[0] = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
          params->width, params->height, 2, 2);
    }

    schro_frame_convert (decoder->reference_frames[0], decoder->frame);
  } else if (decoder->code == SCHRO_PARSE_CODE_INTER_NON_REF) {
    SCHRO_ERROR("inter non-ref");
    schro_decoder_decode_frame_prediction (decoder);

    /* FIXME */
    SCHRO_ASSERT(params->xbsep_luma == 8);
    SCHRO_ASSERT(params->ybsep_luma == 8);

    if (decoder->mc_tmp_frame == NULL) {
      decoder->mc_tmp_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
          params->mc_luma_width, params->mc_luma_height, 2, 2);
    }

    if (decoder->motion_vectors == NULL) {
      decoder->motion_vectors = malloc(sizeof(SchroMotionVector) *
          params->x_num_blocks * params->y_num_blocks);
    }

    schro_decoder_decode_prediction_data (decoder);
    schro_decoder_predict (decoder);

    schro_frame_convert (decoder->output_frame, decoder->mc_tmp_frame);
  }

  schro_buffer_unref (buffer);
  schro_bits_free (decoder->bits);
}

#if 0
void
schro_decoder_copy_from_frame_buffer (SchroDecoder *decoder, SchroBuffer *buffer)
{
  SchroParams *params = &decoder->params;
  int i;
  uint8_t *data;
  int16_t *frame_data;

  data = buffer->data;

  frame_data = (int16_t *)decoder->frame_buffer[0]->data;
  for(i=0;i<params->height;i++){
    oil_convert_u8_s16 (data, frame_data, params->width);
    data += params->width;
    frame_data += params->iwt_luma_width;
  }

  frame_data = (int16_t *)decoder->frame_buffer[1]->data;
  for(i=0;i<params->height/2;i++){
    oil_convert_u8_s16 (data, frame_data, params->width/2);
    data += params->width/2;
    frame_data += params->iwt_chroma_width;
  }

  frame_data = (int16_t *)decoder->frame_buffer[2]->data;
  for(i=0;i<params->height/2;i++){
    oil_convert_u8_s16 (data, frame_data, params->width/2);
    data += params->width/2;
    frame_data += params->iwt_chroma_width;
  }
}
#endif

void
schro_decoder_iwt_transform (SchroDecoder *decoder, int component)
{
  SchroParams *params = &decoder->params;
  int16_t *frame_data;
  int height;
  int width;
  int level;

  if (component == 0) {
    width = params->iwt_luma_width;
    height = params->iwt_luma_height;
  } else {
    width = params->iwt_chroma_width;
    height = params->iwt_chroma_height;
  }

  frame_data = (int16_t *)decoder->frame->components[component].data;
  for(level=params->transform_depth-1;level>=0;level--) {
    int w;
    int h;
    int stride;

    w = width >> level;
    h = height >> level;
    stride = 2*(width << level);

    schro_wavelet_inverse_transform_2d (params->wavelet_filter_index,
        frame_data, stride, w, h, decoder->tmpbuf);
  }

}

void
schro_decoder_decode_parse_header (SchroDecoder *decoder)
{
  int v1, v2, v3, v4;
  
  v1 = schro_bits_decode_bits (decoder->bits, 8);
  v2 = schro_bits_decode_bits (decoder->bits, 8);
  v3 = schro_bits_decode_bits (decoder->bits, 8);
  v4 = schro_bits_decode_bits (decoder->bits, 8);
  SCHRO_DEBUG ("parse header %02x %02x %02x %02x", v1, v2, v3, v4);
  if (v1 != 'B' || v2 != 'B' || v3 != 'C' || v4 != 'D') {
    SCHRO_ERROR ("expected parse header");
    return;
  }


  decoder->code = schro_bits_decode_bits (decoder->bits, 8);
  SCHRO_DEBUG ("parse code %02x", decoder->code);

  decoder->next_parse_offset = schro_bits_decode_bits (decoder->bits, 24);
  SCHRO_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = schro_bits_decode_bits (decoder->bits, 24);
  SCHRO_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

void
schro_decoder_decode_rap (SchroDecoder *decoder)
{
  int bit;
  int index;

  SCHRO_DEBUG("decoding RAP");
  /* parse parameters */
  decoder->rap_frame_number = schro_bits_decode_ue2gol (decoder->bits);
  SCHRO_DEBUG("rap frame number = %d", decoder->rap_frame_number);
  decoder->params.major_version = schro_bits_decode_uegol (decoder->bits);
  SCHRO_DEBUG("major_version = %d", decoder->params.major_version);
  decoder->params.minor_version = schro_bits_decode_uegol (decoder->bits);
  SCHRO_DEBUG("minor_version = %d", decoder->params.minor_version);
  decoder->params.profile = schro_bits_decode_uegol (decoder->bits);
  SCHRO_DEBUG("profile = %d", decoder->params.profile);
  decoder->params.level = schro_bits_decode_uegol (decoder->bits);
  SCHRO_DEBUG("level = %d", decoder->params.level);

  /* sequence parameters */
  index = schro_bits_decode_uegol (decoder->bits);
  schro_params_set_video_format (&decoder->params, index);

  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.width = schro_bits_decode_uegol (decoder->bits);
    decoder->params.height = schro_bits_decode_uegol (decoder->bits);
  }
  SCHRO_DEBUG("size = %d x %d", decoder->params.width,
      decoder->params.height);

  /* chroma format */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.chroma_format_index =
      schro_bits_decode_bits (decoder->bits, 3);
  } else {
    decoder->params.chroma_format_index = 0;
  }
  SCHRO_DEBUG("chroma_format_index %d",
      decoder->params.chroma_format_index);
  decoder->params.chroma_width = (decoder->params.width + 1)/2;
  decoder->params.chroma_height = (decoder->params.height + 1)/2;

  /* signal range */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.luma_offset = schro_bits_decode_uegol (decoder->bits);
      decoder->params.luma_excursion = schro_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_offset = schro_bits_decode_uegol (decoder->bits);
      decoder->params.chroma_excursion =
        schro_bits_decode_uegol (decoder->bits);
    } else {
      /* FIXME */
      //schro_params_set_excursion (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("luma offset %d excursion %d",
      decoder->params.luma_offset, decoder->params.luma_excursion);

  /* display parameters */
  /* interlace */
  decoder->params.interlace = schro_bits_decode_bit (decoder->bits);
  if (decoder->params.interlace) {
    decoder->params.top_field_first = schro_bits_decode_bit (decoder->bits);
  }
  SCHRO_DEBUG("interlace %d top_field_first %d",
      decoder->params.interlace, decoder->params.top_field_first);

  /* frame rate */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.frame_rate_numerator = schro_bits_decode_uegol (decoder->bits);
      decoder->params.frame_rate_denominator = schro_bits_decode_uegol (decoder->bits);
    } else {
      schro_params_set_frame_rate (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("frame rate %d/%d", decoder->params.frame_rate_numerator,
      decoder->params.frame_rate_denominator);

  /* pixel aspect ratio */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      decoder->params.pixel_aspect_ratio_numerator =
        schro_bits_decode_uegol (decoder->bits);
      decoder->params.pixel_aspect_ratio_denominator =
        schro_bits_decode_uegol (decoder->bits);
    } else {
      schro_params_set_pixel_aspect_ratio (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("aspect ratio %d/%d",
      decoder->params.pixel_aspect_ratio_numerator,
      decoder->params.pixel_aspect_ratio_denominator);

  /* clean area */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.clean_tl_x = schro_bits_decode_uegol (decoder->bits);
    decoder->params.clean_tl_y = schro_bits_decode_uegol (decoder->bits);
    decoder->params.clean_width = schro_bits_decode_uegol (decoder->bits);
    decoder->params.clean_height = schro_bits_decode_uegol (decoder->bits);
  }
  SCHRO_DEBUG("clean offset %d %d", decoder->params.clean_tl_x,
      decoder->params.clean_tl_y);
  SCHRO_DEBUG("clean size %d %d", decoder->params.clean_width,
      decoder->params.clean_height);

  /* colur matrix */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    SCHRO_ERROR ("unimplemented");
    /* FIXME */
  }

  /* signal range */
  SCHRO_DEBUG("bit %d",bit);
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    SCHRO_ERROR ("unimplemented");
    /* FIXME */
  }

  /* colour spec */
  SCHRO_DEBUG("bit %d",bit);
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    SCHRO_ERROR ("unimplemented");
    /* FIXME */
  }
  SCHRO_DEBUG("bit %d",bit);

  /* transfer characteristic */
  SCHRO_DEBUG("bit %d",bit);
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    SCHRO_ERROR ("unimplemented");
    /* FIXME */
  }
  SCHRO_DEBUG("bit %d",bit);

  schro_bits_sync (decoder->bits);
}

void
schro_decoder_decode_frame_header (SchroDecoder *decoder)
{
  int n;
  int i;

  decoder->frame_number_offset = schro_bits_decode_se2gol (decoder->bits);
  SCHRO_DEBUG("frame number offset %d", decoder->frame_number_offset);

  n = schro_bits_decode_uegol (decoder->bits);
  for(i=0;i<n;i++){
    schro_bits_decode_se2gol (decoder->bits);
    SCHRO_DEBUG("retire %d", decoder->frame_number_offset);
  }

  schro_bits_sync (decoder->bits);
}

void
schro_decoder_decode_frame_prediction (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  int bit;
  int length;
  int index;

  /* block params flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uegol (decoder->bits);
    if (index == 0) {
      params->xblen_luma = schro_bits_decode_uegol (decoder->bits);
      params->yblen_luma = schro_bits_decode_uegol (decoder->bits);
      params->xbsep_luma = schro_bits_decode_uegol (decoder->bits);
      params->ybsep_luma = schro_bits_decode_uegol (decoder->bits);
    } else {
      schro_params_set_block_params (params, index);
    }
  }
  SCHRO_ERROR("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  /* mv precision flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->mv_precision = schro_bits_decode_uegol (decoder->bits);
  }
  SCHRO_ERROR("mv_precision %d", params->mv_precision);

  /* global motion flag */
  params->global_motion = schro_bits_decode_bit (decoder->bits);
  if (params->global_motion) {
    int i;

    params->global_only_flag = schro_bits_decode_bit (decoder->bits);
    params->global_prec_bits = schro_bits_decode_uegol (decoder->bits);

    for (i=0;i<params->num_refs;i++) {
      /* pan */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        params->b_1[i] = schro_bits_decode_segol (decoder->bits);
        params->b_2[i] = schro_bits_decode_segol (decoder->bits);
      } else {
        params->b_1[i] = 0;
        params->b_2[i] = 0;
      }

      /* matrix */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        params->a_11[i] = schro_bits_decode_segol (decoder->bits);
        params->a_12[i] = schro_bits_decode_segol (decoder->bits);
        params->a_21[i] = schro_bits_decode_segol (decoder->bits);
        params->a_22[i] = schro_bits_decode_segol (decoder->bits);
      } else {
        params->a_11[i] = 1;
        params->a_12[i] = 0;
        params->a_21[i] = 0;
        params->a_22[i] = 1;
      }

      /* perspective */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        params->c_1[i] = schro_bits_decode_segol (decoder->bits);
        params->c_2[i] = schro_bits_decode_segol (decoder->bits);
      } else {
        params->c_1[i] = 0;
        params->c_2[i] = 0;
      }

      SCHRO_ERROR("ref %d pan %d %d matrix %d %d %d %d perspective %d %d",
          i, params->b_1[i], params->b_2[i],
          params->a_11[i], params->a_12[i],
          params->a_21[i], params->a_22[i],
          params->c_1[i], params->c_2[i]);
    }
  }

  /* block data length */
  length = schro_bits_decode_uegol (decoder->bits);
  SCHRO_ERROR("length %d", length);

  schro_bits_sync (decoder->bits);

  schro_params_calculate_mc_sizes (params);
}

static void
copy_block_4x4 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<4;j++){
    *(uint32_t *)(dest + dstr*j) = *(uint32_t *)(src + sstr*j);
  }
}

static void
copy_block_8x8 (uint8_t *dest, int dstr, uint8_t *src, int sstr)
{
  int j;

  for(j=0;j<8;j++){
    *(uint64_t *)(dest + dstr*j) = *(uint64_t *)(src + sstr*j);
  }
}

void
copy_block (uint8_t *dest, int dstr, uint8_t *src, int sstr, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = src[sstr*j+i];
    }
  }
}

void
splat_block (uint8_t *dest, int dstr, int value, int w, int h)
{
  int i,j;

  for(j=0;j<h;j++){
    for(i=0;i<w;i++) {
      dest[dstr*j+i] = value;
    }
  }
}

void
schro_decoder_decode_prediction_data (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  int i, j;

  for(j=0;j<4*params->y_num_mb;j+=4){
    for(i=0;i<4*params->x_num_mb;i+=4){
      schro_decoder_decode_macroblock(decoder, i, j);
    }
  }
}

static void
schro_decoder_predict (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  SchroFrame *frame = decoder->mc_tmp_frame;
  SchroFrame *reference_frame = decoder->reference_frames[0];
  int i, j;
  int dx, dy;
  int x, y;
  uint8_t *data;
  int stride;
  uint8_t *ref_data;
  int ref_stride;

  for(j=0;j<4*params->y_num_mb;j++){
    for(i=0;i<4*params->x_num_mb;i++){
      SchroMotionVector *mv = &decoder->motion_vectors[j*4*params->x_num_mb + i];

      x = i*params->xbsep_luma;
      y = j*params->ybsep_luma;

      if (mv->pred_mode == 0) {
        data = frame->components[0].data;
        stride = frame->components[0].stride;
        splat_block (data + y * stride + x, stride, mv->dc[0], 8, 8);

        data = frame->components[1].data;
        stride = frame->components[1].stride;
        splat_block (data + y/2 * stride + x/2, stride, mv->dc[1], 4, 4);

        data = frame->components[2].data;
        stride = frame->components[2].stride;
        splat_block (data + y/2 * stride + x/2, stride, mv->dc[2], 4, 4);
      } else {
        dx = mv->x;
        dy = mv->y;

        /* FIXME This is only roughly correct */
        SCHRO_ASSERT(x + dx >= 0);
        //SCHRO_ASSERT(x + dx < params->mc_luma_width - params->xbsep_luma);
        SCHRO_ASSERT(x + dx < params->mc_luma_width);
        SCHRO_ASSERT(y + dy >= 0);
        //SCHRO_ASSERT(y + dy < params->mc_luma_height - params->ybsep_luma);
        SCHRO_ASSERT(y + dy < params->mc_luma_height);

        data = frame->components[0].data;
        stride = frame->components[0].stride;
        ref_data = reference_frame->components[0].data;
        ref_stride = reference_frame->components[0].stride;
        copy_block_8x8 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

        x /= 2;
        dx /= 2;
        y /= 2;
        dy /= 2;

        data = frame->components[1].data;
        stride = frame->components[1].stride;
        ref_data = reference_frame->components[1].data;
        ref_stride = reference_frame->components[1].stride;
        copy_block_4x4 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);

        data = frame->components[2].data;
        stride = frame->components[2].stride;
        ref_data = reference_frame->components[2].data;
        ref_stride = reference_frame->components[2].stride;
        copy_block_4x4 (data + y * stride + x, stride,
            ref_data + (y+dy) * ref_stride + x + dx, ref_stride);
      }
    }
  }
}

static void
schro_decoder_decode_macroblock(SchroDecoder *decoder, int i, int j)
{
  SchroParams *params = &decoder->params;
  SchroMotionVector *mv = &decoder->motion_vectors[j*4*params->x_num_mb + i];
  int k,l;
  int mask;

  //SCHRO_ERROR("global motion %d", params->global_motion);
  if (params->global_motion) {
    mv->mb_using_global = schro_bits_decode_bit (decoder->bits);
  } else {
    mv->mb_using_global = FALSE;
  }
  if (!mv->mb_using_global) {
    mv->mb_split = schro_bits_decode_bits (decoder->bits, 2);
    SCHRO_ASSERT(mv->mb_split != 3);
  } else {
    mv->mb_split = 2;
  }
  if (mv->mb_split != 0) {
    mv->mb_common = schro_bits_decode_bit (decoder->bits);
  } else {
    mv->mb_common = FALSE;
  }
  //SCHRO_ERROR("mb_using_global=%d mb_split=%d mb_common=%d",
  //    mv->mb_using_global, mv->mb_split, mv->mb_common);

  mask = 3 >> mv->mb_split;
  for (k=0;k<4;k++) {
    for (l=0;l<4;l++) {
      SchroMotionVector *bv =
        &decoder->motion_vectors[(j+l)*4*params->x_num_mb + (i+k)];

      if ((k&mask) == 0 && (l&mask) == 0) {
        //SCHRO_ERROR("decoding PU %d %d", j+l, i+k);

        schro_decoder_decode_prediction_unit (decoder, bv);
      } else {
        SchroMotionVector *bv1;
       
        bv1 = &decoder->motion_vectors[(j+(l&(~mask)))*4*params->x_num_mb +
          (i+(k&(~mask)))];

        *bv = *bv1;
      }
    }
  }
}

static void
schro_decoder_decode_prediction_unit(SchroDecoder *decoder,
    SchroMotionVector *mv)
{
  mv->pred_mode = schro_bits_decode_bits (decoder->bits, 2);

  if (mv->pred_mode == 0) {
    mv->dc[0] = schro_bits_decode_uegol (decoder->bits);
    mv->dc[1] = schro_bits_decode_uegol (decoder->bits);
    mv->dc[2] = schro_bits_decode_uegol (decoder->bits);
  } else {
    mv->x = schro_bits_decode_segol (decoder->bits);
    mv->y = schro_bits_decode_segol (decoder->bits);
  }
}

void
schro_decoder_decode_transform_parameters (SchroDecoder *decoder)
{
  int bit;
  SchroParams *params = &decoder->params;

  /* transform */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->wavelet_filter_index = schro_bits_decode_uegol (decoder->bits);
    if (params->wavelet_filter_index > 4) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->wavelet_filter_index = SCHRO_WAVELET_DAUB97;
  }
  SCHRO_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->transform_depth = schro_bits_decode_uegol (decoder->bits);
    if (params->transform_depth > 6) {
      params->non_spec_input = TRUE;
    }
  } else {
    params->transform_depth = 4;
  }
  SCHRO_DEBUG ("transform depth %d", params->transform_depth);

  /* spatial partitioning */
  params->spatial_partition = schro_bits_decode_bit (decoder->bits);
  SCHRO_DEBUG ("spatial_partitioning %d", params->spatial_partition);
  if (params->spatial_partition) {
    params->partition_index = schro_bits_decode_uegol (decoder->bits);
    if (params->partition_index > 1) {
      /* FIXME: ? */
      params->non_spec_input = TRUE;
    }
    if (params->partition_index == 0) {
      params->max_xblocks = schro_bits_decode_uegol (decoder->bits);
      params->max_yblocks = schro_bits_decode_uegol (decoder->bits);
    }
    params->multi_quant = schro_bits_decode_bit (decoder->bits);
  }

  schro_params_calculate_iwt_sizes (params);

  schro_bits_sync(decoder->bits);
}

void
schro_decoder_init_subbands (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
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

  decoder->subbands[0].x = 0;
  decoder->subbands[0].y = 0;
  decoder->subbands[0].w = w;
  decoder->subbands[0].h = h;
  decoder->subbands[0].offset = 0;
  decoder->subbands[0].stride = stride;
  decoder->subbands[0].chroma_w = chroma_w;
  decoder->subbands[0].chroma_h = chroma_h;
  decoder->subbands[0].chroma_offset = 0;
  decoder->subbands[0].chroma_stride = chroma_stride;
  decoder->subbands[0].has_parent = 0;
  decoder->subbands[0].scale_factor_shift = 0;
  decoder->subbands[0].horizontally_oriented = 0;
  decoder->subbands[0].vertically_oriented = 0;

  for(i=0; i<params->transform_depth; i++) {
    decoder->subbands[1+3*i].x = 1;
    decoder->subbands[1+3*i].y = 1;
    decoder->subbands[1+3*i].w = w;
    decoder->subbands[1+3*i].h = h;
    decoder->subbands[1+3*i].offset = w + (stride/2/sizeof(int16_t));
    decoder->subbands[1+3*i].stride = stride;
    decoder->subbands[1+3*i].chroma_w = chroma_w;
    decoder->subbands[1+3*i].chroma_h = chroma_h;
    decoder->subbands[1+3*i].chroma_offset = chroma_w + (chroma_stride/2/sizeof(int16_t));
    decoder->subbands[1+3*i].chroma_stride = chroma_stride;
    decoder->subbands[1+3*i].has_parent = (i>0);
    decoder->subbands[1+3*i].scale_factor_shift = i;
    decoder->subbands[1+3*i].horizontally_oriented = 0;
    decoder->subbands[1+3*i].vertically_oriented = 0;

    decoder->subbands[2+3*i].x = 0;
    decoder->subbands[2+3*i].y = 1;
    decoder->subbands[2+3*i].w = w;
    decoder->subbands[2+3*i].h = h;
    decoder->subbands[2+3*i].offset = (stride/2/sizeof(int16_t));
    decoder->subbands[2+3*i].stride = stride;
    decoder->subbands[2+3*i].chroma_w = chroma_w;
    decoder->subbands[2+3*i].chroma_h = chroma_h;
    decoder->subbands[2+3*i].chroma_offset = (chroma_stride/2/sizeof(int16_t));
    decoder->subbands[2+3*i].chroma_stride = chroma_stride;
    decoder->subbands[2+3*i].has_parent = (i>0);
    decoder->subbands[2+3*i].scale_factor_shift = i;
    decoder->subbands[2+3*i].horizontally_oriented = 0;
    decoder->subbands[2+3*i].vertically_oriented = 1;

    decoder->subbands[3+3*i].x = 1;
    decoder->subbands[3+3*i].y = 0;
    decoder->subbands[3+3*i].w = w;
    decoder->subbands[3+3*i].h = h;
    decoder->subbands[3+3*i].offset = w;
    decoder->subbands[3+3*i].stride = stride;
    decoder->subbands[3+3*i].chroma_w = chroma_w;
    decoder->subbands[3+3*i].chroma_h = chroma_h;
    decoder->subbands[3+3*i].chroma_offset = chroma_w;
    decoder->subbands[3+3*i].chroma_stride = chroma_stride;
    decoder->subbands[3+3*i].has_parent = (i>0);
    decoder->subbands[3+3*i].scale_factor_shift = i;
    decoder->subbands[3+3*i].horizontally_oriented = 1;
    decoder->subbands[3+3*i].vertically_oriented = 0;

    w <<= 1;
    h <<= 1;
    stride >>= 1;
    chroma_w <<= 1;
    chroma_h <<= 1;
    chroma_stride >>= 1;
  }

}

void
schro_decoder_decode_transform_data (SchroDecoder *decoder, int component)
{
  int i;
  SchroParams *params = &decoder->params;

#if 1
  for (i=0;i<3;i++){
    uint8_t value = 0;
    oil_splat_u8_ns (decoder->frame->components[component].data, &value,
        decoder->frame->components[component].length);
  }
#endif

  schro_decoder_init_subbands (decoder);

  for(i=0;i<1+3*params->transform_depth;i++) {
    schro_decoder_decode_subband (decoder, component, i);
  }
}

void
schro_decoder_decode_subband (SchroDecoder *decoder, int component, int index)
{
  SchroParams *params = &decoder->params;
  SchroSubband *subband = decoder->subbands + index;
  SchroSubband *parent_subband = NULL;
  int subband_zero_flag;
  int16_t *data;
  int16_t *parent_data = NULL;
  int quant_index = 0;
  int quant_factor = 0;
  int quant_offset = 0;
  int subband_length;
  int scale_factor;
  int ntop;
  int height;
  int width;
  int stride;
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

  SCHRO_DEBUG("subband index=%d %d x %d at offset %d stride=%d",
      index, width, height, offset, stride);

  data = (int16_t *)decoder->frame->components[component].data + offset;
  if (subband->has_parent) {
    parent_subband = subband-3;
    if (component == 0) {
      parent_data = (int16_t *)decoder->frame->components[component].data +
        parent_subband->offset;
    } else {
      parent_data = (int16_t *)decoder->frame->components[component].data +
        parent_subband->chroma_offset;
    }
  }

  scale_factor = 1<<(params->transform_depth - quant_index);
  ntop = (scale_factor>>1) * quant_factor;

  subband_zero_flag = schro_bits_decode_bit (decoder->bits);
  if (!subband_zero_flag) {
    int i,j;
    SchroBuffer *buffer;
    SchroBits *bits;
    SchroArith *arith;

    quant_index = schro_bits_decode_uegol (decoder->bits);
    SCHRO_DEBUG("quant index %d", quant_index);
    if ((unsigned int)quant_index > 60) {
      SCHRO_ERROR("quant_index too big (%u > 60)", quant_index);
      params->non_spec_input = TRUE;
      return;
    }
    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset[quant_index];
    SCHRO_DEBUG("quant factor %d offset %d", quant_factor, quant_offset);

    subband_length = schro_bits_decode_uegol (decoder->bits);
    SCHRO_DEBUG("subband length %d", subband_length);

    schro_bits_sync (decoder->bits);

    buffer = schro_buffer_new_subbuffer (decoder->bits->buffer,
        decoder->bits->offset>>3, subband_length);
    bits = schro_bits_new ();
    schro_bits_decode_init (bits, buffer);

    arith = schro_arith_new ();
    schro_arith_decode_init (arith, bits);
    schro_arith_init_contexts (arith);

    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int sign = 0;
        int v;
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
            context = SCHRO_CTX_Z_BIN1z;
          } else {
            context = SCHRO_CTX_Z_BIN1nz;
          }
          context2 = SCHRO_CTX_Z_BIN2;
        } else {
          if (nhood_sum == 0) {
            context = SCHRO_CTX_NZ_BIN1z;
          } else {
            if (nhood_sum <= ntop) {
              context = SCHRO_CTX_NZ_BIN1a;
            } else {
              context = SCHRO_CTX_NZ_BIN1b;
            }
          }
          context2 = SCHRO_CTX_NZ_BIN2;
        }

        if (previous_value > 0) {
          sign_context = SCHRO_CTX_SIGN_POS;
        } else if (previous_value < 0) {
          sign_context = SCHRO_CTX_SIGN_NEG;
        } else {
          sign_context = SCHRO_CTX_SIGN_ZERO;
        }

        v = schro_arith_context_decode_uu (arith, context, context2);
        if (v) {
          sign = schro_arith_context_decode_bit (arith, sign_context);
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
    schro_arith_free (arith);
    SCHRO_DEBUG("decoded %d bits", bits->offset);
    schro_bits_free (bits);
    schro_buffer_unref (buffer);

    decoder->bits->offset += subband_length * 8;
  } else {
    int i,j;

    SCHRO_DEBUG("subband is zero");
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        data[j*stride + i] = 0;
      }
    }

    schro_bits_sync (decoder->bits);
  }
}



