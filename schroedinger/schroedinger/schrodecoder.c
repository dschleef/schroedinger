
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


//#define DECODE_PREDICTION_ONLY

static void schro_decoder_decode_macroblock(SchroDecoder *decoder,
    SchroArith *arith, int i, int j);
static void schro_decoder_decode_prediction_unit(SchroDecoder *decoder,
    SchroArith *arith, SchroMotionVector *motion_vectors, int x, int y);

static void schro_decoder_reference_add (SchroDecoder *decoder, SchroFrame *frame);
static SchroFrame * schro_decoder_reference_get (SchroDecoder *decoder, int frame_number);
static void schro_decoder_reference_retire (SchroDecoder *decoder, int frame_number);


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
  int i;

  if (decoder->frame) {
    schro_frame_free (decoder->frame);
  }
  for(i=0;i<decoder->n_output_frames;i++) {
    schro_frame_free (decoder->output_frames[i]);
  }
  for(i=0;i<decoder->n_reference_frames;i++) {
    schro_frame_free (decoder->reference_frames[i]);
  }
  for(i=0;i<decoder->frame_queue_length;i++){
    schro_frame_free (decoder->frame_queue[i]);
  }

  free (decoder->tmpbuf);
  free (decoder->tmpbuf2);
  free (decoder);
}

void
schro_decoder_add_output_frame (SchroDecoder *decoder, SchroFrame *frame)
{
  decoder->output_frames[decoder->n_output_frames] = frame;
  decoder->n_output_frames++;
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
schro_decoder_is_access_unit (SchroBuffer *buffer)
{
  uint8_t *data;

  if (buffer->length < 5) return 0;

  data = buffer->data;
  if (data[0] != 'B' || data[1] != 'B' || data[2] != 'C' || data[3] != 'D') {
    return 0;
  }

  if (data[4] == SCHRO_PARSE_CODE_ACCESS_UNIT) return 1;

  return 0;
}


SchroFrame *
schro_decoder_decode (SchroDecoder *decoder, SchroBuffer *buffer)
{
  SchroParams *params = &decoder->params;
  int i;
  SchroFrame *output_frame;
  
  if (buffer == NULL) {
    SCHRO_DEBUG("searching for frame %d", decoder->next_frame_number);
    for(i=0;i<decoder->frame_queue_length;i++) {
      if (decoder->frame_queue[i]->frame_number == decoder->next_frame_number) {
        SchroFrame *frame = decoder->frame_queue[i];

        for(;i<decoder->frame_queue_length-1;i++){
          decoder->frame_queue[i] = decoder->frame_queue[i+1];
        }
        decoder->frame_queue_length--;
        decoder->next_frame_number++;

        return frame;
      }
    }

    return NULL;
  }

  decoder->bits = schro_bits_new ();
  schro_bits_decode_init (decoder->bits, buffer);

  if (schro_decoder_is_access_unit (buffer)) {
    schro_decoder_decode_parse_header(decoder);
    schro_decoder_decode_access_unit(decoder);

    schro_buffer_unref (buffer);
    schro_bits_free (decoder->bits);
    return NULL;
  }

  schro_decoder_decode_parse_header(decoder);
  schro_decoder_decode_frame_header(decoder);

  output_frame = decoder->output_frames[decoder->n_output_frames-1];
  decoder->n_output_frames--;

  decoder->params.num_refs = SCHRO_PARSE_CODE_NUM_REFS(decoder->code);

  if (SCHRO_PARSE_CODE_NUM_REFS(decoder->code) == 0) {
    SCHRO_DEBUG("intra");
    schro_decoder_decode_transform_parameters (decoder);

    if (decoder->frame == NULL) {
      decoder->frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
          params->iwt_luma_width, params->iwt_luma_height, 2, 2);
    }

    schro_decoder_decode_transform_data (decoder, 0);
    schro_decoder_decode_transform_data (decoder, 1);
    schro_decoder_decode_transform_data (decoder, 2);

    schro_frame_inverse_iwt_transform (decoder->frame, &decoder->params,
        decoder->tmpbuf);

    schro_frame_shift_right (decoder->frame, 4);
    schro_frame_convert (output_frame, decoder->frame);

    output_frame->frame_number = decoder->picture_number;

    if (SCHRO_PARSE_CODE_IS_REF(decoder->code)) {
      SchroFrame *ref;

      ref = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
          params->width, params->height, 2, 2);
      schro_frame_convert (ref, decoder->frame);
      ref->frame_number = decoder->picture_number;
      schro_decoder_reference_add (decoder, ref);
    }
  } else {
    SCHRO_DEBUG("inter");

    schro_decoder_decode_frame_prediction (decoder);

    /* FIXME */
    SCHRO_ASSERT(params->xbsep_luma == 8);
    SCHRO_ASSERT(params->ybsep_luma == 8);

    if (decoder->mc_tmp_frame == NULL) {
      decoder->mc_tmp_frame = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16,
          params->mc_luma_width, params->mc_luma_height, 2, 2);
    }

    if (decoder->motion_vectors == NULL) {
      decoder->motion_vectors = malloc(sizeof(SchroMotionVector) *
          params->x_num_blocks * params->y_num_blocks);
    }

    decoder->ref0 = schro_decoder_reference_get (decoder, decoder->reference1);
    SCHRO_ASSERT (decoder->ref0 != NULL);

    if (decoder->n_refs > 1) {
      decoder->ref1 = schro_decoder_reference_get (decoder, decoder->reference2);
      SCHRO_ASSERT (decoder->ref1 != NULL);
    }

    schro_decoder_decode_prediction_data (decoder);
    schro_frame_copy_with_motion (decoder->mc_tmp_frame,
        decoder->ref0, decoder->ref1,
        decoder->motion_vectors, &decoder->params);

    schro_decoder_decode_transform_parameters (decoder);

    schro_decoder_decode_transform_data (decoder, 0);
    schro_decoder_decode_transform_data (decoder, 1);
    schro_decoder_decode_transform_data (decoder, 2);

    schro_frame_inverse_iwt_transform (decoder->frame, &decoder->params,
        decoder->tmpbuf);

    schro_frame_shift_right (decoder->frame, 4);

#ifndef DECODE_PREDICTION_ONLY
    schro_frame_add (decoder->frame, decoder->mc_tmp_frame);

    schro_frame_convert (output_frame, decoder->frame);
#else
    schro_frame_convert (output_frame, decoder->mc_tmp_frame);
#endif
    output_frame->frame_number = decoder->picture_number;

    if (SCHRO_PARSE_CODE_IS_REF(decoder->code)) {
      SchroFrame *ref;

      ref = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8,
          params->width, params->height, 2, 2);
      schro_frame_convert (ref, decoder->frame);
      ref->frame_number = decoder->picture_number;
      schro_decoder_reference_add (decoder, ref);
    }
  }

  for(i=0;i<decoder->n_retire;i++){
    schro_decoder_reference_retire (decoder, decoder->retire_list[i]);
  }

  schro_buffer_unref (buffer);
  schro_bits_free (decoder->bits);

  if (output_frame->frame_number == decoder->next_frame_number) {
    decoder->next_frame_number++;
    return output_frame;
  } else {
    SCHRO_DEBUG("adding %d to queue", output_frame->frame_number);
    decoder->frame_queue[decoder->frame_queue_length] = output_frame;
    decoder->frame_queue_length++;
  }

  return NULL;
}

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

  decoder->n_refs = SCHRO_PARSE_CODE_NUM_REFS(decoder->code);
  SCHRO_DEBUG("n_refs %d", decoder->n_refs);

  decoder->next_parse_offset = schro_bits_decode_bits (decoder->bits, 24);
  SCHRO_DEBUG ("next_parse_offset %d", decoder->next_parse_offset);
  decoder->prev_parse_offset = schro_bits_decode_bits (decoder->bits, 24);
  SCHRO_DEBUG ("prev_parse_offset %d", decoder->prev_parse_offset);
}

void
schro_decoder_decode_access_unit (SchroDecoder *decoder)
{
  int bit;
  int video_format;

  SCHRO_DEBUG("decoding access unit");
  /* parse parameters */
  decoder->au_frame_number = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG("au frame number = %d", decoder->au_frame_number);
  decoder->params.major_version = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("major_version = %d", decoder->params.major_version);
  decoder->params.minor_version = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("minor_version = %d", decoder->params.minor_version);
  decoder->params.profile = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("profile = %d", decoder->params.profile);
  decoder->params.level = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("level = %d", decoder->params.level);

  /* sequence parameters */
  video_format = schro_bits_decode_uint (decoder->bits);
  schro_params_set_video_format (&decoder->params, video_format);

  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.width = schro_bits_decode_uint (decoder->bits);
    decoder->params.height = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("size = %d x %d", decoder->params.width,
      decoder->params.height);

  /* chroma format */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.chroma_format_index =
      schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("chroma_format_index %d",
      decoder->params.chroma_format_index);
  decoder->params.chroma_width = (decoder->params.width + 1)/2;
  decoder->params.chroma_height = (decoder->params.height + 1)/2;

  /* video depth */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.video_depth = schro_bits_decode_uint (decoder->bits);
  }

  /* source parameters */
  /* scan format */
  decoder->params.interlaced_source = schro_bits_decode_bit (decoder->bits);
  if (decoder->params.interlaced_source) {
    bit = schro_bits_decode_bit (decoder->bits);
    if (bit) {
      decoder->params.top_field_first = schro_bits_decode_bit (decoder->bits);
    }
    bit = schro_bits_decode_bit (decoder->bits);
    if (bit) {
      decoder->params.sequential_fields = schro_bits_decode_bit (decoder->bits);
    }
  }
  SCHRO_DEBUG("interlace %d top_field_first %d sequential_fields %d",
      decoder->params.interlaced_source, decoder->params.top_field_first,
      decoder->params.sequential_fields);

  /* frame rate */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      decoder->params.frame_rate_numerator = schro_bits_decode_uint (decoder->bits);
      decoder->params.frame_rate_denominator = schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_frame_rate (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("frame rate %d/%d", decoder->params.frame_rate_numerator,
      decoder->params.frame_rate_denominator);

  /* aspect ratio */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      decoder->params.aspect_ratio_numerator =
        schro_bits_decode_uint (decoder->bits);
      decoder->params.aspect_ratio_denominator =
        schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_aspect_ratio (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("aspect ratio %d/%d",
      decoder->params.aspect_ratio_numerator,
      decoder->params.aspect_ratio_denominator);

  /* clean area */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    decoder->params.clean_width = schro_bits_decode_uint (decoder->bits);
    decoder->params.clean_height = schro_bits_decode_uint (decoder->bits);
    decoder->params.left_offset = schro_bits_decode_uint (decoder->bits);
    decoder->params.top_offset = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("clean offset %d %d", decoder->params.left_offset,
      decoder->params.top_offset);
  SCHRO_DEBUG("clean size %d %d", decoder->params.clean_width,
      decoder->params.clean_height);

  /* signal range */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      decoder->params.luma_offset = schro_bits_decode_uint (decoder->bits);
      decoder->params.luma_excursion = schro_bits_decode_uint (decoder->bits);
      decoder->params.chroma_offset = schro_bits_decode_uint (decoder->bits);
      decoder->params.chroma_excursion =
        schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_signal_range (&decoder->params, index);
    }
  }
  SCHRO_DEBUG("luma offset %d excursion %d",
      decoder->params.luma_offset, decoder->params.luma_excursion);
  SCHRO_DEBUG("chroma offset %d excursion %d",
      decoder->params.chroma_offset, decoder->params.chroma_excursion);

  /* colour spec */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int index;
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      /* colour primaries */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        decoder->params.colour_primaries =
          schro_bits_decode_uint (decoder->bits);
      }
      /* colour matrix */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        decoder->params.colour_matrix =
          schro_bits_decode_uint (decoder->bits);
      }
      /* transfer function */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        decoder->params.transfer_function =
          schro_bits_decode_uint (decoder->bits);
      }
    } else {
      schro_params_set_colour_spec (&decoder->params, index);
    }
  }
}

void
schro_decoder_decode_frame_header (SchroDecoder *decoder)
{
  int i;

  schro_bits_sync(decoder->bits);

  decoder->picture_number = schro_bits_decode_bits (decoder->bits, 32);
  SCHRO_DEBUG("picture number %d", decoder->picture_number);

  if (decoder->n_refs > 0) {
    decoder->reference1 = decoder->picture_number +
      schro_bits_decode_sint (decoder->bits);
    SCHRO_DEBUG("ref1 %d", decoder->reference1);
  }

  if (decoder->n_refs > 1) {
    decoder->reference2 = decoder->picture_number +
      schro_bits_decode_sint (decoder->bits);
    SCHRO_DEBUG("ref2 %d", decoder->reference2);
  }

  decoder->n_retire = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("n_retire %d", decoder->n_retire);
  for(i=0;i<decoder->n_retire;i++){
    int offset;
    offset = schro_bits_decode_sint (decoder->bits);
    decoder->retire_list[i] = decoder->picture_number + offset;
    SCHRO_DEBUG("retire %d", decoder->picture_number + offset);
  }
}

void
schro_decoder_decode_frame_prediction (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  int bit;
  int index;

  schro_bits_sync (decoder->bits);

  /* block params flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    index = schro_bits_decode_uint (decoder->bits);
    if (index == 0) {
      params->xblen_luma = schro_bits_decode_uint (decoder->bits);
      params->yblen_luma = schro_bits_decode_uint (decoder->bits);
      params->xbsep_luma = schro_bits_decode_uint (decoder->bits);
      params->ybsep_luma = schro_bits_decode_uint (decoder->bits);
    } else {
      schro_params_set_block_params (params, index);
    }
  }
  SCHRO_DEBUG("blen_luma %d %d bsep_luma %d %d",
      params->xblen_luma, params->yblen_luma,
      params->xbsep_luma, params->ybsep_luma);

  /* mv precision flag */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->mv_precision = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG("mv_precision %d", params->mv_precision);

  /* global motion flag */
  params->global_motion = schro_bits_decode_bit (decoder->bits);
  if (params->global_motion) {
    int i;

    params->global_only_flag = schro_bits_decode_bit (decoder->bits);

    for (i=0;i<params->num_refs;i++) {
      /* pan/tilt */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        params->b_1[i] = schro_bits_decode_sint (decoder->bits);
        params->b_2[i] = schro_bits_decode_sint (decoder->bits);
      } else {
        params->b_1[i] = 0;
        params->b_2[i] = 0;
      }

      /* matrix */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        int exponent;

        /* FIXME */
        exponent = schro_bits_decode_uint (decoder->bits);
        params->a_11[i] = schro_bits_decode_sint (decoder->bits);
        params->a_12[i] = schro_bits_decode_sint (decoder->bits);
        params->a_21[i] = schro_bits_decode_sint (decoder->bits);
        params->a_22[i] = schro_bits_decode_sint (decoder->bits);
      } else {
        params->a_11[i] = 1;
        params->a_12[i] = 0;
        params->a_21[i] = 0;
        params->a_22[i] = 1;
      }

      /* perspective */
      bit = schro_bits_decode_bit (decoder->bits);
      if (bit) {
        int exponent;

        /* FIXME */
        exponent = schro_bits_decode_uint (decoder->bits);
        params->c_1[i] = schro_bits_decode_sint (decoder->bits);
        params->c_2[i] = schro_bits_decode_sint (decoder->bits);
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

  /* picture prediction mode */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->picture_pred_mode = schro_bits_decode_uint (decoder->bits);
  }

  /* non-default picture weights */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    int precision;

    /* FIXME */
    precision = schro_bits_decode_uint (decoder->bits);
    if (params->num_refs > 0) {
      params->picture_weight_1 = schro_bits_decode_uint (decoder->bits);
    }
    if (params->num_refs > 1) {
      params->picture_weight_2 = schro_bits_decode_uint (decoder->bits);
    }
  }

  schro_params_calculate_mc_sizes (params);
}

void
schro_decoder_decode_prediction_data (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  SchroArith *arith;
  int i, j;
  SchroBuffer *buffer;
  int length;

  schro_bits_sync (decoder->bits);

  length = schro_bits_decode_uint (decoder->bits);

  schro_bits_sync (decoder->bits);

  buffer = schro_buffer_new_subbuffer (decoder->bits->buffer,
      decoder->bits->offset>>3, length);

  arith = schro_arith_new ();
  schro_arith_decode_init (arith, buffer);
  schro_arith_init_contexts (arith);

  for(j=0;j<4*params->y_num_mb;j+=4){
    for(i=0;i<4*params->x_num_mb;i+=4){
      schro_decoder_decode_macroblock(decoder, arith, i, j);
    }
  }

  schro_arith_free (arith);

  decoder->bits->offset += length<<3;
}

static void
schro_decoder_decode_macroblock(SchroDecoder *decoder, SchroArith *arith,
    int i, int j)
{
  SchroParams *params = &decoder->params;
  SchroMotionVector *mv = &decoder->motion_vectors[j*4*params->x_num_mb + i];
  int k,l;
  int split_prediction;

  split_prediction = schro_motion_split_prediction (decoder->motion_vectors,
      params, i, j);
  mv->split = (split_prediction + schro_arith_decode_mode (arith,
        SCHRO_CTX_SPLIT_0, SCHRO_CTX_SPLIT_1))%3;

  if (params->global_motion) {
    mv->using_global = schro_arith_context_decode_bit (arith,
        SCHRO_CTX_GLOBAL_BLOCK);
  } else {
    mv->using_global = FALSE;
  }
#if 0
  if (!mv->using_global) {
    mv->split = schro_bits_decode_bits (decoder->bits, 2);
    SCHRO_ASSERT(mv->split != 3);
  } else {
    mv->split = 2;
  }
#endif
  if (mv->split != 0) {
    mv->common = schro_arith_context_decode_bit (arith, SCHRO_CTX_COMMON);
  } else {
    mv->common = FALSE;
  }
  //SCHRO_ERROR("using_global=%d split=%d common=%d",
  //    mv->using_global, mv->split, mv->common);

  switch (mv->split) {
    case 0:
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_vectors, i, j);
      mv[1] = mv[0];
      mv[2] = mv[0];
      mv[3] = mv[0];
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 2*params->x_num_blocks, mv, 4*sizeof(*mv));
      memcpy(mv + 3*params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 1:
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_vectors, i, j);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_vectors, i + 2, j);
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));

      mv += 2*params->x_num_blocks;
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_vectors, i, j + 2);
      mv[1] = mv[0];
      schro_decoder_decode_prediction_unit (decoder, arith,
          decoder->motion_vectors, i + 2, j + 2);
      memcpy(mv + params->x_num_blocks, mv, 4*sizeof(*mv));
      break;
    case 2:
      for (l=0;l<4;l++) {
        for (k=0;k<4;k++) {
          schro_decoder_decode_prediction_unit (decoder, arith,
              decoder->motion_vectors, i + k, j + l);
        }
      }
      break;
    default:
      SCHRO_ASSERT(0);
  }
}

static void
schro_decoder_decode_prediction_unit(SchroDecoder *decoder, SchroArith *arith,
    SchroMotionVector *motion_vectors, int x, int y)
{
  SchroParams *params = &decoder->params;
  SchroMotionVector *mv = &motion_vectors[y*4*params->x_num_mb + x];

  mv->pred_mode = 
    schro_arith_context_decode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF1) |
    (schro_arith_context_decode_bit (arith, SCHRO_CTX_BLOCK_MODE_REF2) << 1);

  if (mv->pred_mode == 0) {
    int pred[3];

    schro_motion_dc_prediction (motion_vectors, &decoder->params, x, y, pred);

    mv->dc[0] = pred[0] + schro_arith_context_decode_sint (arith,
        SCHRO_CTX_LUMA_DC_CONT_BIN1, SCHRO_CTX_LUMA_DC_VALUE,
        SCHRO_CTX_LUMA_DC_SIGN);
    mv->dc[1] = pred[1] + schro_arith_context_decode_sint (arith,
        SCHRO_CTX_CHROMA1_DC_CONT_BIN1, SCHRO_CTX_CHROMA1_DC_VALUE,
        SCHRO_CTX_CHROMA1_DC_SIGN);
    mv->dc[2] = pred[2] + schro_arith_context_decode_sint (arith,
        SCHRO_CTX_CHROMA2_DC_CONT_BIN1, SCHRO_CTX_CHROMA2_DC_VALUE,
        SCHRO_CTX_CHROMA2_DC_SIGN);
  } else {
    int pred_x, pred_y;

    schro_motion_vector_prediction (motion_vectors, &decoder->params, x, y,
        &pred_x, &pred_y);

    mv->x = pred_x + schro_arith_context_decode_sint (arith,
         SCHRO_CTX_MV_REF1_H_CONT_BIN1, SCHRO_CTX_MV_REF1_H_VALUE,
         SCHRO_CTX_MV_REF1_H_SIGN);
    mv->y = pred_y + schro_arith_context_decode_sint (arith,
         SCHRO_CTX_MV_REF1_V_CONT_BIN1, SCHRO_CTX_MV_REF1_V_VALUE,
         SCHRO_CTX_MV_REF1_V_SIGN);
  }
}

void
schro_decoder_decode_transform_parameters (SchroDecoder *decoder)
{
  int bit;
  SchroParams *params = &decoder->params;

  if (params->num_refs > 0) {
    bit = schro_bits_decode_bit (decoder->bits);

    SCHRO_DEBUG ("zero residual %d", bit);
    /* FIXME */
    SCHRO_ASSERT(bit == 0);
  }

  params->wavelet_filter_index = SCHRO_WAVELET_APPROX97;
  params->transform_depth = 4;

  /* transform */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->wavelet_filter_index = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG ("wavelet filter index %d", params->wavelet_filter_index);

  /* transform depth */
  bit = schro_bits_decode_bit (decoder->bits);
  if (bit) {
    params->transform_depth = schro_bits_decode_uint (decoder->bits);
  }
  SCHRO_DEBUG ("transform depth %d", params->transform_depth);

  /* spatial partitioning */
  params->spatial_partition_flag = schro_bits_decode_bit (decoder->bits);
  SCHRO_DEBUG ("spatial_partitioning %d", params->spatial_partition_flag);
  if (params->spatial_partition_flag) {
    params->nondefault_partition_flag = schro_bits_decode_bit (decoder->bits);
    if (params->nondefault_partition_flag) {
      int i;
      for(i=0;i<params->transform_depth + 1;i++) {
        params->codeblock_width[i] = schro_bits_decode_uint (decoder->bits);
        params->codeblock_height[i] = schro_bits_decode_uint (decoder->bits);
      }
    }
    params->codeblock_mode_index = schro_bits_decode_uint (decoder->bits);
  }

  schro_params_calculate_iwt_sizes (params);
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

  schro_decoder_init_subbands (decoder);

  for(i=0;i<1+3*params->transform_depth;i++) {
    schro_decoder_decode_subband (decoder, component, i);
  }
}

static int table[32][3] = {
  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_2 },

  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_0 },

  { SCHRO_CTX_Z_BIN1_0, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },
  { 0, 0, 0 },
  { SCHRO_CTX_Z_BIN1_1, SCHRO_CTX_Z_VALUE, SCHRO_CTX_Z_SIGN_1 },

  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_2 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_0 },

  { SCHRO_CTX_NZ_BIN1_0, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },
  { SCHRO_CTX_NZ_BIN1_1, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },
  { 0, 0, 0 },
  { SCHRO_CTX_NZ_BIN1_2, SCHRO_CTX_NZ_VALUE, SCHRO_CTX_NZ_SIGN_1 },

  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
  { 0, 0, 0 },
};

void
schro_decoder_decode_subband (SchroDecoder *decoder, int component, int index)
{
  SchroParams *params = &decoder->params;
  SchroSubband *subband = decoder->subbands + index;
  SchroSubband *parent_subband = NULL;
  int16_t *data;
  int16_t *parent_data = NULL;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int subband_length;
  int scale_factor;
  int ntop;
  int height;
  int width;
  int stride;
  int offset;
  int x,y;

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

  schro_bits_sync (decoder->bits);

  subband_length = schro_bits_decode_uint (decoder->bits);
  SCHRO_DEBUG("subband length %d", subband_length);

  if (subband_length > 0) {
    int i,j;
    SchroBuffer *buffer;
    SchroArith *arith;

    quant_index = schro_bits_decode_uint (decoder->bits);
    SCHRO_DEBUG("quant index %d", quant_index);
    if ((unsigned int)quant_index > 60) {
      SCHRO_ERROR("quant_index too big (%u > 60)", quant_index);
      params->non_spec_input = TRUE;
      return;
    }
    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset[quant_index];
    SCHRO_DEBUG("quant factor %d offset %d", quant_factor, quant_offset);

    scale_factor = 1<<(params->transform_depth - subband->scale_factor_shift);
    ntop = (scale_factor>>1) * quant_factor;

    schro_bits_sync (decoder->bits);

    buffer = schro_buffer_new_subbuffer (decoder->bits->buffer,
        decoder->bits->offset>>3, subband_length);

    arith = schro_arith_new ();
    schro_arith_decode_init (arith, buffer);
    schro_arith_init_contexts (arith);

#define MIN(a,b) ((a)<(b) ? (a) : (b))
    for(y=0;y<height;y+=params->codeblock_height[subband->scale_factor_shift]) {
      int ymax = MIN(y + params->codeblock_height[subband->scale_factor_shift],
          height);

      for(x=0;x<width;x+=params->codeblock_width[subband->scale_factor_shift]) {
        int xmax = MIN(x + params->codeblock_width[subband->scale_factor_shift],
            width);
        int zero_codeblock;

        zero_codeblock = schro_arith_context_decode_bit (arith, SCHRO_CTX_ZERO_CODEBLOCK);
        if (zero_codeblock) {
          for(j=y;j<ymax;j++){
            for(i=x;i<xmax;i++){
              data[j*stride + i] = 0;
            }
          }
          continue;
        }

    for(j=y;j<ymax;j++){
      for(i=x;i<xmax;i++){
        int v;
        int parent_zero;
        int cont_context;
        int nhood_sum;
        int previous_value;
        int sign_context;
        int value_context;
        int pred_value;
        int table_index;
        int16_t *p = data + j*stride + i;

        nhood_sum = 0;
        if (j>0) {
          nhood_sum += abs(p[-stride]);
        }
        if (i>0) {
          nhood_sum += abs(p[-1]);
        }
        if (i>0 && j>0) {
          nhood_sum += abs(p[-stride-1]);
        }
//nhood_sum = 0;
        
        if (index == 0) {
          if (j>0) {
            if (i>0) {
              pred_value = (p[-1] + p[-stride] + p[-stride-1] + 1)/3;
            } else {
              pred_value = p[-stride];
            }
          } else {
            if (i>0) {
              pred_value = p[-1];
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
            previous_value = p[-1];
          }
        } else if (subband->vertically_oriented) {
          if (j > 0) {
            previous_value = p[-stride];
          }
        }
//previous_value = 0;

        table_index = (parent_zero == 0)<<4;
        table_index |= (previous_value < 0)<<3;
        table_index |= (previous_value > 0)<<2;
        table_index |= (nhood_sum > ntop)<<1;
        table_index |= (nhood_sum > 0)<<0;

#if 0
        if (parent_zero) {
          if (nhood_sum == 0) {
            cont_context = SCHRO_CTX_Z_BIN1_0;
          } else {
            cont_context = SCHRO_CTX_Z_BIN1_1;
          }
          value_context = SCHRO_CTX_Z_VALUE;
          if (previous_value > 0) {
            sign_context = SCHRO_CTX_Z_SIGN_0;
          } else if (previous_value < 0) {
            sign_context = SCHRO_CTX_Z_SIGN_1;
          } else {
            sign_context = SCHRO_CTX_Z_SIGN_2;
          }
        } else {
          if (nhood_sum == 0) {
            cont_context = SCHRO_CTX_NZ_BIN1_0;
          } else {
            if (nhood_sum <= ntop) {
              cont_context = SCHRO_CTX_NZ_BIN1_1;
            } else {
              cont_context = SCHRO_CTX_NZ_BIN1_2;
            }
          }
          value_context = SCHRO_CTX_NZ_VALUE;
          if (previous_value > 0) {
            sign_context = SCHRO_CTX_NZ_SIGN_0;
          } else if (previous_value < 0) {
            sign_context = SCHRO_CTX_NZ_SIGN_1;
          } else {
            sign_context = SCHRO_CTX_NZ_SIGN_2;
          }
        }
        if (cont_context != table[table_index][0] ||
            value_context != table[table_index][1] ||
            sign_context != table[table_index][2]) { 
          SCHRO_ERROR("c,v,s %d %d %d : %d %d %d [%d]",
              cont_context, value_context, sign_context,
              table[table_index][0], table[table_index][1],
              table[table_index][2], table_index);
        }
#endif
        cont_context = table[table_index][0];
        value_context = table[table_index][1];
        sign_context = table[table_index][2];

        v = schro_arith_context_decode_sint (arith, cont_context,
            value_context, sign_context);
        if (v) {
          if (v>0) {
            data[j*stride + i] = pred_value + quant_offset + quant_factor * v;
          } else {
            data[j*stride + i] = pred_value - quant_offset + quant_factor * v;
          }
        } else {
          data[j*stride + i] = pred_value;
        }
      }
    }
      }
    }
    schro_arith_free (arith);
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
  }
}



/* reference pool */

static void
schro_decoder_reference_add (SchroDecoder *decoder, SchroFrame *frame)
{
  SCHRO_DEBUG("adding %d", frame->frame_number);
  decoder->reference_frames[decoder->n_reference_frames] = frame;
  decoder->n_reference_frames++;
  SCHRO_ASSERT(decoder->n_reference_frames < 10);
}

static SchroFrame *
schro_decoder_reference_get (SchroDecoder *decoder, int frame_number)
{
  int i;
  SCHRO_DEBUG("getting %d", frame_number);
  for(i=0;i<decoder->n_reference_frames;i++){
    if (decoder->reference_frames[i]->frame_number == frame_number) {
      return decoder->reference_frames[i];
    }
  }
  return NULL;

}

static void
schro_decoder_reference_retire (SchroDecoder *decoder, int frame_number)
{
  int i;
  SCHRO_DEBUG("retiring %d", frame_number);
  for(i=0;i<decoder->n_reference_frames;i++){
    if (decoder->reference_frames[i]->frame_number == frame_number) {
      schro_frame_free (decoder->reference_frames[i]);
      memmove (decoder->reference_frames + i, decoder->reference_frames + i + 1,
          sizeof(SchroFrame *)*(decoder->n_reference_frames - i - 1));
      decoder->n_reference_frames--;
      return;
    }
  }
}


