
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>
#include <schroedinger/schroorc.h>
#include <string.h>


/* When defined, this trims -1's and 1's from the end of slices by
 * converting them to 0's.  (Zeros get trimmed by default.)  It
 * doesn't seem to affect psnr any. (limited testing) */
//#define USE_TRAILING_DEAD_ZONE 1


typedef struct _SchroLowDelay SchroLowDelay;
typedef struct _SchroLDSubband SchroLDSubband;

struct _SchroLDSubband
{
  int16_t *data;
  int x_stride;
  int y_stride;

  int slice_y_width;
  int slice_y_height;
  int slice_uv_width;
  int slice_uv_height;
};

struct _SchroLowDelay
{
  SchroFrame *frame;

  SchroParams *params;

  int n_vert_slices;
  int n_horiz_slices;

  SchroFrameData luma_subbands[SCHRO_LIMIT_SUBBANDS];
  SchroFrameData chroma1_subbands[SCHRO_LIMIT_SUBBANDS];
  SchroFrameData chroma2_subbands[SCHRO_LIMIT_SUBBANDS];

  int slice_y_size;
  int slice_uv_size;

  int16_t *saved_dc_values;
  int16_t *quant_data;
  int16_t *u_data;
  int16_t *v_data;

  SchroLDSubband subbands[SCHRO_LIMIT_SUBBANDS];

  int slice_y_width;
  int slice_y_height;
  int slice_uv_width;
  int slice_uv_height;
};


#if 0
void
schro_encoder_init_subbands (SchroEncoderFrame * frame)
{
  int i;
  int pos;
  SchroParams *params = &frame->params;

  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    pos = schro_subband_get_position (i);

    schro_subband_get_frame_data (frame->luma_subbands + i,
        frame->iwt_frame, 0, pos, params);
    schro_subband_get_frame_data (frame->chroma1_subbands + i,
        frame->iwt_frame, 0, pos, params);
    schro_subband_get_frame_data (frame->chroma2_subbands + i,
        frame->iwt_frame, 0, pos, params);
  }
}
#endif


static int
ilog2up (unsigned int x)
{
  int i;

  for (i = 0; i < 32; i++) {
    if (x == 0)
      return i;
    x >>= 1;
  }
  return 0;
}


static void
schro_decoder_decode_slice_slow (SchroPicture * picture,
    SchroLowDelay * lowdelay,
    int slice_x, int slice_y, int offset, int slice_bytes)
{
  SchroParams *params = &picture->params;
  SchroUnpack y_unpack;
  SchroUnpack uv_unpack;
  int quant_index;
  int base_index;
  int length_bits;
  int slice_y_length;
  int i;
  int j;
  int x, y;
  int value;

  schro_unpack_init_with_data (&y_unpack,
      OFFSET (picture->lowdelay_buffer->data, offset), slice_bytes, 1);

  base_index = schro_unpack_decode_bits (&y_unpack, 7);
  length_bits = ilog2up (8 * slice_bytes);

  slice_y_length = schro_unpack_decode_bits (&y_unpack, length_bits);

  schro_unpack_copy (&uv_unpack, &y_unpack);
  schro_unpack_limit_bits_remaining (&y_unpack, slice_y_length);
  schro_unpack_skip_bits (&uv_unpack, slice_y_length);

  j = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line;
    SchroFrameData block;

    schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset_1_2[quant_index];

    for (y = 0; y < block.height; y++) {
      line = SCHRO_FRAME_DATA_GET_LINE (&block, y);
      for (x = 0; x < block.width; x++) {
        value = schro_unpack_decode_sint (&y_unpack);
        line[x] = schro_dequantise (value, quant_factor, quant_offset);
      }
    }
  }

  j = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line1;
    int16_t *line2;
    SchroFrameData block1;
    SchroFrameData block2;

    schro_frame_data_get_codeblock (&block1, lowdelay->chroma1_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
    schro_frame_data_get_codeblock (&block2, lowdelay->chroma2_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);
    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset_1_2[quant_index];

    for (y = 0; y < block1.height; y++) {
      line1 = SCHRO_FRAME_DATA_GET_LINE (&block1, y);
      line2 = SCHRO_FRAME_DATA_GET_LINE (&block2, y);
      for (x = 0; x < block1.width; x++) {
        value = schro_unpack_decode_sint (&uv_unpack);
        line1[x] = schro_dequantise (value, quant_factor, quant_offset);
        value = schro_unpack_decode_sint (&uv_unpack);
        line2[x] = schro_dequantise (value, quant_factor, quant_offset);
      }
    }
  }
}

static int
__schro_dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0)
    return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2) >> 2);
  } else {
    return (q * quant_factor + quant_offset + 2) >> 2;
  }
}

static void
schro_decoder_decode_slice_fast (SchroPicture * picture,
    SchroLowDelay * lowdelay,
    int slice_x, int slice_y, int offset, int slice_bytes)
{
  SchroParams *params = &picture->params;
  SchroUnpack y_unpack;
  SchroUnpack uv_unpack;
  int quant_index;
  int base_index;
  int length_bits;
  int slice_y_length;
  int i;
  int j;
  int x, y;
  int value;
  int16_t *quant_data = lowdelay->quant_data;

  schro_unpack_init_with_data (&y_unpack,
      OFFSET (picture->lowdelay_buffer->data, offset), slice_bytes, 1);

  base_index = schro_unpack_decode_bits (&y_unpack, 7);
  length_bits = ilog2up (8 * slice_bytes);

  slice_y_length = schro_unpack_decode_bits (&y_unpack, length_bits);

  schro_unpack_copy (&uv_unpack, &y_unpack);
  schro_unpack_limit_bits_remaining (&y_unpack, slice_y_length);
  schro_unpack_skip_bits (&uv_unpack, slice_y_length);

  schro_unpack_decode_sint_s16 (quant_data, &y_unpack, lowdelay->slice_y_size);
  j = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line;
    SchroFrameData block;

    block.data = SCHRO_FRAME_DATA_GET_PIXEL_S16 (lowdelay->luma_subbands + i,
        slice_x * lowdelay->subbands[i].slice_y_width,
        slice_y * lowdelay->subbands[i].slice_y_height);
    block.stride = lowdelay->luma_subbands[i].stride;
    block.width = lowdelay->subbands[i].slice_y_width;
    block.height = lowdelay->subbands[i].slice_y_height;

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset_1_2[quant_index];

    if (block.width == 8) {
      orc_dequantise_s16_2d_8xn (block.data, block.stride,
          quant_data + j, block.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block.height);
      j += block.width * block.height;
    } else if (block.width == 4) {
      orc_dequantise_s16_2d_4xn (block.data, block.stride,
          quant_data + j, block.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block.height);
      j += block.width * block.height;
    } else {
      for (y = 0; y < block.height; y++) {
        line = SCHRO_FRAME_DATA_GET_LINE (&block, y);
        for (x = 0; x < block.width; x++) {
          value = quant_data[j];
          j++;
          line[x] = __schro_dequantise (value, quant_factor, quant_offset);
        }
      }
    }
  }

  schro_unpack_decode_sint_s16 (quant_data, &uv_unpack,
      lowdelay->slice_uv_size);
  orc_deinterleave2_s16 (lowdelay->u_data, lowdelay->v_data, quant_data,
      lowdelay->slice_uv_size / 2);
  j = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line1;
    int16_t *line2;
    SchroFrameData block1;
    SchroFrameData block2;

    block1.data =
        SCHRO_FRAME_DATA_GET_PIXEL_S16 (lowdelay->chroma1_subbands + i,
        slice_x * lowdelay->subbands[i].slice_uv_width,
        slice_y * lowdelay->subbands[i].slice_uv_height);
    block1.stride = lowdelay->chroma1_subbands[i].stride;
    block1.width = lowdelay->subbands[i].slice_uv_width;
    block1.height = lowdelay->subbands[i].slice_uv_height;

    block2.data =
        SCHRO_FRAME_DATA_GET_PIXEL_S16 (lowdelay->chroma2_subbands + i,
        slice_x * lowdelay->subbands[i].slice_uv_width,
        slice_y * lowdelay->subbands[i].slice_uv_height);
    block2.stride = lowdelay->chroma2_subbands[i].stride;
    block2.width = lowdelay->subbands[i].slice_uv_width;
    block2.height = lowdelay->subbands[i].slice_uv_height;

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);
    quant_factor = schro_table_quant[quant_index];
    quant_offset = schro_table_offset_1_2[quant_index];

    if (block1.width == 8) {
      orc_dequantise_s16_2d_8xn (block1.data, block1.stride,
          quant_data + j, block1.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block1.height);
      orc_dequantise_s16_2d_8xn (block2.data, block2.stride,
          quant_data + j, block2.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block2.height);
      j += block1.width * block1.height;
    } else if (block1.width == 4) {
      orc_dequantise_s16_2d_4xn (block1.data, block1.stride,
          quant_data + j, block2.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block1.height);
      orc_dequantise_s16_2d_4xn (block2.data, block2.stride,
          quant_data + j, block2.width * sizeof (int16_t),
          quant_factor, quant_offset + 2, block2.height);
      j += block1.width * block1.height;
    } else {
      for (y = 0; y < block1.height; y++) {
        int16_t *u_data = lowdelay->u_data;
        int16_t *v_data = lowdelay->v_data;
        line1 = SCHRO_FRAME_DATA_GET_LINE (&block1, y);
        line2 = SCHRO_FRAME_DATA_GET_LINE (&block2, y);
        for (x = 0; x < block1.width; x++) {
          value = u_data[j];
          line1[x] = __schro_dequantise (value, quant_factor, quant_offset);
          value = v_data[j++];
          line2[x] = __schro_dequantise (value, quant_factor, quant_offset);
        }
      }
    }
  }
}

static void
schro_lowdelay_init (SchroLowDelay * lowdelay, SchroFrame * frame,
    SchroParams * params)
{
  int i;
  int size;

  lowdelay->params = params;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    int position = schro_subband_get_position (i);
    int shift;

    schro_subband_get_frame_data (lowdelay->luma_subbands + i,
        frame, 0, position, params);
    schro_subband_get_frame_data (lowdelay->chroma1_subbands + i,
        frame, 1, position, params);
    schro_subband_get_frame_data (lowdelay->chroma2_subbands + i,
        frame, 2, position, params);

    shift = params->transform_depth - SCHRO_SUBBAND_SHIFT (position);

    lowdelay->subbands[i].slice_y_width =
        (params->iwt_luma_width >> shift) / params->n_horiz_slices;
    lowdelay->subbands[i].slice_y_height =
        (params->iwt_luma_height >> shift) / params->n_vert_slices;
    lowdelay->subbands[i].slice_uv_width =
        (params->iwt_chroma_width >> shift) / params->n_horiz_slices;
    lowdelay->subbands[i].slice_uv_height =
        (params->iwt_chroma_height >> shift) / params->n_vert_slices;
  }

  size = 1000;
  lowdelay->saved_dc_values = schro_malloc (sizeof (int16_t) * size);
}

static void
schro_lowdelay_cleanup (SchroLowDelay * lowdelay)
{

  schro_free (lowdelay->saved_dc_values);
}

void
schro_decoder_decode_lowdelay_transform_data_fast (SchroPicture * picture)
{
  SchroParams *params = &picture->params;
  SchroLowDelay lowdelay;
  int x, y;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;
  int offset;

  memset (&lowdelay, 0, sizeof (SchroLowDelay));
  schro_lowdelay_init (&lowdelay, picture->transform_frame, params);

  lowdelay.n_horiz_slices = params->n_horiz_slices;
  lowdelay.n_vert_slices = params->n_vert_slices;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  lowdelay.slice_y_size = (params->iwt_luma_width / params->n_horiz_slices) *
      (params->iwt_luma_height / params->n_vert_slices);
  lowdelay.slice_uv_size = (params->iwt_chroma_width / params->n_horiz_slices) *
      (params->iwt_chroma_height / params->n_vert_slices) * 2;
  lowdelay.quant_data = schro_malloc (sizeof (int16_t) *
      MAX (lowdelay.slice_y_size, lowdelay.slice_uv_size));
  lowdelay.u_data = schro_malloc (sizeof (int16_t) *
      lowdelay.slice_uv_size / 2);
  lowdelay.v_data = schro_malloc (sizeof (int16_t) *
      lowdelay.slice_uv_size / 2);

  lowdelay.slice_y_width = (params->iwt_luma_width / params->n_horiz_slices);
  lowdelay.slice_y_height = (params->iwt_luma_height / params->n_vert_slices);
  lowdelay.slice_uv_width = (params->iwt_chroma_width / params->n_horiz_slices);
  lowdelay.slice_uv_height =
      (params->iwt_chroma_height / params->n_vert_slices);

  offset = 0;
  accumulator = 0;
  for (y = 0; y < lowdelay.n_vert_slices; y++) {

    for (x = 0; x < lowdelay.n_horiz_slices; x++) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      schro_decoder_decode_slice_fast (picture, &lowdelay,
          x, y, offset, n_bytes + extra);
      offset += n_bytes + extra;
    }
  }

  schro_decoder_subband_dc_predict (lowdelay.luma_subbands + 0);
  schro_decoder_subband_dc_predict (lowdelay.chroma1_subbands + 0);
  schro_decoder_subband_dc_predict (lowdelay.chroma2_subbands + 0);

  schro_free (lowdelay.quant_data);
  schro_free (lowdelay.u_data);
  schro_free (lowdelay.v_data);
  schro_lowdelay_cleanup (&lowdelay);
}

void
schro_decoder_decode_lowdelay_transform_data_slow (SchroPicture * picture)
{
  SchroParams *params = &picture->params;
  SchroLowDelay lowdelay;
  int x, y;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;
  int offset;

  memset (&lowdelay, 0, sizeof (SchroLowDelay));

  schro_lowdelay_init (&lowdelay, picture->transform_frame, params);

  lowdelay.n_horiz_slices = params->n_horiz_slices;
  lowdelay.n_vert_slices = params->n_vert_slices;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  offset = 0;
  accumulator = 0;
  for (y = 0; y < lowdelay.n_vert_slices; y++) {

    for (x = 0; x < lowdelay.n_horiz_slices; x++) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      schro_decoder_decode_slice_slow (picture, &lowdelay,
          x, y, offset, n_bytes + extra);
      offset += n_bytes + extra;
    }
  }

  schro_decoder_subband_dc_predict (lowdelay.luma_subbands + 0);
  schro_decoder_subband_dc_predict (lowdelay.chroma1_subbands + 0);
  schro_decoder_subband_dc_predict (lowdelay.chroma2_subbands + 0);

  schro_lowdelay_cleanup (&lowdelay);
}

void
schro_decoder_decode_lowdelay_transform_data (SchroPicture * picture)
{
  SchroParams *params = &picture->params;

  if ((params->iwt_chroma_width >> params->transform_depth) %
      params->n_horiz_slices == 0 &&
      (params->iwt_chroma_height >> params->transform_depth) %
      params->n_vert_slices == 0) {
    /* This should be true for all formats in the low-delay profile */
    return schro_decoder_decode_lowdelay_transform_data_fast (picture);
  } else {
    return schro_decoder_decode_lowdelay_transform_data_slow (picture);
  }
}

#ifdef ENABLE_ENCODER
static int
schro_dc_predict (int16_t * line, int stride, int x, int y)
{
  int16_t *prev_line = OFFSET (line, -stride);

  if (y > 0) {
    if (x > 0) {
      return schro_divide3 (line[-1] + prev_line[0] + prev_line[-1] + 1);
    } else {
      return prev_line[0];
    }
  } else {
    if (x > 0) {
      return line[-1];
    } else {
      return 0;
    }
  }
}

static int
schro_encoder_encode_slice (SchroEncoderFrame * frame,
    SchroLowDelay * lowdelay,
    int slice_x, int slice_y, int slice_bytes, int base_index)
{
  int length_bits;
  int slice_y_length;
  int i;
  int start_bits;
  int end_bits;
  int16_t *quant_data = frame->quant_data;

  start_bits = schro_pack_get_bit_offset (frame->pack);

  schro_pack_encode_bits (frame->pack, 7, base_index);
  length_bits = ilog2up (8 * slice_bytes);

  slice_y_length = frame->slice_y_bits - frame->slice_y_trailing_zeros;
  schro_pack_encode_bits (frame->pack, length_bits, slice_y_length);

  for (i = 0; i < lowdelay->slice_y_size - frame->slice_y_trailing_zeros; i++) {
    schro_pack_encode_sint (frame->pack, quant_data[i]);
  }

  quant_data += lowdelay->slice_y_size;
  for (i = 0; i < lowdelay->slice_uv_size - frame->slice_uv_trailing_zeros / 2;
      i++) {
    schro_pack_encode_sint (frame->pack, quant_data[i]);
    schro_pack_encode_sint (frame->pack,
        quant_data[i + lowdelay->slice_uv_size]);
  }

  end_bits = schro_pack_get_bit_offset (frame->pack);
  SCHRO_DEBUG ("total bits %d used bits %d expected %d", slice_bytes * 8,
      end_bits - start_bits,
      7 + length_bits + frame->slice_y_bits + frame->slice_uv_bits -
      frame->slice_y_trailing_zeros - frame->slice_uv_trailing_zeros);
  SCHRO_ASSERT (end_bits - start_bits ==
      7 + length_bits + frame->slice_y_bits + frame->slice_uv_bits -
      frame->slice_y_trailing_zeros - frame->slice_uv_trailing_zeros);

  if (end_bits - start_bits > slice_bytes * 8) {
    SCHRO_ERROR
        ("slice overran buffer by %d bits (slice_bytes %d base_index %d)",
        end_bits - start_bits - slice_bytes * 8, slice_bytes, base_index);
    SCHRO_ASSERT (0);
  } else {
    int left = slice_bytes * 8 - (end_bits - start_bits);
    for (i = 0; i < left; i++) {
      schro_pack_encode_bit (frame->pack, 1);
    }
  }

  return end_bits - start_bits;
}

static int
estimate_array (int16_t * data, int n)
{
  int i;
  int n_bits = 0;

  for (i = 0; i < n; i++) {
    n_bits += schro_pack_estimate_sint (data[i]);
  }
  return n_bits;
}

static void
quantise_block (SchroFrameData * block, int16_t * quant_data, int quant_index)
{
  int quant_factor;
  int quant_offset;
  int x, y;
  int n = 0;
  int16_t *line;

  quant_factor = schro_table_quant[quant_index];
  quant_offset = schro_table_offset_1_2[quant_index];

  for (y = 0; y < block->height; y++) {
    line = SCHRO_FRAME_DATA_GET_LINE (block, y);
    for (x = 0; x < block->width; x++) {
      quant_data[n] = schro_quantise (line[x], quant_factor, quant_offset);
      n++;
    }
  }
}

static void
quantise_dc_block (SchroFrameData * block, int16_t * quant_data,
    int quant_index, int slice_x, int slice_y)
{
  int quant_factor;
  int quant_offset;
  int x, y;
  int n = 0;
  int pred_value;
  int16_t *line;

  quant_factor = schro_table_quant[quant_index];
  quant_offset = schro_table_offset_1_2[quant_index];

  for (y = 0; y < block->height; y++) {
    line = SCHRO_FRAME_DATA_GET_LINE (block, y);
    for (x = 0; x < block->width; x++) {
      pred_value = schro_dc_predict (line + x, block->stride,
          slice_x + x, slice_y + y);
      quant_data[n] = schro_quantise (line[x] - pred_value,
          quant_factor, quant_offset);
      line[x] = pred_value + schro_dequantise (quant_data[n],
          quant_factor, quant_offset);
      n++;
    }
  }
}

static void
dequantise_block (SchroFrameData * block, int16_t * quant_data, int quant_index)
{
  int quant_factor;
  int quant_offset;
  int x, y;
  int n = 0;
  int16_t *line;

  quant_factor = schro_table_quant[quant_index];
  quant_offset = schro_table_offset_1_2[quant_index];

  for (y = 0; y < block->height; y++) {
    line = SCHRO_FRAME_DATA_GET_LINE (block, y);
    for (x = 0; x < block->width; x++) {
      line[x] = schro_dequantise (quant_data[n], quant_factor, quant_offset);
      n++;
    }
  }
}

static void
copy_block_out (int16_t * dest, SchroFrameData * block)
{
  int i;
  int x, y;
  int16_t *line;

  i = 0;
  for (y = 0; y < block->height; y++) {
    line = SCHRO_FRAME_DATA_GET_LINE (block, y);
    for (x = 0; x < block->width; x++) {
      dest[i] = line[x];
      i++;
    }
  }
}

static void
copy_block_in (SchroFrameData * block, int16_t * src)
{
  int i;
  int x, y;
  int16_t *line;

  i = 0;
  for (y = 0; y < block->height; y++) {
    line = SCHRO_FRAME_DATA_GET_LINE (block, y);
    for (x = 0; x < block->width; x++) {
      line[x] = src[i];
      i++;
    }
  }
}

static int
schro_encoder_estimate_slice (SchroEncoderFrame * frame,
    SchroLowDelay * lowdelay,
    int slice_x, int slice_y, int slice_bytes, int base_index)
{
  SchroParams *params = &frame->params;
  int quant_index;
  int i;
  int n_bits;
  int n;
  int16_t *quant_data = frame->quant_data;

  n_bits = 7 + ilog2up (8 * slice_bytes);

  /* Figure out how many values are in each component. */
  /* FIXME this should go somewhere else or be elimitated */
  lowdelay->slice_y_size = 0;
  lowdelay->slice_uv_size = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    SchroFrameData block;

    schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
    lowdelay->slice_y_size += block.height * block.width;

    schro_frame_data_get_codeblock (&block, lowdelay->chroma1_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
    lowdelay->slice_uv_size += block.height * block.width;
  }

  /* Estimate Y */
  n = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    SchroFrameData block;

    schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    if (i == 0) {
      quantise_dc_block (&block, quant_data + n, quant_index,
          (lowdelay->luma_subbands[i].width * slice_x) /
          lowdelay->n_horiz_slices,
          (lowdelay->luma_subbands[i].height * slice_y) /
          lowdelay->n_vert_slices);
    } else {
      quantise_block (&block, quant_data + n, quant_index);
    }
    n += block.height * block.width;
  }
#ifdef USE_TRAILING_DEAD_ZONE
  for (i = 0; i < n; i++) {
    if (quant_data[n - 1 - i] < -1 || quant_data[n - 1 - i] > 1)
      break;
    quant_data[n - 1 - i] = 0;
  }
#endif
  frame->slice_y_bits = estimate_array (quant_data, n);

  for (i = 0; i < n; i++) {
    if (quant_data[n - 1 - i] != 0)
      break;
  }
  frame->slice_y_trailing_zeros = i;

  /* Estimate UV */
  n = 0;
  quant_data += lowdelay->slice_y_size;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    SchroFrameData block1;
    SchroFrameData block2;

    schro_frame_data_get_codeblock (&block1, lowdelay->chroma1_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
    schro_frame_data_get_codeblock (&block2, lowdelay->chroma2_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    if (i == 0) {
      quantise_dc_block (&block1, quant_data + n, quant_index,
          (lowdelay->chroma1_subbands[i].width * slice_x) /
          lowdelay->n_horiz_slices,
          (lowdelay->chroma1_subbands[i].height * slice_y) /
          lowdelay->n_vert_slices);
      quantise_dc_block (&block2, quant_data + n + lowdelay->slice_uv_size,
          quant_index,
          (lowdelay->chroma1_subbands[i].width * slice_x) /
          lowdelay->n_horiz_slices,
          (lowdelay->chroma1_subbands[i].height * slice_y) /
          lowdelay->n_vert_slices);
    } else {
      quantise_block (&block1, quant_data + n, quant_index);
      quantise_block (&block2, quant_data + n + lowdelay->slice_uv_size,
          quant_index);
    }
    n += block1.height * block1.width;
  }
#ifdef USE_TRAILING_DEAD_ZONE
  for (i = 0; i < n; i++) {
    if (quant_data[n - 1 - i] < -1 || quant_data[n - 1 - i] > 1)
      break;
    if (quant_data[2 * n - 1 - i] < -1 || quant_data[2 * n - 1 - i] > 1)
      break;
    quant_data[n - 1 - i] = 0;
    quant_data[2 * n - 1 - i] = 0;
  }
#endif
  frame->slice_uv_bits = estimate_array (quant_data, n * 2);

  for (i = 0; i < n; i++) {
    if (quant_data[n - 1 - i] != 0)
      break;
    if (quant_data[2 * n - 1 - i] != 0)
      break;
  }
  frame->slice_uv_trailing_zeros = 2 * i;

  return n_bits + frame->slice_y_bits + frame->slice_uv_bits -
      frame->slice_y_trailing_zeros - frame->slice_uv_trailing_zeros;
}

static void
schro_encoder_dequantise_slice (SchroEncoderFrame * frame,
    SchroLowDelay * lowdelay,
    int slice_x, int slice_y, int slice_bytes, int base_index)
{
  SchroParams *params = &frame->params;
  int quant_index;
  int i;
  int n;
  int16_t *quant_data = frame->quant_data;

  n = 0;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    SchroFrameData block;

    schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    if (i == 0) {
      /* dc dequant is handled by estimation */
    } else {
      dequantise_block (&block, quant_data + n, quant_index);
    }
    n += block.height * block.width;
  }

  n = 0;
  quant_data += lowdelay->slice_y_size;
  for (i = 0; i < 1 + 3 * params->transform_depth; i++) {
    SchroFrameData block1;
    SchroFrameData block2;

    schro_frame_data_get_codeblock (&block1, lowdelay->chroma1_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
    schro_frame_data_get_codeblock (&block2, lowdelay->chroma2_subbands + i,
        slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);

    quant_index = CLAMP (base_index - params->quant_matrix[i], 0, 60);

    if (i == 0) {
      /* dc dequant is handled by estimation */
    } else {
      dequantise_block (&block1, quant_data + n, quant_index);
      dequantise_block (&block2, quant_data + n + lowdelay->slice_uv_size,
          quant_index);
    }
    n += block1.height * block1.width;
  }
}

static void
save_dc_values (SchroEncoderFrame * frame, int16_t * dc_values,
    SchroLowDelay * lowdelay, int slice_x, int slice_y)
{
  SchroFrameData block;

  schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_out (dc_values, &block);
  dc_values += block.width * block.height;

  schro_frame_data_get_codeblock (&block, lowdelay->chroma1_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_out (dc_values, &block);
  dc_values += block.width * block.height;

  schro_frame_data_get_codeblock (&block, lowdelay->chroma2_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_out (dc_values, &block);
}

static void
restore_dc_values (SchroEncoderFrame * frame, int16_t * dc_values,
    SchroLowDelay * lowdelay, int slice_x, int slice_y)
{
  SchroFrameData block;

  schro_frame_data_get_codeblock (&block, lowdelay->luma_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_in (&block, dc_values);
  dc_values += block.width * block.height;

  schro_frame_data_get_codeblock (&block, lowdelay->chroma1_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_in (&block, dc_values);
  dc_values += block.width * block.height;

  schro_frame_data_get_codeblock (&block, lowdelay->chroma2_subbands + 0,
      slice_x, slice_y, lowdelay->n_horiz_slices, lowdelay->n_vert_slices);
  copy_block_in (&block, dc_values);
}

static int
schro_encoder_pick_slice_index (SchroEncoderFrame * frame,
    SchroLowDelay * lowdelay, int slice_x, int slice_y, int slice_bytes)
{
  int i;
  int n;
  int size;

  save_dc_values (frame, lowdelay->saved_dc_values, lowdelay, slice_x, slice_y);

  i = 0;
  n = schro_encoder_estimate_slice (frame, lowdelay,
      slice_x, slice_y, slice_bytes, i);
  if (n <= slice_bytes * 8) {
    schro_encoder_dequantise_slice (frame, lowdelay,
        slice_x, slice_y, slice_bytes, i);
    return i;
  }
  restore_dc_values (frame, lowdelay->saved_dc_values, lowdelay,
      slice_x, slice_y);

  size = 32;
  while (size >= 1) {
    n = schro_encoder_estimate_slice (frame, lowdelay,
        slice_x, slice_y, slice_bytes, i + size);
    restore_dc_values (frame, lowdelay->saved_dc_values, lowdelay,
        slice_x, slice_y);
    if (n >= slice_bytes * 8) {
      i += size;
    }
    size >>= 1;
  }

  schro_encoder_estimate_slice (frame, lowdelay,
      slice_x, slice_y, slice_bytes, i + 1);
  schro_encoder_dequantise_slice (frame, lowdelay,
      slice_x, slice_y, slice_bytes, i + 1);
  return i + 1;
}

void
schro_encoder_encode_lowdelay_transform_data (SchroEncoderFrame * frame)
{
  SchroParams *params = &frame->params;
  SchroLowDelay lowdelay;
  int x, y;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;
  int base_index;
  int total_bits;

  schro_lowdelay_init (&lowdelay, frame->iwt_frame, params);

  lowdelay.n_horiz_slices = params->n_horiz_slices;
  lowdelay.n_vert_slices = params->n_vert_slices;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  accumulator = 0;
  total_bits = 0;
  for (y = 0; y < lowdelay.n_vert_slices; y++) {

    for (x = 0; x < lowdelay.n_horiz_slices; x++) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      base_index = schro_encoder_pick_slice_index (frame, &lowdelay,
          x, y, n_bytes + extra);
      total_bits += schro_encoder_encode_slice (frame, &lowdelay,
          x, y, n_bytes + extra, base_index);
    }
  }

  SCHRO_INFO ("used bits %d of %d", total_bits,
      lowdelay.n_horiz_slices * lowdelay.n_vert_slices *
      params->slice_bytes_num * 8 / params->slice_bytes_denom);

  schro_lowdelay_cleanup (&lowdelay);
}
#endif
