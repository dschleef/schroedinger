
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <schroedinger/schrooil.h>
#include <schroedinger/schrounpack.h>
#include <string.h>



typedef struct _SchroSliceRun SchroSliceRun;

struct _SchroSliceRun {
  int16_t *data1;
  int16_t *data2;

  int x_stride;
  int y_stride;

  int width;
  int height;
  int stride;
};



void
schro_lowdelay_get_luma_slice_run (SchroFrame *frame,
    int position, SchroParams *params, SchroSliceRun *run)
{
  int shift;
  int w;
  SchroFrameComponent *comp = &frame->components[0];
  
  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);

  run->stride = comp->stride << shift;
  run->width = (1<<params->slice_width_exp) >> shift;
  run->height = (1<<params->slice_height_exp) >> shift;
  run->x_stride = run->width * sizeof(int16_t);
  run->y_stride = run->height * run->stride;

  w = params->iwt_luma_width >> shift;

  run->data1 = comp->data;
  if (position & 2) {
    run->data1 = OFFSET(run->data1, run->stride>>1);
  }
  if (position & 1) {
    run->data1 = OFFSET(run->data1, w*sizeof(int16_t));
  }

  SCHRO_DEBUG("Y pos %d wxh %dx%d str %d xstr %d ystr %d",
      position, run->width, run->height, run->stride, run->x_stride, run->y_stride);
}

void
schro_lowdelay_get_chroma_slice_run (SchroFrame *frame,
    int position, SchroParams *params, SchroSliceRun *run)
{
  int shift;
  int w;
  SchroFrameComponent *comp1 = &frame->components[1];
  SchroFrameComponent *comp2 = &frame->components[2];
  
  shift = params->transform_depth - SCHRO_SUBBAND_SHIFT(position);

  SCHRO_ASSERT(comp1->stride == comp2->stride);

  run->stride = comp1->stride << shift;
  run->width = 1<<(params->slice_width_exp - shift - params->video_format->chroma_h_shift);
  run->height = 1<<(params->slice_height_exp - shift - params->video_format->chroma_v_shift);
  run->x_stride = run->width * sizeof(int16_t);
  run->y_stride = run->height * run->stride;

  w = params->iwt_chroma_width >> shift;

  run->data1 = comp1->data;
  run->data2 = comp2->data;
  if (position & 2) {
    run->data1 = OFFSET(run->data1, run->stride>>1);
    run->data2 = OFFSET(run->data2, run->stride>>1);
  }
  if (position & 1) {
    run->data1 = OFFSET(run->data1, w*sizeof(int16_t));
    run->data2 = OFFSET(run->data2, w*sizeof(int16_t));
  }

  SCHRO_DEBUG("UV pos %d wxh %dx%d str %d xstr %d ystr %d",
      position, run->width, run->height, run->stride, run->x_stride, run->y_stride);
}


static int
dequantize (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

static int
ilog2up (unsigned int x)
{
  int i;

  for(i=0;i<32;i++){
    if (x == 0) return i;
    x >>= 1;
  }
  return 0;
}


void
schro_decoder_decode_slice (SchroDecoder *decoder, SchroSliceRun *luma_runs,
    SchroSliceRun *chroma_runs,
    int slice_x, int slice_y, int slice_bytes)
{
  SchroParams *params = &decoder->params;
  SchroUnpack y_unpack;
  SchroUnpack uv_unpack;
  SchroSliceRun *run;
  int quant_index;
  int base_index;
  int length_bits;
  int slice_y_length;
  int i;
  int x,y;
  int value;

  schro_unpack_init_with_data (&y_unpack,
      decoder->bits->buffer->data + decoder->bits->n, slice_bytes, 1);

  base_index = schro_unpack_decode_bits (&y_unpack, 7);
  length_bits = ilog2up(8*slice_bytes);

  slice_y_length = schro_unpack_decode_bits (&y_unpack, length_bits);

  schro_unpack_copy (&uv_unpack, &y_unpack);
  schro_unpack_limit_bits_remaining (&y_unpack, slice_y_length);
  schro_unpack_skip_bits (&uv_unpack, slice_y_length);

  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line;

    run = luma_runs + i;

    quant_index = base_index - params->quant_matrix[i] + params->luma_quant_offset;

    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    for(y=0;y<run->height;y++){
      for (x=0; x<run->width; x++){
        value = schro_unpack_decode_sint (&y_unpack);
        line[x] = dequantize (value, quant_factor, quant_offset);
      }
      line = OFFSET(line, run->stride);
    }
  }

  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_factor1;
    int quant_offset1;
    int quant_factor2;
    int quant_offset2;
    int16_t *line1;
    int16_t *line2;

    run = chroma_runs + i;

    quant_index = base_index - params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_factor1 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset1 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    quant_index = base_index - params->quant_matrix[i] + params->chroma2_quant_offset;
    quant_factor2 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset2 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line1 = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    line2 = OFFSET(run->data2, run->y_stride * slice_y + run->x_stride * slice_x);

    for(y=0;y<run->height;y++){
      for (x=0; x<run->width; x++){
        value = schro_unpack_decode_sint (&uv_unpack);
        line1[x] = dequantize (value, quant_factor1, quant_offset1);
        value = schro_unpack_decode_sint (&uv_unpack);
        line2[x] = dequantize (value, quant_factor2, quant_offset2);
      }
      line1 = OFFSET(line1, run->stride);
      line2 = OFFSET(line2, run->stride);
    }
  }

  schro_bits_skip (decoder->bits, slice_bytes);
}

void
schro_decoder_decode_lowdelay_transform_data (SchroDecoder *decoder)
{
  SchroParams *params = &decoder->params;
  SchroSliceRun luma_runs[SCHRO_MAX_SUBBANDS];
  SchroSliceRun chroma_runs[SCHRO_MAX_SUBBANDS];
  int x,y;
  int n_horiz_slices;
  int n_vert_slices;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;
  int i;

  for(i=0;i<1+3*params->transform_depth;i++){
    int position = schro_subband_get_position(i);
    
    schro_lowdelay_get_luma_slice_run (decoder->frame, position, params,
        luma_runs + i);
    schro_lowdelay_get_chroma_slice_run (decoder->frame, position, params,
        chroma_runs + i);
  }

  n_horiz_slices = params->iwt_luma_width>>params->slice_width_exp;
  n_vert_slices = params->iwt_luma_height>>params->slice_height_exp;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  accumulator = 0;
  for(y=0;y<n_vert_slices;y++) {

    for(x=0;x<n_horiz_slices;x++) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      schro_decoder_decode_slice (decoder, luma_runs, chroma_runs,
          x, y, n_bytes + extra);
    }
  }

  if (decoder->n_refs == 0) {
    int i;
    int16_t *data;
    int stride;
    int width;
    int height;

    for(i=0;i<3;i++){
      schro_subband_get (decoder->frame, i, 0,
          params, &data, &stride, &width, &height);

      schro_decoder_subband_dc_predict (data, stride, width, height);
    }
  }
}


static int
schro_dc_predict (int16_t *data, int stride, int x, int y)
{
  int16_t *line = OFFSET(data, stride * y);
  int16_t *prev_line = OFFSET(data, stride * (y-1));

  if (y > 0) {
    if (x > 0) {
      return schro_divide(line[x-1] + prev_line[x] + prev_line[x-1] + 1,3);
    } else {
      return prev_line[x];
    }
  } else {
    if (x > 0) {
      return line[x-1];
    } else {
      return 0;
    }
  }
}

static int
schro_dc_predict_2 (int16_t *line, int stride, int x, int y)
{
  int16_t *prev_line = OFFSET(line, -stride);

  if (y > 0) {
    if (x > 0) {
      return schro_divide(line[-1] + prev_line[0] + prev_line[-1] + 1,3);
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
quantize (int value, int quant_factor, int quant_offset)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)<<2;
    x /= quant_factor;
    value = -x;
  } else {
    x = value<<2;
    x /= quant_factor;
    value = x;
  }
  return value;
}

int
schro_encoder_encode_slice (SchroEncoderFrame *frame, SchroSliceRun *luma_runs,
    SchroSliceRun *chroma_runs,
    int slice_x, int slice_y, int slice_bytes, int base_index)
{
  SchroParams *params = &frame->params;
  SchroSliceRun *run;
  int length_bits;
  int slice_y_length;
  int i;
  int x,y;
  int value;
  int start_bits;
  int end_bits;
  int pred_value;
  int quant_index;

  start_bits = schro_bits_get_bit_offset (frame->bits);

  schro_bits_encode_bits (frame->bits, 7, base_index);
  length_bits = ilog2up(8*slice_bytes);

  slice_y_length = 0;
  schro_bits_encode_bits (frame->bits, length_bits, frame->slice_y_bits);

#if 0
  schro_bits_copy (&ybits, frame->bits);
  schro_bits_set_length (&ybits, slice_y_length);

  schro_bits_skip_bits (frame->bits, slice_y_length);
#endif

  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line;

    run = luma_runs + i;

    quant_index = base_index - params->quant_matrix[i] + params->luma_quant_offset;

    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    if (i == 0) {
      for(y=0;y<run->height;y++){
        for (x=0; x<run->width; x++){
          pred_value = schro_dc_predict (run->data1, run->stride,
              run->width * slice_x + x, run->height * slice_y + y);
          value = quantize (line[x] - pred_value, quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, value);
          line[x] = pred_value + dequantize (value, quant_factor, quant_offset);
        }
        line = OFFSET(line, run->stride);
      }
    } else {
      for(y=0;y<run->height;y++){
        for (x=0; x<run->width; x++){
          value = quantize (line[x], quant_factor, quant_offset);
          schro_bits_encode_sint (frame->bits, value);
          line[x] = dequantize (value, quant_factor, quant_offset);
        }
        line = OFFSET(line, run->stride);
      }
    }
  }

  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_factor1;
    int quant_offset1;
    int quant_factor2;
    int quant_offset2;
    int16_t *line1;
    int16_t *line2;

    run = chroma_runs + i;

    quant_index = base_index - params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_factor1 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset1 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    quant_index = base_index - params->quant_matrix[i] + params->chroma2_quant_offset;
    quant_factor2 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset2 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line1 = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    line2 = OFFSET(run->data2, run->y_stride * slice_y + run->x_stride * slice_x);

    if (i == 0) {
      for(y=0;y<run->height;y++){
        for (x=0; x<run->width; x++){
          pred_value = schro_dc_predict (run->data1, run->stride,
              run->width * slice_x + x, run->height * slice_y + y);
          value = quantize (line1[x] - pred_value, quant_factor1, quant_offset1);
          schro_bits_encode_sint (frame->bits, value);
          line1[x] = pred_value + dequantize (value, quant_factor1, quant_offset1);

          pred_value = schro_dc_predict (run->data2, run->stride,
              run->width * slice_x + x, run->height * slice_y + y);
          value = quantize (line2[x] - pred_value, quant_factor2, quant_offset2);
          schro_bits_encode_sint (frame->bits, value);
          line2[x] = pred_value + dequantize (value, quant_factor2, quant_offset2);
        }
        line1 = OFFSET(line1, run->stride);
        line2 = OFFSET(line2, run->stride);
      }
    } else {
      for(y=0;y<run->height;y++){
        for (x=0; x<run->width; x++){
          value = quantize (line1[x], quant_factor1, quant_offset1);
          schro_bits_encode_sint (frame->bits, value);
          line1[x] = dequantize (value, quant_factor1, quant_offset1);

          value = quantize (line2[x], quant_factor2, quant_offset2);
          schro_bits_encode_sint (frame->bits, value);
          line2[x] = dequantize (value, quant_factor2, quant_offset2);
        }
        line1 = OFFSET(line1, run->stride);
        line2 = OFFSET(line2, run->stride);
      }
    }
  }

  end_bits = schro_bits_get_bit_offset (frame->bits);
  SCHRO_DEBUG("total bits %d used bits %d", slice_bytes*8,
      end_bits - start_bits);

  if (end_bits - start_bits > slice_bytes*8) {
    SCHRO_ERROR("slice overran buffer by %d bits (slice_bytes %d base_index %d)",
        end_bits - start_bits - slice_bytes*8, slice_bytes, base_index);
    SCHRO_ASSERT(0);
  } else {
    int left = slice_bytes*8 - (end_bits - start_bits);
    for(i=0;i<left; i++) {
      schro_bits_encode_bit (frame->bits, 0);
    }
  }

  return end_bits - start_bits;
}

static int
estimate_array (int16_t *data, int n)
{
  int i;
  int n_bits = 0;

  for(i=0;i<n;i++){
    n_bits += schro_bits_estimate_sint (data[i]);
  }
  return n_bits;
}

void
quantise_run (SchroSliceRun *run, int16_t *line, int16_t *quant_data,
    int quant_index)
{
  int quant_factor;
  int quant_offset;
  int x,y;
  int n = 0;

  quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
  quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

  for(y=0;y<run->height;y++){
    for (x=0; x<run->width; x++){
      quant_data[n] = quantize (line[x], quant_factor, quant_offset);
      n++;
    }
    line = OFFSET(line, run->stride);
  }
}

void
quantise_dc_run (SchroSliceRun *run, int16_t *line, int16_t *quant_data,
    int quant_index, int slice_x, int slice_y)
{
  int quant_factor;
  int quant_offset;
  int x,y;
  int n = 0;
  int pred_value;

  quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
  quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

  for(y=0;y<run->height;y++){
    for (x=0; x<run->width; x++){
      pred_value = schro_dc_predict_2 (line + x, run->stride,
          run->width * slice_x + x, run->height * slice_y + y);
      quant_data[n] = quantize (line[x] - pred_value, quant_factor, quant_offset);
      line[x] = pred_value + dequantize (quant_data[n], quant_factor, quant_offset);
      n++;
    }
    line = OFFSET(line, run->stride);
  }
}

void
copy_slice_run_out (int16_t *dest, int16_t *line, SchroSliceRun *run)
{
  int i;
  int x, y;

  i = 0;
  for(y=0;y<run->height;y++){
    for (x=0; x<run->width; x++){
      dest[i] = line[x];
      i++;
    }
    line = OFFSET(line, run->stride);
  }
}

void
copy_slice_run_in (int16_t *line, int16_t *src, SchroSliceRun *run)
{
  int i;
  int x, y;

  i = 0;
  for(y=0;y<run->height;y++){
    for (x=0; x<run->width; x++){
      line[x] = src[i];
      i++;
    }
    line = OFFSET(line, run->stride);
  }
}

int
schro_encoder_estimate_slice (SchroEncoderFrame *frame, SchroSliceRun *luma_runs,
    SchroSliceRun *chroma_runs,
    int slice_x, int slice_y, int slice_bytes, int base_index)
{
  SchroParams *params = &frame->params;
  SchroSliceRun *run;
  int length_bits;
  int i;
  int n_bits;
  int16_t *quant_data = frame->quant_data;
  int16_t tmp_dc_values[100];
  int n;

  n_bits = 0;

  n_bits += 7;
  length_bits = ilog2up(8*slice_bytes);
  n_bits += length_bits;

  n = 0;
  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_index;
    int16_t *line;

    run = luma_runs + i;

    quant_index = base_index - params->quant_matrix[i] + params->luma_quant_offset;

    line = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    if (i == 0) {
      copy_slice_run_out (tmp_dc_values, line, run);
      quantise_dc_run (run, line, quant_data + n, quant_index, slice_x, slice_y);
      copy_slice_run_in (line, tmp_dc_values, run);
      n += run->height*run->width;
    } else {
      quantise_run (run, line, quant_data + n, quant_index);
      n += run->height*run->width;
    }
  }
  frame->slice_y_bits = estimate_array (quant_data, n);

  n = 0;
  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_index1;
    int quant_index2;
    int16_t *line1;
    int16_t *line2;

    run = chroma_runs + i;

    quant_index1 = base_index - params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_index2 = base_index - params->quant_matrix[i] + params->chroma2_quant_offset;

    line1 = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    line2 = OFFSET(run->data2, run->y_stride * slice_y + run->x_stride * slice_x);

    if (i == 0) {
      copy_slice_run_out (tmp_dc_values, line1, run);
      quantise_dc_run (run, line1, quant_data + n, quant_index1, slice_x, slice_y);
      copy_slice_run_in (line1, tmp_dc_values, run);
      n += run->height*run->width;

      copy_slice_run_out (tmp_dc_values, line2, run);
      quantise_dc_run (run, line2, quant_data + n, quant_index2, slice_x, slice_y);
      copy_slice_run_in (line2, tmp_dc_values, run);
      n += run->height*run->width;
    } else {
      quantise_run (run, line1, quant_data + n, quant_index1);
      n += run->height*run->width;

      quantise_run (run, line2, quant_data + n, quant_index2);
      n += run->height*run->width;
    }
  }
  frame->slice_uv_bits = estimate_array (quant_data, n);

  return n_bits + frame->slice_y_bits + frame->slice_uv_bits;
}

int
schro_encoder_pick_slice_index (SchroEncoderFrame *frame,
    SchroSliceRun *luma_runs, SchroSliceRun *chroma_runs,
    int slice_x, int slice_y, int slice_bytes)
{
  int i;
  int n;
  int size;

  i = 0;
  n = schro_encoder_estimate_slice (frame, luma_runs, chroma_runs,
      slice_x, slice_y, slice_bytes, i);
  if (n <= slice_bytes*8) return i;

  size = 32;
  while (size >= 1) {
    n = schro_encoder_estimate_slice (frame, luma_runs, chroma_runs,
        slice_x, slice_y, slice_bytes, i + size);
    if (n >= slice_bytes*8) {
      i += size;
    }
    size >>= 1;
  }

  schro_encoder_estimate_slice (frame, luma_runs, chroma_runs,
      slice_x, slice_y, slice_bytes, i + 1);
  return i+1;
}

void
schro_encoder_encode_lowdelay_transform_data (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroSliceRun luma_runs[SCHRO_MAX_SUBBANDS];
  SchroSliceRun chroma_runs[SCHRO_MAX_SUBBANDS];
  int x,y;
  int n_horiz_slices;
  int n_vert_slices;
  int n_bytes;
  int remainder;
  int accumulator;
  int extra;
  int i;
  int base_index;
  int total_bits;

  for(i=0;i<1+3*params->transform_depth;i++){
    int position = schro_subband_get_position(i);
    
    schro_lowdelay_get_luma_slice_run (frame->iwt_frame, position, params,
        luma_runs + i);
    schro_lowdelay_get_chroma_slice_run (frame->iwt_frame, position, params,
        chroma_runs + i);
  }

  n_horiz_slices = params->iwt_luma_width>>params->slice_width_exp;
  n_vert_slices = params->iwt_luma_height>>params->slice_height_exp;

  n_bytes = params->slice_bytes_num / params->slice_bytes_denom;
  remainder = params->slice_bytes_num % params->slice_bytes_denom;

  accumulator = 0;
  total_bits = 0;
  for(y=0;y<n_vert_slices;y++) {

    for(x=0;x<n_horiz_slices;x++) {
      accumulator += remainder;
      if (accumulator >= params->slice_bytes_denom) {
        extra = 1;
        accumulator -= params->slice_bytes_denom;
      } else {
        extra = 0;
      }

      base_index = schro_encoder_pick_slice_index (frame, luma_runs,
          chroma_runs, x, y, n_bytes + extra);
      total_bits += schro_encoder_encode_slice (frame, luma_runs, chroma_runs,
          x, y, n_bytes + extra, base_index);
    }
  }

  SCHRO_INFO("used bits %d of %d", total_bits,
      n_horiz_slices * n_vert_slices * params->slice_bytes_num * 8 /
      params->slice_bytes_denom);
}



