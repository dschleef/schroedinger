
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <schroedinger/schrooil.h>
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
  SchroBits slice_bits;
  SchroSliceRun *run;
  int quant_index;
  int qindex;
  int length_bits;
  int slice_y_length;
  int i;
  int x,y;
  int value;

  schro_bits_copy (&slice_bits, decoder->bits);
  schro_bits_set_length (&slice_bits, slice_bytes * 8);

  qindex = schro_bits_decode_bits (&slice_bits, 7);
  length_bits = ilog2up(8*slice_bytes);

  slice_y_length = schro_bits_decode_bits (&slice_bits, length_bits);

#if 0
  schro_bits_copy (&ybits, decoder->bits);
  schro_bits_set_length (&ybits, slice_y_length);

  schro_bits_skip_bits (&slice_bits, slice_y_length);
#endif

  for(i=0;i<1+3*params->transform_depth;i++) {
    int quant_factor;
    int quant_offset;
    int16_t *line;

    run = luma_runs + i;

    quant_index = qindex + params->quant_matrix[i] + params->luma_quant_offset;

    quant_factor = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    for(y=0;y<run->height;y++){
      for (x=0; x<run->width; x++){
        value = schro_bits_decode_sint (&slice_bits);
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

    quant_index = qindex + params->quant_matrix[i] + params->chroma1_quant_offset;
    quant_factor1 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset1 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    quant_index = qindex + params->quant_matrix[i] + params->chroma2_quant_offset;
    quant_factor2 = schro_table_quant[CLAMP(quant_index,0,60)];
    quant_offset2 = schro_table_offset_1_2[CLAMP(quant_index,0,60)];

    line1 = OFFSET(run->data1, run->y_stride * slice_y + run->x_stride * slice_x);
    line2 = OFFSET(run->data2, run->y_stride * slice_y + run->x_stride * slice_x);

    for(y=0;y<run->height;y++){
      for (x=0; x<run->width; x++){
        value = schro_bits_decode_sint (&slice_bits);
        line1[x] = dequantize (value, quant_factor1, quant_offset1);
        value = schro_bits_decode_sint (&slice_bits);
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


