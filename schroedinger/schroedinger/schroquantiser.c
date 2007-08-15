
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schrohistogram.h>
#include <liboil/liboil.h>
#include <string.h>
#include <math.h>


#if 0
/* This is 64*log2 of the gain of the DC part of the wavelet transform */
static const int wavelet_gain[] = { 64, 64, 64, 0, 64, 128, 128, 103 };
/* horizontal/vertical part */
static const int wavelet_gain_hv[] = { 64, 64, 64, 0, 64, 128, 0, 65 };
/* diagonal part */
static const int wavelet_gain_diag[] = { 128, 128, 128, 64, 128, 256, -64, 90 };
#endif

void schro_encoder_choose_quantisers_simple (SchroEncoderFrame *frame);
void schro_encoder_choose_quantisers_hardcoded (SchroEncoderFrame *frame);

void
schro_encoder_choose_quantisers (SchroEncoderFrame *frame)
{

  switch (frame->encoder->quantiser_engine) {
    case 0:
      schro_encoder_choose_quantisers_hardcoded (frame);
      break;
    case 1:
      schro_encoder_choose_quantisers_simple (frame);
      break;
  }
}

void
schro_encoder_choose_quantisers_simple (SchroEncoderFrame *frame)
{
  int depth = frame->params.transform_depth;
  int base;
  int i;
  int component;

  base = frame->encoder->prefs[SCHRO_PREF_QUANT_BASE];

  for(component=0;component<3;component++){
    if (depth >= 1) {
      frame->quant_index[component][(depth-1)*3 + 1] = base;
      frame->quant_index[component][(depth-1)*3 + 2] = base;
      frame->quant_index[component][(depth-1)*3 + 3] = base + 4;
    }
    if (depth >= 2) {
      frame->quant_index[component][(depth-2)*3 + 1] = base - 5;
      frame->quant_index[component][(depth-2)*3 + 2] = base - 5;
      frame->quant_index[component][(depth-2)*3 + 3] = base - 1;
    }
    for(i=3;i<=depth;i++){
      frame->quant_index[component][(depth-i)*3 + 1] = base - 6;
      frame->quant_index[component][(depth-i)*3 + 2] = base - 6;
      frame->quant_index[component][(depth-i)*3 + 3] = base - 2;
    }
    frame->quant_index[component][0] = base - 10;

    if (!frame->is_ref) {
      for(i=0;i<depth*3+1;i++){
        frame->quant_index[component][i] += 4;
      }
    }

    frame->quant_index[component][(depth-1)*3 + 1] = 4;
    frame->quant_index[component][(depth-1)*3 + 2] = 5;
    frame->quant_index[component][(depth-1)*3 + 3] = 6;
  }

}

void
schro_encoder_choose_quantisers_hardcoded (SchroEncoderFrame *frame)
{
  int depth = frame->params.transform_depth;
  int i;
  int component;

  /* hard coded.  muhuhuhahaha */

  /* these really only work for DVD-ish quality with 5,3, 9,3 and 13,5 */

  for(component=0;component<3;component++){
    if (depth >= 1) {
      frame->quant_index[component][(depth-1)*3 + 1] = 22;
      frame->quant_index[component][(depth-1)*3 + 2] = 22;
      frame->quant_index[component][(depth-1)*3 + 3] = 26;
    }
    if (depth >= 2) {
      frame->quant_index[component][(depth-2)*3 + 1] = 17;
      frame->quant_index[component][(depth-2)*3 + 2] = 17;
      frame->quant_index[component][(depth-2)*3 + 3] = 21;
    }
    for(i=3;i<=depth;i++){
      frame->quant_index[component][(depth-i)*3 + 1] = 16;
      frame->quant_index[component][(depth-i)*3 + 2] = 16;
      frame->quant_index[component][(depth-i)*3 + 3] = 20;
    }
    frame->quant_index[component][0] = 12;

    if (!frame->is_ref) {
      for(i=0;i<depth*3+1;i++){
        frame->quant_index[component][i] += 4;
      }
    }
  }
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

static double pow4(double x)
{
  x = x*x;
  return x*x;
}

/* log(2.0) */
#define LOG_2 0.69314718055994528623
/* 1.0/log(2.0) */
#define INV_LOG_2 1.44269504088896338700

static double
probability_to_entropy (double x)
{
  if (x <= 0 || x >= 1.0) return 0;

  return -(x * log(x) + (1-x) * log(1-x))*INV_LOG_2;
}

static double
entropy (double a, double total)
{
  double x;

  if (total == 0) return 0;

  x = a / total;
  return probability_to_entropy (x) * total;
}

void
error_calculation_subband (SchroEncoderFrame *frame, int component,
    int index)
{
  int i,j;
  double error_total;
  int count_pos;
  int count_neg;
  int count_0;
  int data_entropy;
  int quant_index;
  int quant_factor;
  int quant_offset;
  int q;
  int value;
  int16_t *data;
  int16_t *line;
  double vol;
  double entropy;
  double x;
  int width;
  int height;
  int stride;
  int position;

  data_entropy = 0;
  count_pos = 0;
  count_neg = 0;
  count_0 = 0;
  error_total = 0;

  vol = width * height;

  quant_index = frame->quant_index[component][index];
  quant_factor = schro_table_quant[quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);

  for(j=0;j<height;j++){
    line = OFFSET(data, j*stride);
    for(i=0;i<width;i++){
      q = quantize(line[i], quant_factor, quant_offset);
      value = dequantize(q, quant_factor, quant_offset);

      if (q > 0) count_pos++;
      if (q < 0) count_neg++;
      if (q == 0) count_0++;
      data_entropy += abs(q);

      error_total += pow4(value - line[i]);
    }
  }

  entropy = 0;

  /* data symbols (assume probability of 0.5) */
  x = 0.5;
  entropy += probability_to_entropy(x) * (data_entropy/4.0);

  /* first continue symbol (zero vs. non-zero) */
  x = (double)count_0 / vol;
  entropy += probability_to_entropy(x) * vol;

  /* subsequent continue symbols (assume probability of 0.5) */
  x = 0.5;
  entropy += probability_to_entropy(x) * (data_entropy/4.0);

  /* sign symbols */
  if (count_pos + count_neg > 0) {
    x = count_pos / (count_pos + count_neg);
    entropy += probability_to_entropy(x) * (count_pos + count_neg);
  }

  frame->estimated_entropy = entropy;
}

static double
estimate_histogram_entropy (SchroHistogram *hist, int quant_index, int volume)
{
  double estimated_entropy = 0;
  double bin1, bin2, bin3, bin4, bin5, bin6;
  int quant_factor;

  quant_factor = schro_table_quant[quant_index];

  bin1 = schro_histogram_get_range (hist, (quant_factor+3)/4, 32000);
  bin2 = schro_histogram_get_range (hist, (quant_factor*3+3)/4, 32000);
  bin3 = schro_histogram_get_range (hist, (quant_factor*7+3)/4, 32000);
  bin4 = schro_histogram_get_range (hist, (quant_factor*15+3)/4, 32000);
  bin5 = schro_histogram_get_range (hist, (quant_factor*31+3)/4, 32000);
  bin6 = schro_histogram_get_range (hist, (quant_factor*63+3)/4, 32000);

  /* entropy of sign bit */
  estimated_entropy += bin1;

  /* entropy of first continue bit */
  estimated_entropy += entropy (bin1, volume);
  estimated_entropy += entropy (bin2, bin1);
  estimated_entropy += entropy (bin3, bin2);
  estimated_entropy += entropy (bin4, bin3);
  estimated_entropy += entropy (bin5, bin4);
  estimated_entropy += entropy (bin6, bin5);

  /* data entropy */
  estimated_entropy += bin1 + bin2 + bin3 + bin4 + bin5 + bin6;
  
  return estimated_entropy;
}

static double
measure_error_subband (SchroEncoderFrame *frame, int component, int index,
    int quant_index)
{
  int i;
  int j;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;
  int skip = 1;
  double error = 0;
  int q;
  int quant_factor;
  int quant_offset;
  int value;
  int position;

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);

  quant_factor = schro_table_quant[quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  error = 0;
  for(j=0;j<height;j+=skip){
    line = OFFSET(data, j*stride);
    for(i=0;i<width;i+=skip){
      q = quantize(abs(line[i]), quant_factor, quant_offset);
      value = dequantize(q, quant_factor, quant_offset);
      error += pow4(value - abs(line[i]));
    }
  }
  error *= skip*skip;

  return error;
}

static double
estimate_histogram_error (SchroHistogram *hist, int quant_index,
    int num_refs, int volume)
{
  return schro_histogram_apply_table (hist,
    (SchroHistogramTable *)(schro_table_error_hist_shift3_1_2[quant_index]));
}

double
schro_encoder_estimate_subband_arith (SchroEncoderFrame *frame, int component,
    int index, int quant_index)
{
  int i;
  int j;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;
  int q;
  int quant_factor;
  int estimated_entropy;
  SchroArith *arith;
  int position;

  arith = schro_arith_new ();
  schro_arith_estimate_init (arith);

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);

  quant_factor = schro_table_quant[quant_index];

  for(j=0;j<height;j++) {
    line = OFFSET(data, j*stride);
    for(i=0;i<width;i++) {
      q = quantize(line[i], quant_factor, 0);
      schro_arith_estimate_sint (arith,
          SCHRO_CTX_ZPZN_F1, SCHRO_CTX_COEFF_DATA, SCHRO_CTX_SIGN_ZERO, q);
    }
  }

  estimated_entropy = 0;

  estimated_entropy += arith->contexts[SCHRO_CTX_ZPZN_F1].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F2].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F3].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F4].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F5].n_bits;
  estimated_entropy += arith->contexts[SCHRO_CTX_ZP_F6p].n_bits;

  estimated_entropy += arith->contexts[SCHRO_CTX_COEFF_DATA].n_bits;

  estimated_entropy += arith->contexts[SCHRO_CTX_SIGN_ZERO].n_bits;

  schro_arith_free (arith);

  return estimated_entropy;
}

void
schro_encoder_generate_subband_histogram (SchroEncoderFrame *frame,
    int component, int index, SchroHistogram *hist, int skip)
{
  int i;
  int j;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;
  int position;

  schro_histogram_init (hist);

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);

  if (index > 0) {
    for(j=0;j<height;j+=skip){
      schro_histogram_add_array_s16 (hist, OFFSET(data, j*stride), width);
    }
    schro_histogram_scale (hist, skip);
  } else {
    for(j=0;j<height;j+=skip){
      line = OFFSET(data, j*stride);
      for(i=1;i<width;i+=skip){
        schro_histogram_add(hist, (line[i] - line[i-1]));
      }
    }
    schro_histogram_scale (hist, skip*skip);
  }
}

static int
pick_quant (SchroHistogram *hist, double lambda, int vol)
{
  double error;
  double entropy;
  double new_error;
  double new_entropy;
  int index;

  index = 30;
  entropy = estimate_histogram_entropy (hist, index, vol);
  error = estimate_histogram_error (hist, index, 0, vol);

  while (index > 0) {
    new_entropy = estimate_histogram_entropy (hist, index - 1, vol);
    new_error = estimate_histogram_error (hist, index - 1, 0, vol);

    if (new_entropy - entropy > lambda * (error - new_error)) {
      return index;
    }

    entropy = new_entropy;
    error = new_error;
    index--;
  }

  return 0;
}

void
schro_encoder_estimate_subband (SchroEncoderFrame *frame, int component,
    int index)
{
  int i;
  SchroHistogram hist;
  int vol;
  int16_t *data;
  int stride;
  int width, height;
  double lambda;
  int position;

  schro_encoder_generate_subband_histogram (frame, component, index, &hist, 4);

  position = schro_subband_get_position (index);
  schro_subband_get (frame->iwt_frame, component, position,
      &frame->params, &data, &stride, &width, &height);
  vol = width * height;

  if (frame->encoder->internal_testing) {
    for(i=0;i<30;i++){
      double est_entropy;
      double error;
      double est_error;
      double arith_entropy;

      error = measure_error_subband (frame, component, index, i);
      est_entropy = estimate_histogram_entropy (&hist, i, vol);
      est_error = estimate_histogram_error (&hist, i, frame->params.num_refs,
          vol);
      arith_entropy = schro_encoder_estimate_subband_arith (frame, component,
          index, i);

      SCHRO_INFO("SUBBAND_CURVE: %d %d %d %g %g %g %g", component, index, i,
          est_entropy/vol, arith_entropy/vol,
          sqrt(est_error/vol), sqrt(error/vol));
    }
  }

  //lambda = 0.0001;
  if (index == 0) {
    lambda = 0.0001;
  } else {
    lambda = 0.00001;
  }
  if (component > 0) {
    lambda *= 0.1;
  }
  frame->quant_index[component][index] = pick_quant (&hist, lambda, vol);

  frame->estimated_entropy =
    estimate_histogram_entropy (&hist, frame->quant_index[component][index],
        vol);
}


