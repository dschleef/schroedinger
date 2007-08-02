
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
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
  SchroSubband *subbands = frame->subbands;
  int depth = frame->params.transform_depth;
  int base;
  int i;

  base = frame->encoder->prefs[SCHRO_PREF_QUANT_BASE];

  if (depth >= 1) {
    subbands[(depth-1)*3 + 1].quant_index = base;
    subbands[(depth-1)*3 + 2].quant_index = base;
    subbands[(depth-1)*3 + 3].quant_index = base + 4;
  }
  if (depth >= 2) {
    subbands[(depth-2)*3 + 1].quant_index = base - 5;
    subbands[(depth-2)*3 + 2].quant_index = base - 5;
    subbands[(depth-2)*3 + 3].quant_index = base - 1;
  }
  for(i=3;i<=depth;i++){
    subbands[(depth-i)*3 + 1].quant_index = base - 6;
    subbands[(depth-i)*3 + 2].quant_index = base - 6;
    subbands[(depth-i)*3 + 3].quant_index = base - 2;
  }
  subbands[0].quant_index = base - 10;

  if (!frame->is_ref) {
    for(i=0;i<depth*3+1;i++){
      subbands[i].quant_index += 4;
    }
  }

  subbands[(depth-1)*3 + 1].quant_index = 4;
  subbands[(depth-1)*3 + 2].quant_index = 5;
  subbands[(depth-1)*3 + 3].quant_index = 6;

}

void
schro_encoder_choose_quantisers_hardcoded (SchroEncoderFrame *frame)
{
  SchroSubband *subbands = frame->subbands;
  int depth = frame->params.transform_depth;
  int i;

  /* hard coded.  muhuhuhahaha */

  /* these really only work for DVD-ish quality with 5,3, 9,3 and 13,5 */

  if (depth >= 1) {
    subbands[(depth-1)*3 + 1].quant_index = 22;
    subbands[(depth-1)*3 + 2].quant_index = 22;
    subbands[(depth-1)*3 + 3].quant_index = 26;
  }
  if (depth >= 2) {
    subbands[(depth-2)*3 + 1].quant_index = 17;
    subbands[(depth-2)*3 + 2].quant_index = 17;
    subbands[(depth-2)*3 + 3].quant_index = 21;
  }
  for(i=3;i<=depth;i++){
    subbands[(depth-i)*3 + 1].quant_index = 16;
    subbands[(depth-i)*3 + 2].quant_index = 16;
    subbands[(depth-i)*3 + 3].quant_index = 20;
  }
  subbands[0].quant_index = 12;

  if (!frame->is_ref) {
    for(i=0;i<depth*3+1;i++){
      subbands[i].quant_index += 4;
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
  SchroSubband *subband = frame->subbands + index;
  int i,j;
  double error_total;
  int count_pos;
  int count_neg;
  int count_0;
  int data_entropy;
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

  data_entropy = 0;
  count_pos = 0;
  count_neg = 0;
  count_0 = 0;
  error_total = 0;

  vol = width * height;

  quant_factor = schro_table_quant[subband->quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[subband->quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[subband->quant_index];
  }

  schro_subband_get (frame->iwt_frame, component, subband->position,
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

#if 0
  {
    double error_per_pixel;
    double perceptual_weight = 1.0;

    error_per_pixel = error_total / vol;
    error = sqrt(error_per_pixel) / (perceptual_weight * perceptual_weight);
  }
#endif

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

  //SCHRO_ERROR("estimated entropy %g", entropy);

  frame->estimated_entropy = entropy;
}

#define SHIFT 3

static int
ilogx (int x)
{
  int i = 0;
  if (x < 0) x = -x;
  while (x >= 2<<SHIFT) {
    x >>= 1;
    i++;
  }
  return x + (i << SHIFT);
}

static int
iexpx (int x)
{
  if (x < (1<<SHIFT)) return x;

  return ((1<<SHIFT)|(x&((1<<SHIFT)-1))) << ((x>>SHIFT)-1);
}

static int
ilogx_size (int i)
{
  if (i < (1<<SHIFT)) return 1;
  return 1 << ((i>>SHIFT)-1);
}

static double
rehist (int hist[], int start, int end)
{
  int i;
  int iend;
  int size;
  double x;

  if (start >= end) return 0;

  i = ilogx(start);
  size = ilogx_size(i);
  x = (double)(iexpx(i+1) - start)/size * hist[i];

  i++;
  iend = ilogx(end);
  while (i <= iend) {
    x += hist[i];
    i++;
  }

  size = ilogx_size(iend);
  x -= (double)(iexpx(iend+1) - end)/size * hist[iend];

  return x;
}

static double
estimate_histogram_entropy (int hist[], int quant_index, int volume)
{
  double estimated_entropy = 0;
  double bin1, bin2, bin3, bin4, bin5, bin6;
  int quant_factor;

  quant_factor = schro_table_quant[quant_index];

  bin1 = rehist (hist, (quant_factor+3)/4, 32000);
  bin2 = rehist (hist, (quant_factor*3+3)/4, 32000);
  bin3 = rehist (hist, (quant_factor*7+3)/4, 32000);
  bin4 = rehist (hist, (quant_factor*15+3)/4, 32000);
  bin5 = rehist (hist, (quant_factor*31+3)/4, 32000);
  bin6 = rehist (hist, (quant_factor*63+3)/4, 32000);

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
  SchroSubband *subband = frame->subbands + index;
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

  schro_subband_get (frame->iwt_frame, component, subband->position,
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
estimate_histogram_error (int hist[], int quant_index, int num_refs, int volume)
{
#if 0
  int i;
  double x;
  double estimated_error;
  int j;
  int range_start, range_end;

  estimated_error = 0;

  for(i=0;i<quant_index;i++){
    range_start = schro_table_quant[i]/4;
    range_end = schro_table_quant[i+1]/4;
    for(j=range_start;j<range_end;j++){
      estimated_error += pow4(j) * hist[i]/(range_end - range_start);
    }
  }

  range_end = schro_table_quant[quant_index]/4;
  x = 0;
  for(i=0;i<range_end;i++){
    x += pow4(i);
  }
  x /= range_end;

  for(i=quant_index;i<64;i++){
    estimated_error += x * hist[i];
  }

  return estimated_error;
#else
#if 0
  int i;
  double estimated_error;
  int j;
  int q;
  int quant_factor, quant_offset;
  int err;
  int size;

  estimated_error = 0;

  quant_factor = schro_table_quant[quant_index];
  if (num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  for(i=0;i<32000;i++){
    q = quantize (i, quant_factor, quant_offset);
    err = dequantize (q, quant_factor, quant_offset) - i;
    j = ilogx(i);
    size = ilogx_size (j);
    estimated_error += pow4(err) * hist[j] / size;
  }

  return estimated_error;
#else
  int i;
  double estimated_error;

  estimated_error = 0;

  for(i=0;i<((16-SHIFT)<<SHIFT);i++){
    estimated_error += hist[i] * schro_table_error_hist_shift3_1_2[quant_index][i];
  }

  return estimated_error;
#endif
#endif
}

double
schro_encoder_estimate_subband_arith (SchroEncoderFrame *frame, int component,
    int index, int quant_index)
{
  SchroSubband *subband = frame->subbands + index;
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

  arith = schro_arith_new ();
  schro_arith_estimate_init (arith);

  schro_subband_get (frame->iwt_frame, component, subband->position,
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

#if 0
  for(i=0;i<SCHRO_CTX_LAST;i++){
    estimated_entropy += arith->contexts[i].n_bits;
  }
#endif

  schro_arith_free (arith);

  return estimated_entropy;
}

void
schro_encoder_generate_subband_histogram (SchroEncoderFrame *frame,
    int component, int index, int *hist, int skip)
{
  SchroSubband *subband = frame->subbands + index;
  int i;
  int j;
  int16_t *data;
  int16_t *line;
  int stride;
  int width;
  int height;

  for(i=0;i<(16<<SHIFT)+24;i++){
    hist[i] = 0;
  }

  schro_subband_get (frame->iwt_frame, component, subband->position,
      &frame->params, &data, &stride, &width, &height);

  if (index > 0) {
    for(j=0;j<height;j+=skip){
      line = OFFSET(data, j*stride);
      for(i=0;i<width;i+=skip){
        hist[ilogx(line[i])]++;
      }
    }
  } else {
    for(j=0;j<height;j+=skip){
      line = OFFSET(data, j*stride);
      for(i=1;i<width;i+=skip){
        hist[ilogx(line[i] - line[i-1])]++;
      }
    }
  }
  for(i=0;i<(16<<SHIFT)+24;i++){
    hist[i] *= skip*skip;
  }
}

static int
pick_quant (int *hist, double lambda, int vol)
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
  SchroSubband *subband = frame->subbands + index;
  int i;
  int hist[(16<<SHIFT)+24] = { 0 };
  int vol;
  int16_t *data;
  int stride;
  int width, height;
  double lambda;

  schro_encoder_generate_subband_histogram (frame, component, index, hist, 4);

  schro_subband_get (frame->iwt_frame, component, subband->position,
      &frame->params, &data, &stride, &width, &height);
  vol = width * height;

  if (frame->encoder->internal_testing) {
    for(i=0;i<30;i++){
      double est_entropy;
      double error;
      double est_error;
      double arith_entropy;

      error = measure_error_subband (frame, component, index, i);
      est_entropy = estimate_histogram_entropy (hist, i, vol);
      est_error = estimate_histogram_error (hist, i, frame->params.num_refs,
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
  subband->quant_index = pick_quant (hist, lambda, vol);

  frame->estimated_entropy =
    estimate_histogram_entropy (hist, subband->quant_index, vol);
}


