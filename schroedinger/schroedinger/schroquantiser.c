
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

  base = frame->encoder->prefs[SCHRO_PREF_QUANT_BASE] - 4;

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

static double probability_to_entropy (double x)
{
  if (x <= 0 || x >= 1.0) return 0;

  return -(x * log(x) + (1-x) * log(1-x))*INV_LOG_2;
}

static double entropy (int a, int total)
{
  double x;

  if (total == 0) return 0;

  x = (double)a / total;
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
  double vol = subband->w * subband->h;
  double entropy;
  double x;

  data_entropy = 0;
  count_pos = 0;
  count_neg = 0;
  count_0 = 0;
  error_total = 0;

  quant_factor = schro_table_quant[subband->quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[subband->quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[subband->quant_index];
  }

  data = OFFSET(frame->iwt_frame->components[component].data, 2*subband->offset);
  for(j=0;j<subband->h;j++){
    for(i=0;i<subband->w;i++){
      q = quantize(data[i], quant_factor, quant_offset);
      value = dequantize(q, quant_factor, quant_offset);

      if (q > 0) count_pos++;
      if (q < 0) count_neg++;
      if (q == 0) count_0++;
      data_entropy += abs(q);

      error_total += pow4(value - data[i]);
    }
    data = OFFSET(data, subband->stride);
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



static int
ilog2 (int x)
{
  int i;
  if (x<0) x = -x;
  for(i=0;i<60;i++){
    if (x*4 < schro_table_quant[i+1]) return i;
  }
#if 0
  for(i=0;x>1;i++){
    x >>= 1;
  }
#endif
  return i;
}

#if 0
#define SHIFT 2

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
#endif


static double
estimate_histogram_entropy (int hist[], int quant_index, int volume)
{
  int i;
  double x;
  int cont1;
  double data_entropy;
  double estimated_entropy;
  int bin1, bin2, bin3, bin4, bin5, bin6;
  int n;

  data_entropy = 0;
  bin1 = 0;
  bin2 = 0;
  bin3 = 0;
  bin4 = 0;
  bin5 = 0;
  bin6 = 0;
  for(i=0;i<4;i++){
    bin1 += hist[quant_index + i];
    bin2 += hist[quant_index + i + 4];
    bin3 += hist[quant_index + i + 8];
    bin4 += hist[quant_index + i + 12];
    bin5 += hist[quant_index + i + 16];
  }
  for(i=quant_index + 20; i < 64; i++) {
    bin6 += hist[quant_index + i];
  }
  for(i=quant_index;i<64;i++){
    x = (i - quant_index)/4.0;
    data_entropy += x * hist[i];
  }
  for(i=quant_index+4;i<64;i++){
    cont1 += hist[i];
  }

  estimated_entropy = 0;

  bin1 = 0;
  for(i=quant_index;i<64;i++){
    bin1 += hist[i];
  }

  bin2 = 0;
  /* 6.5 is good for some points, 7 good for others. */
  bin2 += hist[quant_index+6]/2;
  for(i=quant_index+7;i<64;i++){
    bin2 += hist[i];
  }

  bin3 = 0;
  bin3 += hist[quant_index+11]/2;
  for(i=quant_index+12;i<64;i++){
    bin3 += hist[i];
  }

  bin4 = 0;
  for(i=quant_index+16;i<64;i++){
    bin4 += hist[i];
  }

  bin5 = 0;
  for(i=quant_index+20;i<64;i++){
    bin5 += hist[i];
  }

  bin6 = 0;
  for(i=quant_index+24;i<64;i++){
    bin6 += hist[i];
  }

  n = bin1+bin2+bin3+bin4+bin5+bin6;

  /* entropy of sign bit */
  estimated_entropy += bin1;

  /* entropy of first continue bit */
  //x = (double)n / (width*height);
  /* HACK needs scale factor of about 0.75 (ranges from 0.5 to 0.95) */
  estimated_entropy += entropy (bin1, volume);
  estimated_entropy += entropy (bin2, bin1);
  estimated_entropy += entropy (bin3, bin2);
  estimated_entropy += entropy (bin4, bin3);
  estimated_entropy += entropy (bin5, bin4);
  estimated_entropy += entropy (bin6, bin5);

  /* data entropy */
  /* both seem to work equally well */
  estimated_entropy += bin1 + bin2 + bin3 + bin4 + bin5 + bin6;
  //frame->estimated_entropy += data_entropy;

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
  int stride;
  int width;
  int height;
  int offset;
  int skip = 1;
  double error = 0;
  int q;
  int quant_factor;
  int quant_offset;
  int value;

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

  quant_factor = schro_table_quant[quant_index];
  if (frame->params.num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  error = 0;
  data = (int16_t *)frame->iwt_frame->components[component].data + offset;
  for(j=0;j<height;j+=skip){
    for(i=0;i<width;i+=skip){
      q = quantize(abs(data[i]), quant_factor, quant_offset);
      value = dequantize(q, quant_factor, quant_offset);
      error += pow4(value - abs(data[i]));
    }
    data += stride * skip;
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
#if 1
  int i;
  double estimated_error;
  int j;
  int q;
  int range_start, range_end;
  int quant_factor, quant_offset;
  int err;

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
    j = ilog2(i);
    range_start = schro_table_quant[j]/4;
    range_end = schro_table_quant[j+1]/4;

    if (range_end > range_start) {
      estimated_error += pow4(err) * hist[j]/(range_end - range_start);
    }
  }

#if 0
  /* uh, don't ask. */
  if (quant_index > 0 && quant_index < 20) {
    static double fixups[] =
    { 0, 1.102, 1.00, 1.08, 1.09, 1.47, 1.41, 1.09, 1.05, 1.39, 1.12,
      1.04, 1.15, 0.81, 0.89, 0.94, 1.03, 1.00, 0.91, 0.99 };
    estimated_error *= fixups[quant_index];
  }
#endif

  return estimated_error;
#else
  int i;
  double estimated_error;
  double x;
  int j;
  int q;
  int range_start, range_end;
  int quant_factor, quant_offset;
  int err;

  estimated_error = 0;

  quant_factor = schro_table_quant[quant_index];
  if (num_refs > 0) {
    quant_offset = schro_table_offset_3_8[quant_index];
  } else {
    quant_offset = schro_table_offset_1_2[quant_index];
  }

  for(i=0;i<quant_factor*4;i++){
    q = quantize (i, quant_factor, quant_offset);
    err = dequantize (q, quant_factor, quant_offset) - i;
    j = ilog2(i);
    range_start = schro_table_quant[j]/4;
    range_end = schro_table_quant[j+1]/4;

    if (range_end > range_start) {
      estimated_error += pow4(err) * hist[j]/(range_end - range_start);
    }
  }

  range_end = schro_table_quant[quant_index];
  x = 0;
  for(i=0;i<range_end;i++){
    x += pow4(i*0.25);
  }
  x /= range_end;

  for(i=quant_index+8;i<64;i++){
    estimated_error += x * hist[i];
  }

  return estimated_error;
#endif
#endif
}

void
schro_encoder_estimate_subband (SchroEncoderFrame *frame, int component,
    int index)
{
  SchroSubband *subband = frame->subbands + index;
  int i;
  int j;
  int16_t *data;
  int stride;
  int width;
  int height;
  int offset;
  int hist[64 + 24];
  int skip = 4;

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

  for(i=0;i<64 + 24;i++) hist[i] = 0;

  data = (int16_t *)frame->iwt_frame->components[component].data + offset;
  for(j=0;j<height;j+=skip){
    for(i=0;i<width;i+=skip){
      hist[ilog2(data[i])]++;
    }
    data += stride * skip;
  }
  for(i=0;i<64;i++){
    hist[i] *= skip*skip;
  }

  if (component == 0 && index == 10) {
    static int n = 0;
    static double sum[64] = { 0 };

    n++;
    for(i=0;i<20;i++){
      double est_entropy;
      double error;
      double est_error;

      error = measure_error_subband (frame, component, index, i);
      est_entropy = estimate_histogram_entropy (hist, i, width*height);
      est_error = estimate_histogram_error (hist, i, frame->params.num_refs,
          width*height);

      SCHRO_ERROR("SUBBAND_CURVE: %d %d %d %g %g %g", component, index, i,
          est_entropy/(width*height), sqrt(error/(width*height)),
          sqrt(est_error/(width*height)));

      sum[i] += sqrt(error/(width*height)) / sqrt(est_error/(width*height));
      SCHRO_ERROR("FIXUP: %d %g", i, sum[i]/n);
    }


    SCHRO_ERROR("SUBBAND_CURVE:  ");
  }

  frame->estimated_entropy =
    estimate_histogram_entropy (hist, subband->quant_index, width*height);
}


