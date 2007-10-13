
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schro-stdint.h>

#include <math.h>


int
muldiv64 (int a, int b, int c)
{
  int64_t x;

  x = a;
  x *= b;
  x /= c;

  return (int)x;
}

int
schro_utils_multiplier_to_quant_index (double x)
{
  int i = 0;
  
  x *= x;
  x *= x;
  while (x*x > 2) {
    x *= 0.5;
    i++;
  }

  return i;
}


int
schro_dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

int
schro_quantise (int value, int quant_factor, int quant_offset)
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

/* log(2.0) */
#define LOG_2 0.69314718055994528623
/* 1.0/log(2.0) */
#define INV_LOG_2 1.44269504088896338700

double
schro_utils_probability_to_entropy (double x)
{ 
  if (x <= 0 || x >= 1.0) return 0;
  
  return -(x * log(x) + (1-x) * log(1-x))*INV_LOG_2;
}

double
schro_utils_entropy (double a, double total)
{ 
  double x;
  
  if (total == 0) return 0;
  
  x = a / total;
  return schro_utils_probability_to_entropy (x) * total;
}



