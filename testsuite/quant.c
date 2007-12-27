
#include <stdio.h>
#include <math.h>

static int
dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return ((q * quant_factor - quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

static int
quantise (int value, int quant_factor, int quant_offset)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = ((-value)<<2) - quant_offset + (quant_factor>>1);
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = -x;
  } else {
    x = (value<<2) - quant_offset + (quant_factor>>1);
    //x = (x*(uint64_t)inv_quant_factor)>>32;
    x /= quant_factor;
    value = x;
  }
  return value;
}

int
main (int argc, char *argv[])
{
  int i;
  int quant_factor = 128;
  int quant_offset = 48;
  int q;

  for(i=-1000;i<1000;i++){
    q = quantise(i,quant_factor,quant_offset);
    printf("%d %d\n", i, dequantise(q,quant_factor,quant_offset));
  }

  return 0;
}

