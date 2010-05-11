
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <schroedinger/schrohistogram.h>

unsigned int
get_quant (int i)
{
  unsigned long long base;
  base = 1ULL<<(i/4);
  switch(i&3) {
    case 0:
      return 4*base;
    case 1:
      return (503829 * base + 52958) / 105917;
    case 2:
      return (665857 * base + 58854) / 117708;
    case 3:
      return (440253 * base + 32722) / 65444;
  }
  return 0;
}

unsigned int
get_offset_3_8 (int i)
{
  unsigned long long quant = get_quant(i);
  if (i == 0)
      return 1;
  return (quant * 3 + 4)/8;
}

int
get_offset_1_2 (int i)
{
  unsigned long long quant = get_quant(i);
  if (i == 0)
      return 1;
  if (i == 1)
      return 2;
  return (quant + 1)/2;
}

uint16_t
get_inv_quant (int i)
{
  int quant_factor = get_quant(i);
  int quant_shift = i/4 + 2;
  return floor(0.5 + 65536.0 / quant_factor * (1<<quant_shift));
}

int
get_factor (int i)
{
  if (i<2) return 0;
  return (0x10000 + i/2)/i;
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

typedef struct _ErrorFuncInfo ErrorFuncInfo;
struct _ErrorFuncInfo {
  int quant_factor;
  int quant_offset;
  double power;
};

static double error_pow2(int x, void *priv)
{
  ErrorFuncInfo *efi = priv;
  int q;
  int value;
  int y;

  q = quantize (x, efi->quant_factor, efi->quant_offset);
  value = dequantize (q, efi->quant_factor, efi->quant_offset);

  y = abs (value - x);

  return pow (y, efi->power);
}

static int
maxbit (unsigned int x)
{
  int i;
  for(i=0;x;i++){
    x >>= 1;
  }
  return i;
}

static int onebits (int value)
{
  int i;
  int n_bits;
  int ones = 0;

  if (value < 0) value = -value;
  if (value == 0) return 0;
  value++;
  n_bits = maxbit (value);
  for(i=0;i<n_bits - 1;i++){
    ones += (value>>(n_bits - 2 - i))&1;
  }
  return ones;
}

static int zerobits (int value)
{
  int i;
  int n_bits;
  int zeros = 0;

  if (value < 0) value = -value;
  if (value == 0) return 0;
  value++;
  n_bits = maxbit (value);
  for(i=0;i<n_bits - 1;i++){
    zeros += 1^((value>>(n_bits - 2 - i))&1);
  }
  return zeros;
}

#define SHIFT 3
static int
iexpx (int x)
{
  if (x < (1<<SHIFT)) return x;

  return ((1<<SHIFT)|(x&((1<<SHIFT)-1))) << ((x>>SHIFT)-1);
}

int
main (int argc, char *argv[])
{
  int i;
  int n = 60;

  printf("\n");
  printf("#include \"config.h\"\n");
  printf("\n");
  printf("#include <schroedinger/schrotables.h>\n");
  printf("\n");

  /* schro_table_offset_3_8 */
  printf("const uint32_t schro_table_offset_3_8[%d] = {\n", n + 1);
  for(i=0;i<n;i+=4) {
    printf("  %10uu, %10uu, %10uu, %10uu,\n",
        get_offset_3_8(i),
        get_offset_3_8(i+1),
        get_offset_3_8(i+2),
        get_offset_3_8(i+3));
  }
  printf("  %10uu\n", get_offset_3_8(i));
  printf("};\n");
  printf("\n");

  /* schro_table_offset_1_2 */
  printf("const uint32_t schro_table_offset_1_2[%d] = {\n", n+1);
  for(i=0;i<n;i+=4) {
    printf("  %10uu, %10uu, %10uu, %10uu,\n",
        get_offset_1_2(i),
        get_offset_1_2(i+1),
        get_offset_1_2(i+2),
        get_offset_1_2(i+3));
  }
  printf("  %10uu\n", get_offset_1_2(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("const uint32_t schro_table_quant[%d] = {\n", n + 1);
  for(i=0;i<n;i+=4) {
    printf("  %10uu, %10uu, %10uu, %10uu,\n",
        get_quant(i),
        get_quant(i+1),
        get_quant(i+2),
        get_quant(i+3));
  }
  printf("  %10uu\n", get_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_inverse_quant */
  printf("const uint16_t schro_table_inverse_quant[%d] = {\n", n + 1);
  for(i=0;i<n;i+=4) {
    printf("  %6uu, %6uu, %6uu, %6uu,\n",
        get_inv_quant(i),
        get_inv_quant(i+1),
        get_inv_quant(i+2),
        get_inv_quant(i+3));
  }
  printf("  %10uu\n", get_inv_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_division_factor */
  printf("const uint16_t schro_table_division_factor[257] = {\n");
  for(i=0;i<256;i+=4) {
    printf("  %5u, %5u, %5u, %5u,\n",
        get_factor(i),
        get_factor(i+1),
        get_factor(i+2),
        get_factor(i+3));
  }
  printf("  %5u\n", get_factor(i));
  printf("};\n\n");

  /* schro_table_error_hist */
  printf("#ifdef ENABLE_ENCODER\n");
  printf("const double schro_table_error_hist_shift3_1_2[60][%d] = {\n",
      ((16-SHIFT)<<SHIFT));
  for(i=0;i<60;i++){
    SchroHistogramTable table;
    ErrorFuncInfo efi;
    int j;

    efi.quant_factor = get_quant(i);
    efi.quant_offset = get_offset_1_2(i);
    efi.power = 2.0;
    schro_histogram_table_generate (&table, error_pow2, &efi);

    printf("  { /* %d */\n", i);
    for(j=0;j<SCHRO_HISTOGRAM_SIZE;j++){
      if ((j&0x7)==0) printf("    ");
      printf("%g, ", table.weights[j]);
      if ((j&0x7)==0x7) printf("\n");
    }
    printf("  },\n");
  }
  printf("};\n\n");

  /* schro_table_onebits_hist */
  printf("const double schro_table_onebits_hist_shift3_1_2[60][%d] = {\n",
      ((16-SHIFT)<<SHIFT));
  for(i=0;i<60;i++){
    int quant_factor = get_quant(i);
    int quant_offset = get_offset_1_2(i);
    int j;

    printf("  { /* %d */\n", i);
    for(j=0;j<((16-SHIFT)<<SHIFT);j++){
      int kmin = iexpx(j);
      int kmax = iexpx(j+1);
      int k;
      double x = 0;
      int q;

      if ((j&0x7)==0) printf("    ");
      for(k=kmin;k<kmax;k++){
        q = quantize(abs(k), quant_factor, quant_offset);
        x += onebits(q);
      }
      if ((j>>SHIFT) > 0) {
        x *= 1.0/(1<<((j>>SHIFT)-1));
      }
      printf("%g, ", x);
      if ((j&0x7)==0x7) printf("\n");
    }
    printf("  },\n");
  }
  printf("};\n\n");

  /* schro_table_zerobits_hist */
  printf("const double schro_table_zerobits_hist_shift3_1_2[60][%d] = {\n",
      ((16-SHIFT)<<SHIFT));
  for(i=0;i<60;i++){
    int quant_factor = get_quant(i);
    int quant_offset = get_offset_1_2(i);
    int j;

    printf("  { /* %d */\n", i);
    for(j=0;j<((16-SHIFT)<<SHIFT);j++){
      int kmin = iexpx(j);
      int kmax = iexpx(j+1);
      int k;
      double x = 0;
      int q;

      if ((j&0x7)==0) printf("    ");
      for(k=kmin;k<kmax;k++){
        q = quantize(abs(k), quant_factor, quant_offset);
        x += zerobits(q);
      }
      if ((j>>SHIFT) > 0) {
        x *= 1.0/(1<<((j>>SHIFT)-1));
      }
      printf("%g, ", x);
      if ((j&0x7)==0x7) printf("\n");
    }
    printf("  },\n");
  }
  printf("};\n\n");
  printf("#endif\n");

  return 0;
}

