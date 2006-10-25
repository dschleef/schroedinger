
#include <stdio.h>
#include <math.h>

int
get_quant (int i)
{
  unsigned long long base;
  base = 1<<(i/4);
  switch(i&3) {
    case 0:
      return 4*base;
    case 1:
      return (78892 * base + 8292) / 16585;
    case 2:
      return (228486 * base + 20195) / 40391;
    case 3:
      return (440253 * base + 32722) / 65444;
  }
}

int
get_offset (int i)
{
  return (get_quant(i) * 3 + 4)/8;
}

unsigned int
get_inv_quant (int i)
{
  int q = get_quant(i);
  return (1ULL<<32)/q;
}

int
get_factor (int i)
{
  if (i<2) return 0;
  return (0x10000 + i/2)/i;
}

int
get_arith_shift (int i)
{
  int range1 = (i&0xf0)<<8;
  int range0 = (i&0xf)<<12;
  int n = 0;
  int flip = 0;

  if (range1 < range0) {
    return 0;
  }
  while (((range0 ^ range1)&(1<<15)) == 0 && n<4) {
    range0 <<= 1;
    range1 <<= 1;
    n++;
  }
  while (((range0 & ~range1)&(1<<14)) && n<3) {
    range0 <<= 1;
    range1 <<= 1;
    range0 ^= (1<<15);
    range1 ^= (1<<15);
    flip = 1;
    n++;
  }

  return (flip<<15) | n;
}

int
main (int argc, char *argv[])
{
  int i;

  printf("\n");
  printf("#include <schroedinger/schrotables.h>\n");
  printf("\n");

  /* schro_table_offset */
  printf("uint32_t schro_table_offset[61] = {\n");
  for(i=0;i<60;i+=4) {
    printf("  %7d, %7d, %7d, %7d,\n",
        get_offset(i),
        get_offset(i+1),
        get_offset(i+2),
        get_offset(i+3));
  }
  printf("  %7d\n", get_offset(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("uint32_t schro_table_quant[61] = {\n");
  for(i=0;i<60;i+=4) {
    printf("  %7d, %7d, %7d, %7d,\n",
        get_quant(i),
        get_quant(i+1),
        get_quant(i+2),
        get_quant(i+3));
  }
  printf("  %7d\n", get_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("uint32_t schro_table_inverse_quant[61] = {\n");
  for(i=0;i<60;i+=4) {
    printf("  %10uu, %10uu, %10uu, %10uu,\n",
        get_inv_quant(i),
        get_inv_quant(i+1),
        get_inv_quant(i+2),
        get_inv_quant(i+3));
  }
  printf("  %10uu\n", get_inv_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("uint16_t schro_table_division_factor[257] = {\n");
  for(i=0;i<256;i+=4) {
    printf("  %5u, %5u, %5u, %5u,\n",
        get_factor(i),
        get_factor(i+1),
        get_factor(i+2),
        get_factor(i+3));
  }
  printf("  %5u\n", get_factor(i));
  printf("};\n");

  /* arith shift table */
  printf("\n");
  printf("uint16_t schro_table_arith_shift[256] = {\n");
  for(i=0;i<252;i+=4) {
    printf("  /* 0x%02x */ 0x%04x, 0x%04x, 0x%04x, 0x%04x,\n", i,
        get_arith_shift(i),
        get_arith_shift(i+1),
        get_arith_shift(i+2),
        get_arith_shift(i+3));
  }
  printf("  /* 0x%02x */ 0x%04x, 0x%04x, 0x%04x, 0x%04x\n", i,
      get_arith_shift(i),
      get_arith_shift(i+1),
      get_arith_shift(i+2),
      get_arith_shift(i+3));
  printf("};\n");


  return 0;
}

