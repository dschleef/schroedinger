
#include <stdio.h>
#include <math.h>

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
main (int argc, char *argv[])
{
  int i;
  int n = 60;

  printf("\n");
  printf("#include <schroedinger/schrotables.h>\n");
  printf("\n");

  /* schro_table_offset_3_8 */
  printf("uint32_t schro_table_offset_3_8[%d] = {\n", n + 1);
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
  printf("uint32_t schro_table_offset_1_2[%d] = {\n", n+1);
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
  printf("uint32_t schro_table_quant[%d] = {\n", n + 1);
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
  printf("uint32_t schro_table_inverse_quant[%d] = {\n", n + 1);
  for(i=0;i<n;i+=4) {
    printf("  %10uu, %10uu, %10uu, %10uu,\n",
        get_inv_quant(i),
        get_inv_quant(i+1),
        get_inv_quant(i+2),
        get_inv_quant(i+3));
  }
  printf("  %10uu\n", get_inv_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_division_factor */
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

  return 0;
}

