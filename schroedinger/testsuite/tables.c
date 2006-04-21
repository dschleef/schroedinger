
#include <stdio.h>
#include <math.h>

int
get_quant (int i)
{
  return floor(pow(2.0,i/4.0) + 0.5);
}

int
get_offset (int i)
{
  return floor(get_quant(i)*0.375 + 0.5);
}

int
get_factor (int i)
{
  return (1U<<31)/(i+1);
}

int
main (int argc, char *argv[])
{
  int i;

  printf("\n");
  printf("#include <schro/schro-stdint.h>\n");
  printf("\n");

  /* schro_table_offset */
  printf("int16_t schro_table_offset[61] = {\n");
  for(i=0;i<60;i+=4) {
    printf("  %5d, %5d, %5d, %5d,\n",
        get_offset(i),
        get_offset(i+1),
        get_offset(i+2),
        get_offset(i+3));
  }
  printf("  %5d\n", get_offset(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("int16_t schro_table_quant[61] = {\n");
  for(i=0;i<60;i+=4) {
    printf("  %5d, %5d, %5d, %5d,\n",
        get_quant(i),
        get_quant(i+1),
        get_quant(i+2),
        get_quant(i+3));
  }
  printf("  %5d\n", get_quant(i));
  printf("};\n");
  printf("\n");

  /* schro_table_quant */
  printf("uint32_t schro_table_division_factor[1024] = {\n");
  for(i=0;i<1020;i+=4) {
    printf("  %10u, %10u, %10u, %10u,\n",
        get_factor(i),
        get_factor(i+1),
        get_factor(i+2),
        get_factor(i+3));
  }
  printf("  %10u, %10u, %10u, %10u\n",
      get_factor(i),
      get_factor(i+1),
      get_factor(i+2),
      get_factor(i+3));
  printf("};\n");

  return 0;
}

