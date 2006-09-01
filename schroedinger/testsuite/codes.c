
#include <stdio.h>

#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>



void
schro_arith_context_encode_bit (SchroArith *arith, int i, int value)
{
  printf(" %d", value);
}

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  printf("unsigned int\n");
  for(i=0;i<11;i++) {
    printf("%3d:", i);
    schro_arith_context_encode_uint(NULL,0,0,i);
    printf("\n");
  }
  printf("\n");

  printf("signed int\n");
  for(i=-5;i<6;i++) {
    printf("%3d:", i);
    schro_arith_context_encode_sint(NULL,0,0,0,i);
    printf("\n");
  }
  printf("\n");

  return 0;

}



