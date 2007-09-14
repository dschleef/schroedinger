
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schroarith.h>
#include <schroedinger/schrotables.h>




#define PRINT_OFFSET(prefix,str,member) \
  printf("    \".set " # prefix "_" #member ", 0x%x\\n\"\n", \
      (int)(&((str *)0)->member))


int
main (int argc, char *argv[])
{
  printf("__asm__ __volatile__ (\n");
  PRINT_OFFSET(a,SchroArith,range);
  PRINT_OFFSET(a,SchroArith,range_size);
  PRINT_OFFSET(a,SchroArith,probabilities);
  PRINT_OFFSET(a,SchroArith,lut);
  printf(");\n");

  return 0;
}


