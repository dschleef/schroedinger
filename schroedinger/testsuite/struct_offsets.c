
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
  PRINT_OFFSET(a,SchroArith,code);
  PRINT_OFFSET(a,SchroArith,range);
  PRINT_OFFSET(a,SchroArith,value);
  PRINT_OFFSET(a,SchroArith,probability0);
  PRINT_OFFSET(a,SchroArith,count);
  PRINT_OFFSET(a,SchroArith,range_value);
  PRINT_OFFSET(a,SchroArith,division_factor);
  PRINT_OFFSET(a,SchroArith,fixup_shift);
  PRINT_OFFSET(a,SchroArith,contexts);
  PRINT_OFFSET(a,SchroArith,cntr);
  PRINT_OFFSET(a,SchroArith,buffer);
  PRINT_OFFSET(a,SchroArith,offset);
  PRINT_OFFSET(a,SchroArith,nextcode);
  PRINT_OFFSET(a,SchroArith,nextbits);
  PRINT_OFFSET(a,SchroArith,dataptr);
  PRINT_OFFSET(a,SchroArith,maxdataptr);
  PRINT_OFFSET(c,SchroArithContext,count);
  PRINT_OFFSET(c,SchroArithContext,next);
  printf(");\n");

  return 0;
}


