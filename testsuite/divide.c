
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <schroedinger/schro.h>
#include <schroedinger/schrounpack.h>


int
main (int argc, char *argv[])
{
  int i;
  int a, b;

  schro_init();

  for(i=-32768;i<32768;i++){
    a = schro_divide(i,3);
    //b = (i*21845 + 10922)>>16;
    b = schro_divide3(i);
    if (a != b) printf("%d: %d %d %c\n", i, a, b, (a==b)?' ':'*');
  }

  exit(0);
}



