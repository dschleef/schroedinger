
#include <config.h>

#include <stdio.h>


void ramp(int n);
void test(int n);

int sum;

int
main (int argc, char *argv[])
{
  int i;

  for(i=2;i<=32;i+=2){
    ramp(i);
  }

  sum = 0;
  for(i=0;i<256;i++){
    test(i);
  }
  printf("sum %d\n", sum);

  return 0;
}


void
ramp(int n)
{
  int i;
  int xoff = n/2;

  printf("%d: ", n);
  for(i=0;i<n;i++){
    printf("%d ", 1 + (6*(i+1) + xoff)/(2*xoff + 1));
  }
  printf ("\n");

  printf("%d: ", n);
  for(i=0;i<n;i++){
    printf("%g ", 1.0 + (6.0*(i+1) + xoff)/(2.0*xoff + 1));
  }
  printf ("\n");

  printf ("\n");
}


void
test (int n)
{
  int i;
  int v, ref;

  printf("%d: ", n);
  for(i=0;i<8;i++){
    //v = ((n*i + 4)>>3) + ((n*(8-i) + 4)>>3);
    v = ((n*i)>>3) + ((n*(8-i))>>3);
    ref = ((n*i + n*(8-i) + 4)>>3);
    printf("%d ", v);
    sum += (ref - v) * (ref - v);
  }
  printf ("\n");
}

