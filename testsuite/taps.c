
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrofilter.h>
#include <schroedinger/schro.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <math.h>


double filter[100];

double sinc(double x)
{
  if (x == 0) return 1.0;
  return sin(x)/x;
}

double env(double x, double width)
{
  return (x + width/2) * (x - width/2);
}

void
create_filter (int n)
{
  int i;
  double sum;
  double x;

  for(i=0;i<n;i++) {
    x = i - (n-1)*0.5;
    filter[i] = sinc(M_PI*x/((n+1)/(n*0.5))) * env(x,n+3);
  }

  sum = 0;
  for(i=0;i<n;i++) {
    sum += filter[i];
  }

  for(i=0;i<n;i++){
    filter[i] = floor(0.5 + 256.0*filter[i]/sum);
  }
 
  printf("%d: ",n);
  sum = 0;
  for(i=0;i<n;i++){
    printf(" %g", filter[i]);
    sum += filter[i];
  }
  printf(" (%g)", sum);
  printf("\n");


}


int
main (int argc, char *argv[])
{

  create_filter(2);
  create_filter(4);
  create_filter(6);
  create_filter(8);
  create_filter(10);
  create_filter(12);

  return 0;
}

