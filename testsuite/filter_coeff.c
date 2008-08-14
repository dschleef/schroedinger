
#include <stdio.h>
#include <math.h>



double
sinc (double x) 
{
  if (x==0) return 1;
  return sin(x)/x;
}

double
envelope (double x, double zp)
{
#if 0
  x /= zp;

  if (x > 1) return 0;
  return 1-x*x;
#endif
  x /= zp;
  if (x < -1 || x > 1) return 0;
  return sinc(M_PI*x);
}

int main (int argc, char *argv[])
{
  int i;
  double x;
  double cutoff;
  double t[100];
  double sum = 0;
  double offset = 0.25;
  double w;
  double center;
  int n_taps;
  int j;

  cutoff = 1.0;
  offset = 0.25;
  w = 4;
  for(i = 0; i < 50; i++) {
    offset = i/50.0;
    //w = 1 + i*0.1;

    n_taps = floor(w*2 + 1);
    n_taps = (n_taps + 1)&(~1);

    center = offset;
    sum = 0;
    for(j=0;j<n_taps;j++){

      x = j - offset - n_taps/2 + 1;

      t[j] = sinc(cutoff*x*M_PI)*envelope(x,w);
      sum += t[j];
    }

    printf("w %g  offset %g taps %d:\n", w, offset, n_taps);
    for(j=0;j<n_taps;j++){
      printf("%d: %0.2f\n", j, 64*t[j]/sum);
    }
  }

  return 0;
}

