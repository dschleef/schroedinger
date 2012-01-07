
#include <stdio.h>
#include <math.h>
#include <schroedinger/schro.h>


//#define MAX(a,b) (((a)>(b))?(a):(b))
#define ABS(a) (((a)>0)?(a):(-a))

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

#if 0
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
#endif

typedef struct _Filter2D Filter2D;
struct _Filter2D {
  int n_taps;
  int shift;
  double w;

  double *d_taps;
  int *i_taps;

};

#define FILTER2D_D_TAP(filter,x,y) ((filter)->d_taps[(x) + (y)*(filter)->n_taps])
#define FILTER2D_I_TAP(filter,x,y) ((filter)->i_taps[(x) + (y)*(filter)->n_taps])

Filter2D *
filter2d_new (int n_taps, int shift)
{
  Filter2D *f;

  f = schro_malloc0 (sizeof(Filter2D));
  f->n_taps = n_taps;
  f->shift = shift;
  f->d_taps = schro_malloc0 (n_taps * n_taps * sizeof(double));
  f->i_taps = schro_malloc0 (n_taps * n_taps * sizeof(int));

  return f;
}

double
filter2d_calc_sum (Filter2D *f)
{
  int i,j;
  double sum = 0;

  for(i=0;i<f->n_taps;i++){
    for(j=0;j<f->n_taps;j++){
      sum += FILTER2D_D_TAP (f,i,j);
    }
  }
  return sum;
}

int
filter2d_calc_int_sum (Filter2D *f)
{
  int i,j;
  int sum = 0;

  for(i=0;i<f->n_taps;i++){
    for(j=0;j<f->n_taps;j++){
      sum += FILTER2D_I_TAP (f,i,j);
    }
  }
  return sum;
}

int
filter2d_calc_int_abs_sum (Filter2D *f)
{
  int i,j;
  int sum = 0;

  for(i=0;i<f->n_taps;i++){
    for(j=0;j<f->n_taps;j++){
      sum += ABS(FILTER2D_I_TAP (f,i,j));
    }
  }
  return sum;
}

void
filter2d_generate_sinc (Filter2D *filter, double cutoff, double w)
{
  int j,k;
  double x;
  double center_x = (filter->n_taps - 1.0) * 0.5;
  double center_y = (filter->n_taps - 1.0) * 0.5;

  for(j=0;j<filter->n_taps;j++){
    for(k=0;k<filter->n_taps;k++){
      x = sqrt((j-center_x)*(j-center_x)+(k-center_y)*(k-center_y));

      FILTER2D_D_TAP(filter,j,k) = sinc(cutoff*x*M_PI)*envelope(x,w);
    }
  }
}

void
filter2d_quantise (Filter2D *filter, double adjust)
{
  int j,k;
  double sum;
  int target = 1<<filter->shift;
  
  sum = filter2d_calc_sum (filter);
  for(j=0;j<filter->n_taps;j++){
    for(k=0;k<filter->n_taps;k++){
      FILTER2D_I_TAP (filter,j,k) =
        floor(0.5 + target*FILTER2D_D_TAP(filter,j,k)/sum + adjust);
    }
  }

}

void
filter2d_dump (Filter2D *filter)
{
  int j,k;

  for(j=0;j<filter->n_taps;j++){
    for(k=0;k<filter->n_taps;k++){
      if (FILTER2D_I_TAP(filter,j,k) == 0) {
        printf("    . ");
      } else {
        printf("%5d ", FILTER2D_I_TAP(filter,j,k));
      }
    }
    printf("\n");
  }

}

double
filter2d_adjust (Filter2D *filter)
{
  int isum;
  double lo_adj, hi_adj, mid_adj;
  int lo_isum, hi_isum, mid_isum;
  int i;
  int target = 1<<filter->shift;

  filter2d_quantise (filter, 0.0);
  isum = filter2d_calc_int_sum (filter);

  if (isum == target) return 0;
  if (isum > target) {
    hi_adj = 0.0;
    hi_isum = isum;
    lo_adj = -0.5;
    filter2d_quantise (filter, lo_adj);
    lo_isum = filter2d_calc_int_sum (filter);
    if (lo_isum == target) {
      return lo_adj;
    }
  } else {
    lo_adj = 0.0;
    lo_isum = isum;
    hi_adj = 0.5;
    filter2d_quantise (filter, hi_adj);
    hi_isum = filter2d_calc_int_sum (filter);
    if (hi_isum == target) {
      return hi_adj;
    }
  }

  for(i=0;i<10;i++){
    //printf("[%g,%g] -> [%d,%d]\n", lo_adj, hi_adj, lo_isum, hi_isum);

    mid_adj = 0.5*(hi_adj + lo_adj);
    filter2d_quantise (filter, mid_adj);
    mid_isum = filter2d_calc_int_sum (filter);

    if (mid_isum == target) return mid_adj;
    if (mid_isum < target) {
      lo_adj = mid_adj;
      lo_isum = mid_isum;
    } else {
      hi_adj = mid_adj;
      hi_isum = mid_isum;
    }

  }

  printf("unresolved sum=%d\n", mid_isum);
  return mid_adj;
}


int main (int argc, char *argv[])
{
  Filter2D *filter;
  int i;
  double cutoff;
  double offset = 0.25;
  double w;
  int isum;
  int asum;

  schro_init ();

  filter = filter2d_new (9, 12);

  cutoff = 0.5;
  offset = 0.25;
  w = 4;
  for(i = 0; i <= 50; i++) {
    offset = i/50.0;
    w = 1 + i*0.1;

    filter2d_generate_sinc (filter, cutoff, w);

    filter2d_calc_sum (filter);

    printf("w %g  offset %g:\n", w, offset);

    filter2d_quantise (filter, 0.0);

    filter2d_adjust (filter);
    isum = filter2d_calc_int_sum (filter);
    asum = filter2d_calc_int_abs_sum (filter);
    filter2d_dump (filter);

    printf("sum = %d  abs sum = %d\n",isum, asum);
  }

  return 0;
}

