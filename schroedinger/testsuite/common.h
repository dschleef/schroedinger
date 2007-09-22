
#ifndef __COMMON_H__
#define __COMMON_H__

#include <math.h>

double sgn(double x);
double random_std (void);
double random_triangle (void);
int gain_to_quant_index (double x);
double sum_f64 (double *a, int n);
double multsum_f64 (double *a, double *b, int n);


#endif

