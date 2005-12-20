
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <carid/caridarith.h>


unsigned int division_factor[1024];

static void
_carid_arith_division_factor_init (void)
{
  static int inited = 0;

  if (!inited) {
    int i;
    for(i=0;i<1024;i++){
      division_factor[i] = (1<<31)/(i+1);
    }
  }
}

CaridArith
carid_arith_new (void)
{
  CaridArith *arith;
  
  arith = malloc (sizeof(*arith));
  memset (arith, 0, sizeof(*arith));

  _carid_arith_division_factor_init();
}

void
carid_arith_free (CaridArith *arith)
{
  free(arith);
}

void
carid_arith_init (CaridArith *arith)
{
  arith->low = 0;
  arith->high = 0xffff;
  arith->underflow = 0;
  arith->code = 0;
}



void
carid_arith_context_halve_counts (CaridArith *arith, int i)
{
  arith->contexts[i].count0 >>= 1;
  arith->contexts[i].count0++;
  arith->contexts[i].count1 >>= 1;
  arith->contexts[i].count1++;
}

void
carid_arith_context_halve_all_counts (CaridArith *arith)
{
  int i;
  for(i=0;i<arith->n_contexts;i++) {
    arith->contexts[i].count0 >>= 1;
    arith->contexts[i].count0++;
    arith->contexts[i].count1 >>= 1;
    arith->contexts[i].count1++;
  }
}

void
carid_arith_context_update (CaridArith *arith, int i, int value)
{
  if (value) {
    arith->contexts[i].count1++;
  } else {
    arith->contexts[i].count0++;
  }
  if (arith->contexts[i].count0 + arith->contexts[i].count0 >= 1024) {
    carid_arith_halve_counts (arith, i);
  }
}

