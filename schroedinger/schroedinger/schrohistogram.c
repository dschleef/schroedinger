
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <string.h>
#include <math.h>
#include <schroedinger/schrohistogram.h>


static int
ilogx (int x)
{
  int i = 0;
  if (x < 0) x = -x;
  while (x >= 2<<SCHRO_HISTOGRAM_SHIFT) {
    x >>= 1;
    i++;
  }
  return x + (i << SCHRO_HISTOGRAM_SHIFT);
}

static int
iexpx (int x)
{
  if (x < (1<<SCHRO_HISTOGRAM_SHIFT)) return x;

  return ((1<<SCHRO_HISTOGRAM_SHIFT)|(x&((1<<SCHRO_HISTOGRAM_SHIFT)-1))) << ((x>>SCHRO_HISTOGRAM_SHIFT)-1);
}

static int
ilogx_size (int i)
{
  if (i < (1<<SCHRO_HISTOGRAM_SHIFT)) return 1;
  return 1 << ((i>>SCHRO_HISTOGRAM_SHIFT)-1);
}

double
schro_histogram_get_range (SchroHistogram *hist, int start, int end)
{
  int i;
  int iend;
  int size;
  double x;

  if (start >= end) return 0;

  i = ilogx(start);
  size = ilogx_size(i);
  x = (double)(iexpx(i+1) - start)/size * hist->bins[i];

  i++;
  iend = ilogx(end);
  while (i <= iend) {
    x += hist->bins[i];
    i++;
  }

  size = ilogx_size(iend);
  x -= (double)(iexpx(iend+1) - end)/size * hist->bins[iend];

  return x;
}

void
schro_histogram_table_generate (SchroHistogramTable *table,
    double (*func)(int value, void *priv), void *priv)
{
  int i;
  int j;

  for(i=0;i<SCHRO_HISTOGRAM_SIZE;i++){
    int jmin, jmax;
    double sum;

    jmin = iexpx(i);
    jmax = iexpx(i+1);

    sum = 0;
    for(j=jmin;j<jmax;j++){
      sum += func(j, priv);
    }
    table->weights[i] = sum / ilogx_size(i);
  }
}

double
schro_histogram_apply_table (SchroHistogram *hist,
    SchroHistogramTable *table)
{
  int i;
  double sum;

  sum = 0;
  for(i=0;i<SCHRO_HISTOGRAM_SIZE;i++){
    sum += hist->bins[i] * table->weights[i];
  }

  return sum;
}


void
schro_histogram_init (SchroHistogram *hist)
{
  memset (hist, 0, sizeof(*hist));
}

void
schro_histogram_add (SchroHistogram *hist, int value)
{
  hist->bins[ilogx(value)]++;
  hist->n++;
}

void
schro_histogram_add_array_s16 (SchroHistogram *hist, int16_t *src, int n)
{
  int i;
  for(i=0;i<n;i++){
    hist->bins[ilogx(src[i])]++;
  }
  hist->n+=n;
}

void
schro_histogram_scale (SchroHistogram *hist, int scale)
{
  int i;
  for(i=0;i<SCHRO_HISTOGRAM_SIZE;i++){
    hist->bins[i]*=scale;
  }
  hist->n*=scale;
}



