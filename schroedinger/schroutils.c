
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schroutils.h>
#include <schroedinger/schrotables.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schro-stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#undef USE_MMAP
#ifdef USE_MMAP
#include <sys/mman.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef USE_MMAP
void *
schro_malloc (int size)
{
  void *ptr;

  ptr = malloc (size);
  SCHRO_DEBUG("alloc %p %d", ptr, size);

  return ptr;
}

void *
schro_malloc0 (int size)
{
  void *ptr;

  ptr = malloc (size);
  memset (ptr, 0, size);
  SCHRO_DEBUG("alloc %p %d", ptr, size);

  return ptr;
}

#ifdef unused
void *
schro_realloc (void *ptr, int size)
{
  ptr = realloc (ptr, size);
  SCHRO_DEBUG("realloc %p %d", ptr, size);

  return ptr;
}
#endif

void
schro_free (void *ptr)
{
  SCHRO_DEBUG("free %p", ptr);
  free (ptr);
}
#else
void *
schro_malloc (int size)
{
  void *ptr;
  int rsize;

  rsize = ROUND_UP_POW2(size + 16, 12);
  ptr = mmap(NULL, rsize + 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  SCHRO_ASSERT(ptr != MAP_FAILED);

  mprotect (ptr, 4096, PROT_NONE);
  mprotect (OFFSET(ptr, 4096 + rsize), 4096, PROT_NONE);

  SCHRO_DEBUG("alloc %p %d", ptr, size);

  *(int *)OFFSET(ptr, 4096) = rsize;

  //return OFFSET(ptr, rsize-size);
  return OFFSET(ptr, 4096 + rsize - size);
}

void *
schro_malloc0 (int size)
{
  return schro_malloc (size);
}

void *
schro_realloc (void *ptr, int size)
{
  SCHRO_ASSERT(size <= 0);

  return ptr;
}

void
schro_free (void *ptr)
{
  unsigned long page = ((unsigned long)ptr) & ~(4095);
  int rsize;

  rsize = *(int *)page;

  munmap((void *)(page - 4096), rsize + 8192);
}
#endif

int
muldiv64 (int a, int b, int c)
{
  int64_t x;

  x = a;
  x *= b;
  x /= c;

  return (int)x;
}

int
schro_utils_multiplier_to_quant_index (double x)
{
  return CLAMP(rint(log(x)/log(2)*4.0),0,60);
}


static int
__schro_dequantise (int q, int quant_factor, int quant_offset)
{
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}

int
schro_dequantise (int q, int quant_factor, int quant_offset)
{
  return __schro_dequantise(q,quant_factor,quant_offset);
}

static int
__schro_quantise (int value, int quant_factor, int quant_offset)
{
  unsigned int x;

  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)<<2;
    if (x < quant_factor) {
      x = 0;
    } else {
      x /= quant_factor;
    }
    value = -x;
  } else {
    x = value<<2;
    if (x < quant_factor) {
      x = 0;
    } else {
      x /= quant_factor;
    }
    value = x;
  }
  return value;
}

int
schro_quantise (int value, int quant_factor, int quant_offset)
{
  return __schro_quantise (value, quant_factor, quant_offset);
}

void
schro_quantise_s16 (int16_t *dest, int16_t *src, int quant_factor,
    int quant_offset, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = __schro_quantise (src[i], quant_factor, quant_offset);
    src[i] = __schro_dequantise (dest[i], quant_factor, quant_offset);
  }
}

void
schro_quantise_s16_table (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;
  int16_t *table;
  
  table = schro_tables_get_quantise_table(quant_index);

  table += 32768;

  for(i=0;i<n;i++){
    dest[i] = table[src[i]];
  }
}

void
schro_dequantise_s16_table (int16_t *dest, int16_t *src, int quant_index,
    schro_bool is_intra, int n)
{
  int i;
  int16_t *table;
  
  table = schro_tables_get_dequantise_table(quant_index, is_intra);

  table += 32768;

  for(i=0;i<n;i++){
    dest[i] = table[src[i]];
  }
}

void
schro_dequantise_s16 (int16_t *dest, int16_t *src, int quant_factor,
    int quant_offset, int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = __schro_dequantise (src[i], quant_factor, quant_offset);
  }
}

/* log(2.0) */
#define LOG_2 0.69314718055994528623
/* 1.0/log(2.0) */
#define INV_LOG_2 1.44269504088896338700

double
schro_utils_probability_to_entropy (double x)
{
  if (x <= 0 || x >= 1.0) return 0;

  return -(x * log(x) + (1-x) * log(1-x))*INV_LOG_2;
}

double
schro_utils_entropy (double a, double total)
{
  double x;

  if (total == 0) return 0;

  x = a / total;
  return schro_utils_probability_to_entropy (x) * total;
}

void
schro_utils_reduce_fraction (int *n, int *d)
{
  static const int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    91 };
  int i;
  int p;

  SCHRO_DEBUG("reduce %d/%d", *n, *d);
  for(i=0;i<sizeof(primes)/sizeof(primes[0]);i++){
    p = primes[i];
    while (*n % p == 0 && *d % p == 0) {
      *n /= p;
      *d /= p;
    }
    if (*d == 1) break;
  }
  SCHRO_DEBUG("to %d/%d", *n, *d);
}

double
schro_utils_get_time (void)
{
#ifndef _WIN32
  struct timeval tv;

  gettimeofday (&tv, NULL);

  return tv.tv_sec + 1e-6*tv.tv_usec;
#else
  return (double)GetTickCount() / 1000.;
#endif
}

