
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <pthread.h>
#include <string.h>


typedef struct _SchroAsync SchroAsync;
typedef struct _SchroThread SchroThread;

struct _SchroAsync {
  int dummy;
};

SchroAsync *
schro_async_new(int n_threads)
{
  SchroAsync *async;

  async = malloc(sizeof(SchroAsync));

  return async;
}

void
schro_async_run (SchroAsync *async, void (*func)(void *), void *ptr)
{
  func(ptr);
}

void
schro_async_wait_for_ready (SchroAsync *async)
{

}

