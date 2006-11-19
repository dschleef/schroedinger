
#ifndef __SCHRO_ASYNC_H__
#define __SCHRO_ASYNC_H__

#include <schroedinger/schro.h>

typedef struct _SchroAsync SchroAsync;
typedef struct _SchroThread SchroThread;
typedef struct _SchroJob SchroJob;

struct _SchroJob {
  SchroJob *next;

  void (*func) (void *);
  void *priv;
};

SchroAsync *schro_async_new(int n_threads);
void schro_async_free (SchroAsync *async);

void schro_async_add_job (SchroAsync *async, SchroJob *job);
void schro_async_wait_for_completion (SchroAsync *async);
SchroJob *schro_async_get_job (SchroAsync *async);

#endif

