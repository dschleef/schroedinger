
#ifndef __SCHRO_ASYNC_H__
#define __SCHRO_ASYNC_H__

#include <schroedinger/schro.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroAsync SchroAsync;
typedef struct _SchroThread SchroThread;
typedef struct _SchroAsyncTask SchroAsyncTask;


SchroAsync *schro_async_new(int n_threads);
void schro_async_free (SchroAsync *async);

void schro_async_run (SchroAsync *async, void (*func)(void *), void *ptr);
int schro_async_get_num_completed (SchroAsync *async);
int schro_async_get_num_waiting (SchroAsync *async);
void schro_async_wait (SchroAsync *async, int min_waiting);
void schro_async_wait_one (SchroAsync *async);
void *schro_async_pull (SchroAsync *async);

SCHRO_END_DECLS

#endif

