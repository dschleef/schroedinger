
#ifndef __SCHRO_ASYNC_H__
#define __SCHRO_ASYNC_H__

#include <schroedinger/schro.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _SchroAsync SchroAsync;
typedef struct _SchroThread SchroThread;



SchroAsync *schro_async_new(int n_threads);
void schro_async_free (SchroAsync *async);

void schro_async_run (SchroAsync *async, int slot, void (*func)(void *), void *ptr);
int schro_async_get_idle_thread (SchroAsync *async);
void schro_async_wait_all (SchroAsync *async);

#ifdef __cplusplus
}
#endif

#endif

