
#ifndef __SCHRO_ASYNC_H__
#define __SCHRO_ASYNC_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schrodomain.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroAsync SchroAsync;
typedef struct _SchroThread SchroThread;
typedef struct _SchroAsyncTask SchroAsyncTask;

#ifdef SCHRO_ENABLE_UNSTABLE_API

typedef int (*SchroAsyncScheduleFunc)(void *, SchroExecDomain exec_domain);
typedef void (*SchroAsyncCompleteFunc)(void *);

SchroAsync * schro_async_new(int n_threads,
    SchroAsyncScheduleFunc schedule,
    SchroAsyncCompleteFunc complete,
    void *closure);
void schro_async_free (SchroAsync *async);

void schro_async_run_locked (SchroAsync *async, void (*func)(void *), void *ptr);
int schro_async_get_num_completed (SchroAsync *async);
void schro_async_wait_one (SchroAsync *async);
int schro_async_wait_locked (SchroAsync *async);
void schro_async_wait (SchroAsync *async, int min_waiting);
void *schro_async_pull (SchroAsync *async);
void * schro_async_pull_locked (SchroAsync *async);
void schro_async_signal_scheduler (SchroAsync *async);
void schro_async_lock (SchroAsync *async);
void schro_async_unlock (SchroAsync *async);
SchroExecDomain schro_async_get_exec_domain (void);

void schro_async_add_cuda (SchroAsync *async);

#endif

SCHRO_END_DECLS

#endif

