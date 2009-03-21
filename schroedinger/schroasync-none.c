
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <schroedinger/schrodomain.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

struct _SchroAsync {
  int n_idle;

  volatile int n_completed;

  void *done_priv;

  SchroAsyncScheduleFunc schedule;
  SchroAsyncCompleteFunc complete;
  void *schedule_closure;

  void (*task_func)(void *);
  void *task_priv;
};

struct _SchroThread {
  SchroAsync *async;
  int state;
  int index;
};

void
schro_async_init (void)
{

}

SchroAsync *
schro_async_new(int n_threads,
    SchroAsyncScheduleFunc schedule,
    SchroAsyncCompleteFunc complete, void *closure)
{
  SchroAsync *async;

  async = schro_malloc0 (sizeof(SchroAsync));

  async->schedule = schedule;
  async->schedule_closure = closure;
  async->complete = complete;

  return async;
}

void
schro_async_free (SchroAsync *async)
{
  schro_free(async);
}

void
schro_async_start (SchroAsync *async)
{
}

void
schro_async_stop (SchroAsync *async)
{
}

#ifdef unused
void
schro_async_run_locked (SchroAsync *async, void (*func)(void *), void *ptr)
{
  SCHRO_ASSERT(async->task_func == NULL);

  async->task_func = func;
  async->task_priv = ptr;
}
#endif

void
schro_async_run_stage_locked (SchroAsync *async, SchroAsyncStage *stage)
{
  SCHRO_ASSERT(async->task_func == NULL);

  async->task_func = stage->task_func;
  async->task_priv = stage;
}

#ifdef unused
int schro_async_get_num_completed (SchroAsync *async)
{
  if (async->done_priv) return 1;
  return 0;
}
#endif

void *schro_async_pull (SchroAsync *async)
{
  void *ptr;

  if (!async->done_priv) {
    return NULL;
  }

  ptr = async->done_priv;
  async->done_priv = NULL;
  async->n_completed--;

  return ptr;
}

void *
schro_async_pull_locked (SchroAsync *async)
{
  void *ptr;

  if (!async->done_priv) {
    return NULL;
  }

  ptr = async->done_priv;
  async->done_priv = NULL;
  async->n_completed--;

  return ptr;
}

int
schro_async_wait_locked (SchroAsync *async)
{
  async->schedule (async->schedule_closure, SCHRO_EXEC_DOMAIN_CPU);
  if (async->task_func) {
    async->task_func (async->task_priv);
    async->task_func = NULL;
    async->complete (async->task_priv);
  }
  return TRUE;
}

#ifdef unused
void
schro_async_wait_one (SchroAsync *async)
{
  async->schedule (async->schedule_closure, SCHRO_EXEC_DOMAIN_CPU);
  if (async->task_func) {
    async->task_func (async->task_priv);
    async->task_func = NULL;
    async->complete (async->task_priv);
  }
}
#endif

#ifdef unused
void
schro_async_wait (SchroAsync *async, int min_waiting)
{
  async->schedule (async->schedule_closure, SCHRO_EXEC_DOMAIN_CPU);
  if (async->task_func) {
    async->task_func (async->task_priv);
    async->task_func = NULL;
    async->complete (async->task_priv);
  }
}
#endif

void schro_async_lock (SchroAsync *async)
{
}

void schro_async_unlock (SchroAsync *async)
{
}

void schro_async_signal_scheduler (SchroAsync *async)
{
}

SchroMutex *
schro_mutex_new (void)
{
  return NULL;
}

void
schro_mutex_lock (SchroMutex *mutex)
{
}

void
schro_mutex_unlock (SchroMutex *mutex)
{
}

void
schro_mutex_free (SchroMutex *mutex)
{
}

