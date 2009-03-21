
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <glib.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

enum {
  STATE_IDLE,
  STATE_BUSY,
  STATE_STOP
};

struct _SchroAsync {
  int n_threads;
  int n_threads_running;
  int n_idle;

  volatile int n_completed;

  GMutex *mutex;
  GCond *app_cond;
  GCond *thread_cond;

  SchroThread *threads;

  void (*task_func)(void *);
  void *task_priv;

  SchroAsyncScheduleFunc schedule;
  void *schedule_closure;

  SchroAsyncCompleteFunc complete;
};

struct _SchroThread {
  GThread *thread;
  SchroExecDomain exec_domain;
  SchroAsync *async;
  int state;
  int index;
};

static int domain_key_inited;
static GPrivate *domain_key;

static void * schro_thread_main (void *ptr);

void
schro_async_init (void)
{
  if (!g_thread_supported ()) g_thread_init (NULL);
}

SchroAsync *
schro_async_new(int n_threads,
    SchroAsyncScheduleFunc schedule,
    SchroAsyncCompleteFunc complete,
    void *closure)
{
  SchroAsync *async;
  int i;

  if (n_threads == 0) {
    char *s;

    s = getenv ("SCHRO_THREADS");
    if (s && s[0]) {
      char *end;
      int n;
      n = strtoul (s, &end, 0);
      if (end[0] == 0) {
        n_threads = n;
      }
    }
    if (n_threads == 0) {
#if defined(_WIN32)
      const char *s = getenv("NUMBER_OF_PROCESSORS");
      if (s) {
        n_threads = atoi(s);
      }
#elif defined(__APPLE__)
      {
        int    mib[]    = {CTL_HW, HW_NCPU};
        size_t dataSize =  sizeof(int);

        if (sysctl(mib, 2, &n_threads, &dataSize, NULL, 0)) {
          n_threads = 0;
        }
      }
#else
      n_threads = sysconf(_SC_NPROCESSORS_CONF);
#endif
    }
    if (n_threads == 0) {
      n_threads = 1;
    }
  }
  async = schro_malloc0 (sizeof(SchroAsync));

  SCHRO_DEBUG("%d", n_threads);
  async->n_threads = n_threads;
  async->threads = schro_malloc0 (sizeof(SchroThread) * (n_threads + 1));

  async->schedule = schedule;
  async->schedule_closure = closure;
  async->complete = complete;

  async->mutex = g_mutex_new ();
  async->app_cond = g_cond_new ();
  async->thread_cond = g_cond_new ();

  if (!domain_key_inited) {
    domain_key = g_private_new (NULL);
    domain_key_inited = TRUE;
  }

  g_mutex_lock (async->mutex);

  for(i=0;i<n_threads;i++){
    SchroThread *thread = async->threads + i;
    GError *error = NULL;

    thread->async = async;
    thread->index = i;
    thread->exec_domain = SCHRO_EXEC_DOMAIN_CPU;
    async->threads[i].thread = g_thread_create (schro_thread_main,
        async->threads + i, TRUE, &error);
    g_mutex_lock (async->mutex);
  }
  g_mutex_unlock (async->mutex);

  return async;
}

void
schro_async_free (SchroAsync *async)
{
  int i;

  g_mutex_lock (async->mutex);
  for(i=0;i<async->n_threads;i++){
    async->threads[i].state = STATE_STOP;
  }
  while(async->n_threads_running > 0) {
    g_cond_signal (async->thread_cond);
    g_cond_wait (async->app_cond, async->mutex);
  }
  g_mutex_unlock (async->mutex);

  for(i=0;i<async->n_threads;i++){
    g_thread_join (async->threads[i].thread);
  }

  g_mutex_free (async->mutex);
  g_cond_free (async->app_cond);
  g_cond_free (async->thread_cond);

  schro_free(async->threads);
  schro_free(async);
}

void
schro_async_start (SchroAsync *async)
{
  g_cond_broadcast (async->thread_cond);
}

void
schro_async_stop (SchroAsync *async)
{
  g_mutex_lock (async->mutex);
  while(async->n_idle < async->n_threads_running) {
    g_cond_wait (async->app_cond, async->mutex);
  }
  g_mutex_unlock (async->mutex);
}

#ifdef unused
void
schro_async_run_locked (SchroAsync *async, void (*func)(void *), void *ptr)
{
  SCHRO_ASSERT(async->task_func == NULL);

  async->task_func = func;
  async->task_priv = ptr;

  g_cond_signal (async->thread_cond);
}
#endif

void
schro_async_run_stage_locked (SchroAsync *async, SchroAsyncStage *stage)
{
  SCHRO_ASSERT(async->task_func == NULL);

  async->task_func = stage->task_func;
  async->task_priv = stage;

  g_cond_signal (async->thread_cond);
}

#ifdef unused
int schro_async_get_num_completed (SchroAsync *async)
{
  return async->n_completed;
}
#endif

static void
schro_async_dump (SchroAsync *async)
{
  int i;
  for(i=0;i<async->n_threads;i++){
    SchroThread *thread = async->threads + i;
    const char *states[] = { "idle", "busy", "stopped" };

    SCHRO_WARNING ("thread %d: %s", i, states[thread->state]);
  }
}

int
schro_async_wait_locked (SchroAsync *async)
{
  GTimeVal ts;
  int ret;

  g_get_current_time (&ts);
  g_time_val_add (&ts, 1000000);
  ret = g_cond_timed_wait (async->app_cond, async->mutex, &ts);
  if (ret != 0) {
    int i;
    for(i=0;i<async->n_threads;i++){
      if (async->threads[i].state != 0) break;
    }
    if (i == async->n_threads) {
      SCHRO_WARNING("timeout.  deadlock?");
      schro_async_dump (async);
      return FALSE;
    }
  }
  return TRUE;
}

#ifdef unused
void
schro_async_wait_one (SchroAsync *async)
{
  g_mutex_lock (async->mutex);
  if (async->n_completed > 0) {
    g_mutex_unlock (async->mutex);
    return;
  }

  g_cond_wait (async->app_cond, async->mutex);
  g_mutex_unlock (async->mutex);
}
#endif

#ifdef unused
void
schro_async_wait (SchroAsync *async, int min_waiting)
{
  if (min_waiting < 1) min_waiting = 1;

  g_mutex_lock (async->mutex);
  if (async->n_completed > 0) {
    g_mutex_unlock (async->mutex);
    return;
  }

  g_cond_wait (async->app_cond, async->mutex);
  g_mutex_unlock (async->mutex);
}
#endif

static void *
schro_thread_main (void *ptr)
{
  void (*func)(void *);
  void *priv;
  SchroThread *thread = ptr;
  SchroAsync *async = thread->async;
  int ret;

  /* thread starts with async->mutex locked */

  g_private_set (domain_key, (void *)(unsigned long)thread->exec_domain);

  async->n_threads_running++;
  thread->state = STATE_IDLE;
  while (1) {
    switch (thread->state) {
      case STATE_IDLE:
        async->n_idle++;
        SCHRO_DEBUG("thread %d: idle", thread->index);
        g_cond_wait (async->thread_cond, async->mutex);
        SCHRO_DEBUG("thread %d: got signal", thread->index);
        async->n_idle--;
        if (thread->state == STATE_IDLE) {
          thread->state = STATE_BUSY;
        }
        break;
      case STATE_STOP:
        g_cond_signal (async->app_cond);
        async->n_threads_running--;
        g_mutex_unlock (async->mutex);
        SCHRO_DEBUG("thread %d: stopping", thread->index);
        return NULL;
      case STATE_BUSY:
        ret = async->schedule (async->schedule_closure, thread->exec_domain);
        /* FIXME ignoring ret */
        if (!async->task_func) {
          thread->state = STATE_IDLE;
          break;
        }

        thread->state = STATE_BUSY;
        func = async->task_func;
        priv = async->task_priv;
        async->task_func = NULL;

        if (async->n_idle > 0) {
          g_cond_signal (async->thread_cond);
        }
        g_mutex_unlock (async->mutex);

        SCHRO_DEBUG("thread %d: running", thread->index);
        func (priv);
        SCHRO_DEBUG("thread %d: done", thread->index);

        g_mutex_lock (async->mutex);

        async->complete (priv);

        g_cond_signal (async->app_cond);
#ifdef HAVE_CUDA
        /* FIXME */
        /* This is required because we don't have a better mechanism
         * for indicating to threads in other exec domains that it is
         * their turn to run.  It's mostly harmless, although causes
         * a lot of unnecessary wakeups in some cases. */
        g_cond_broadcast (async->thread_cond);
#endif

        break;
      default:
        SCHRO_ASSERT(0);
    }
  }
}

void schro_async_lock (SchroAsync *async)
{
  g_mutex_lock (async->mutex);
}

void schro_async_unlock (SchroAsync *async)
{
  g_mutex_unlock (async->mutex);
}

void schro_async_signal_scheduler (SchroAsync *async)
{
  g_cond_broadcast (async->thread_cond);
}

void
schro_async_add_cuda (SchroAsync *async)
{
  SchroThread *thread;
  int i;
  GError *error = NULL;

  g_mutex_lock (async->mutex);

  /* We allocated a spare thread structure just for this case. */
  async->n_threads++;
  i = async->n_threads - 1;

  thread = async->threads + i;
  memset (thread, 0, sizeof(SchroThread));

  thread->async = async;
  thread->index = i;
  thread->exec_domain = SCHRO_EXEC_DOMAIN_CUDA;
  async->threads[i].thread = g_thread_create (schro_thread_main,
      async->threads + i, TRUE, &error);
  g_mutex_lock (async->mutex);
  g_mutex_unlock (async->mutex);
}

SchroExecDomain
schro_async_get_exec_domain (void)
{
  void *domain;
  domain = g_private_get (domain_key);
  return (int)(unsigned long)domain;
}

SchroMutex *schro_mutex_new (void)
{
  return (SchroMutex *)g_mutex_new();
}

void schro_mutex_lock (SchroMutex *mutex)
{
  g_mutex_lock((GMutex *)mutex);
}

void
schro_mutex_unlock (SchroMutex *mutex)
{
  g_mutex_unlock((GMutex *)mutex);
}

void
schro_mutex_free (SchroMutex *mutex)
{
  g_mutex_free((GMutex *)mutex);
}

