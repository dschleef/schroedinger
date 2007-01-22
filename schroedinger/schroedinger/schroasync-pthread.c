
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <pthread.h>
#include <string.h>

enum {
  STATE_IDLE,
  STATE_START,
  STATE_BUSY,
  STATE_DONE,
  STATE_STOP,
};

struct _SchroAsync {
  int n_threads;

  pthread_cond_t cond;
  pthread_mutex_t mutex;

  SchroThread *threads;
};

struct _SchroThread {
  pthread_t pthread;
  pthread_cond_t cond;
  SchroAsync *async;
  int state;
  void (*callback) (void *);
  void (*complete) (void *);
  void *priv;
  int index;
};

static void * schro_thread_main (void *ptr);

SchroAsync *
schro_async_new(int n_threads)
{
  SchroAsync *async;
  pthread_attr_t attr;
  pthread_mutexattr_t mutexattr;
  pthread_condattr_t condattr;
  int i;

  async = malloc(sizeof(SchroAsync));

  SCHRO_ERROR("%d", n_threads);
  async->n_threads = n_threads;
  async->threads = malloc(sizeof(SchroThread) * n_threads);

  pthread_mutexattr_init (&mutexattr);
  pthread_mutex_init (&async->mutex, &mutexattr);
  pthread_condattr_init (&condattr);
  pthread_cond_init (&async->cond, &condattr);

  pthread_attr_init (&attr);
  //pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
  
  pthread_mutex_lock (&async->mutex);
  for(i=0;i<n_threads;i++){
    SchroThread *thread = async->threads + i;

    thread->async = async;
    thread->index = i;
    pthread_cond_init (&async->threads[i].cond, &condattr);
    pthread_create (&async->threads[i].pthread, &attr,
        schro_thread_main, async->threads + i);
    pthread_mutex_lock (&async->mutex);
  }
  pthread_mutex_unlock (&async->mutex);

  pthread_attr_destroy (&attr);
  pthread_mutexattr_destroy (&mutexattr);
  pthread_condattr_destroy (&condattr);

  return async;
}

void
schro_async_free (SchroAsync *async)
{
  int i;

  schro_async_wait_all (async);

  pthread_mutex_lock (&async->mutex);
  for(i=0;i<async->n_threads;i++){
    async->threads[i].callback = NULL;
    async->threads[i].state = STATE_STOP;
    pthread_cond_signal (&async->threads[i].cond);
  }
  pthread_mutex_unlock (&async->mutex);

  for(i=0;i<async->n_threads;i++){
    void *ignore;
    pthread_join (async->threads[i].pthread, &ignore);
    pthread_cond_destroy (&async->threads[i].cond);
  }

  free(async->threads);
  free(async);
}

static void
schro_async_handle_done (SchroAsync *async)
{
  int i;

  for(i=0;i<async->n_threads;i++){
    if (async->threads[i].state == STATE_DONE) {
      SCHRO_ERROR("thread %d: completing", i);
      if (async->threads[i].complete) {
        async->threads[i].complete (async->threads[i].priv);
      }
      async->threads[i].state = STATE_IDLE;
    }
  }
}

void
schro_async_run (SchroAsync *async, int i, void (*func)(void *),
    void (*complete)(void *), void *ptr)
{
  pthread_mutex_unlock (&async->mutex);
  if (async->threads[i].state != STATE_IDLE) {
    SCHRO_ASSERT(0);
    /* FIXME bug */
    return;
  }

  async->threads[i].callback = func;
  async->threads[i].priv = ptr;
  async->threads[i].state = STATE_START;

  SCHRO_ERROR("starting thread %d", i);
  pthread_cond_signal (&async->threads[i].cond);
  pthread_mutex_unlock (&async->mutex);
}

void
schro_async_wait_one (SchroAsync *async)
{
  int i;
  int n;

  SCHRO_ERROR("getting idle thread");
  pthread_mutex_lock (&async->mutex);
  n = 0;
  for(i=0;i<async->n_threads;i++){
    if (async->threads[i].state == STATE_IDLE) {
      n++;
    }
  }

  if (n == async->n_threads) {
    /* all threads are idle, so there's nothing to wait on */
    pthread_mutex_unlock (&async->mutex);
    return;
  }

  SCHRO_ERROR("waiting");
  pthread_cond_wait (&async->cond, &async->mutex);
  schro_async_handle_done (async);
  pthread_mutex_unlock (&async->mutex);
}

int
schro_async_get_idle_thread (SchroAsync *async)
{
  int i;

  SCHRO_ERROR("getting idle thread");
  pthread_mutex_lock (&async->mutex);
  for(i=0;i<async->n_threads;i++){
    if (async->threads[i].state == STATE_IDLE) {
      pthread_mutex_unlock (&async->mutex);
      SCHRO_ERROR("thread %d is idle", i);
      return i;
    }
  }

  SCHRO_ERROR("waiting");
  pthread_cond_wait (&async->cond, &async->mutex);
  schro_async_handle_done (async);

  for(i=0;i<async->n_threads;i++){
    if (async->threads[i].state == STATE_IDLE) {
      pthread_mutex_unlock (&async->mutex);
      SCHRO_ERROR("thread %d is idle", i);
      return i;
    }
  }
  pthread_mutex_unlock (&async->mutex);

  SCHRO_ASSERT(0);
  return -1;
}

/* Call with lock */
static int
schro_async_get_n_busy (SchroAsync *async)
{
  int i;
  int n = 0;

  for(i=0;i<async->n_threads;i++){
    if (async->threads[i].state != STATE_IDLE) {
      n++;
    }
  }
  return n;
}

void
schro_async_wait_all (SchroAsync *async)
{
  int n;

  while (1) {
    pthread_mutex_lock (&async->mutex);
    n = schro_async_get_n_busy (async);

    if (n == 0) {
      pthread_mutex_unlock (&async->mutex);
      return;
    }

    pthread_cond_wait (&async->cond, &async->mutex);
    schro_async_handle_done (async);
  }
}

static void *
schro_thread_main (void *ptr)
{
  SchroThread *thread = ptr;

  while (1) {
    pthread_cond_wait (&thread->cond, &thread->async->mutex);

    SCHRO_ERROR("thread %d: got signal", thread->index);

    if (thread->state == STATE_STOP) {
      pthread_mutex_unlock (&thread->async->mutex);
      SCHRO_ERROR("thread %d: stopping", thread->index);
      return NULL;
    }

    SCHRO_ASSERT (thread->state == STATE_START);
    thread->state = STATE_BUSY;
    pthread_mutex_unlock (&thread->async->mutex);

    SCHRO_ERROR("thread %d: running", thread->index);
    thread->callback (thread->priv);
    SCHRO_ERROR("thread %d: done", thread->index);

    pthread_mutex_lock (&thread->async->mutex);
    thread->state = STATE_DONE;
    pthread_cond_signal (&thread->async->cond);
  }
}


