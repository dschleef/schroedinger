
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <schroedinger/schrodebug.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

enum {
  STATE_IDLE,
  STATE_BUSY,
  STATE_STOP
};

struct _SchroAsync {
  int n_threads;
  int n_threads_running;

  volatile int n_completed;
  volatile int n_waiting;

  pthread_mutex_t mutex;
  pthread_cond_t async_cond;
  pthread_cond_t thread_cond;

  SchroThread *threads;

  SchroAsyncTask *list_first;
  SchroAsyncTask *list_last;

  SchroAsyncTask *done_first;
  SchroAsyncTask *done_last;
};

struct _SchroThread {
  pthread_t pthread;
  SchroAsync *async;
  int state;
  int index;
};

struct _SchroAsyncTask {
  SchroAsyncTask *next;
  SchroAsyncTask *prev;
  void (*func)(void *);
  void *priv;
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

  if (n_threads == 0) {
    n_threads = 1;
  }
  async = malloc(sizeof(SchroAsync));

  SCHRO_ERROR("%d", n_threads);
  async->n_threads = n_threads;
  async->threads = malloc(sizeof(SchroThread) * n_threads);

  pthread_mutexattr_init (&mutexattr);
  pthread_mutex_init (&async->mutex, &mutexattr);
  pthread_condattr_init (&condattr);
  pthread_cond_init (&async->async_cond, &condattr);
  pthread_cond_init (&async->thread_cond, &condattr);

  pthread_attr_init (&attr);
  
  pthread_mutex_lock (&async->mutex);
  for(i=0;i<n_threads;i++){
    SchroThread *thread = async->threads + i;

    thread->async = async;
    thread->index = i;
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

  pthread_mutex_lock (&async->mutex);
  for(i=0;i<async->n_threads;i++){
    async->threads[i].state = STATE_STOP;
  }
  while(async->n_threads_running > 0) {
    pthread_cond_signal (&async->thread_cond);
    pthread_cond_wait (&async->async_cond, &async->mutex);
  }
  pthread_mutex_unlock (&async->mutex);

  for(i=0;i<async->n_threads;i++){
    void *ignore;
    pthread_join (async->threads[i].pthread, &ignore);
  }

  free(async->threads);
  free(async);
}

void
schro_async_run (SchroAsync *async, void (*func)(void *), void *ptr)
{
  SchroAsyncTask *atask;

  atask = malloc(sizeof(SchroAsyncTask));

  SCHRO_ERROR("queueing task %p", atask);
  atask->func = func;
  atask->priv = ptr;

  pthread_mutex_lock (&async->mutex);
  atask->next = NULL;
  atask->prev = async->list_last;
  if (async->list_last) {
    async->list_last->next = atask;
  } else {
    async->list_first = atask;
  }
  async->list_last = atask;
  async->n_waiting++;

  pthread_cond_signal (&async->thread_cond);
  pthread_mutex_unlock (&async->mutex);
}

int schro_async_get_num_waiting (SchroAsync *async)
{
  return async->n_waiting;
}

int schro_async_get_num_completed (SchroAsync *async)
{
  return async->n_completed;
}

void *schro_async_pull (SchroAsync *async)
{
  SchroAsyncTask *atask;
  void *ptr;

  pthread_mutex_lock (&async->mutex);
  if (!async->done_first) {
    pthread_mutex_unlock (&async->mutex);
    return NULL;
  }
  SCHRO_ASSERT(async->done_first != NULL);

  atask = async->done_first;
  async->done_first = atask->next;
  if (async->done_first) {
    async->done_first->prev = NULL;
  } else {
    async->done_last = NULL;
  }
  async->n_completed--;
  pthread_mutex_unlock (&async->mutex);

  ptr = atask->priv;
  free (atask);

  return ptr;
}

void
schro_async_wait (SchroAsync *async, int min_waiting)
{
  if (min_waiting < 1) min_waiting = 1;

  pthread_mutex_lock (&async->mutex);
  if (async->n_waiting < min_waiting) {
    pthread_mutex_unlock (&async->mutex);
    return;
  }
  if (async->n_completed > 0) {
    pthread_mutex_unlock (&async->mutex);
    return;
  }

  pthread_cond_wait (&async->async_cond, &async->mutex);
  pthread_mutex_unlock (&async->mutex);
}

static void *
schro_thread_main (void *ptr)
{
  SchroThread *thread = ptr;
  SchroAsync *async = thread->async;

  /* thread starts with async->mutex locked */

  async->n_threads_running++;
  while (1) {
    pthread_cond_wait (&async->thread_cond, &async->mutex);

    SCHRO_ERROR("thread %d: got signal", thread->index);

    if (thread->state == STATE_STOP) {
      pthread_cond_signal (&async->async_cond);
      async->n_threads_running--;
      pthread_mutex_unlock (&async->mutex);
      SCHRO_ERROR("thread %d: stopping", thread->index);
      return NULL;
    }

    if (async->list_first == NULL) {
      SCHRO_ERROR("wake with nothing to do");
    }
    thread->state = STATE_BUSY;
    while (async->list_first) {
      SchroAsyncTask *atask;

      atask = async->list_first;

      async->list_first = atask->next;
      if (atask->next) {
        atask->next->prev = NULL;
      } else {
        async->list_last = NULL;
      }
      async->n_waiting--;

      pthread_mutex_unlock (&async->mutex);

      SCHRO_ERROR("thread %d: running", thread->index);
      atask->func (atask->priv);
      SCHRO_ERROR("thread %d: done", thread->index);

      pthread_mutex_lock (&async->mutex);
      
      atask->prev = async->done_last;
      atask->next = NULL;
      if (async->done_last) {
        async->done_last->next = atask;
      } else {
        async->done_first = atask;
      }
      async->done_last = atask;
      async->n_completed++;

      pthread_cond_signal (&async->async_cond);
    }
    thread->state = STATE_IDLE;
  }
}


