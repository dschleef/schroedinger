
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroasync.h>
#include <pthread.h>
#include <string.h>


struct _SchroAsync {
  int n_threads;

  pthread_cond_t cond1;
  pthread_cond_t cond2;
  pthread_mutex_t mutex;

  void (*callback) (void *);
  void *priv;

  SchroThread *threads;
};

struct _SchroThread {
  pthread_t pthread;
  SchroAsync *async;
  int busy;
  void (*callback) (void *);
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

  async = malloc(sizeof(SchroAsync));

  async->n_threads = n_threads;
  async->threads = malloc(sizeof(SchroThread) * n_threads);

  pthread_mutexattr_init (&mutexattr);
  pthread_mutex_init (&async->mutex, &mutexattr);
  pthread_condattr_init (&condattr);
  pthread_cond_init (&async->cond1, &condattr);
  pthread_cond_init (&async->cond2, &condattr);

  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
  for(i=0;i<n_threads;i++){
    SchroThread *thread = async->threads + i;

    thread->async = async;

    pthread_create (&async->threads[i].pthread, &attr,
        schro_thread_main, async->threads + i);
  }

  pthread_attr_destroy (&attr);
  pthread_mutexattr_destroy (&mutexattr);
  pthread_condattr_destroy (&condattr);

  return async;
}

static void
schro_async_wait_for_ready_and_lock (SchroAsync *async)
{
  int i;

  pthread_mutex_lock (&async->mutex);
  for(i=0;i<async->n_threads;i++){
    if (!async->threads[i].busy) return;
  }

  if (i == async->n_threads) {
    pthread_cond_wait (&async->cond2, &async->mutex);
  }
}

void
schro_async_run (SchroAsync *async, void (*func)(void *), void *ptr)
{
  schro_async_wait_for_ready_and_lock (async);

  async->callback = func;
  async->priv = ptr;

  pthread_cond_signal (&async->cond1);
  pthread_mutex_unlock (&async->mutex);
}

void
schro_async_wait_for_ready (SchroAsync *async)
{

}

static void *
schro_thread_main (void *ptr)
{
  SchroThread *thread = ptr;

  while (1) {
    pthread_mutex_lock (&thread->async->mutex);
    pthread_cond_wait (&thread->async->cond1, &thread->async->mutex);

    thread->callback = thread->async->callback;
    thread->priv = thread->async->priv;
    thread->async->callback = NULL;
    thread->busy = 1;

    pthread_mutex_unlock (&thread->async->mutex);

    if (thread->callback == NULL) {
      return NULL;
    }

    thread->callback (thread->priv);

    pthread_mutex_lock (&thread->async->mutex);
    thread->busy = 0;
    pthread_cond_signal (&thread->async->cond2);
    pthread_mutex_unlock (&thread->async->mutex);
  }
}


