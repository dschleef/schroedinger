
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrobuffer.h>
//#include <schroedinger/schrodebug.h>
#include <string.h>
#include <stdlib.h>
#include <liboil/liboil.h>

static void schro_buffer_free_mem (SchroBuffer * buffer, void *);
static void schro_buffer_free_subbuffer (SchroBuffer * buffer, void *priv);


#define g_new0(type, n) memset(malloc(sizeof(type) * (n)), 0, sizeof(type) * (n))
#define g_malloc(size) malloc(size)
#define g_free(size) free(size)


SchroBuffer *
schro_buffer_new (void)
{
  SchroBuffer *buffer;

  buffer = g_new0 (SchroBuffer, 1);
  buffer->ref_count = 1;
  return buffer;
}

SchroBuffer *
schro_buffer_new_and_alloc (int size)
{
  SchroBuffer *buffer = schro_buffer_new ();

  buffer->data = g_malloc (size);
  buffer->length = size;
  buffer->free = schro_buffer_free_mem;

  return buffer;
}

SchroBuffer *
schro_buffer_new_with_data (void *data, int size)
{
  SchroBuffer *buffer = schro_buffer_new ();

  buffer->data = data;
  buffer->length = size;

  return buffer;
}

SchroBuffer *
schro_buffer_new_subbuffer (SchroBuffer * buffer, int offset, int length)
{
  SchroBuffer *subbuffer = schro_buffer_new ();

  if (buffer->parent) {
    schro_buffer_ref (buffer->parent);
    subbuffer->parent = buffer->parent;
  } else {
    schro_buffer_ref (buffer);
    subbuffer->parent = buffer;
  }
  subbuffer->data = buffer->data + offset;
  subbuffer->length = length;
  subbuffer->free = schro_buffer_free_subbuffer;

  return subbuffer;
}

SchroBuffer *
schro_buffer_ref (SchroBuffer * buffer)
{
  buffer->ref_count++;
  return buffer;
}

void
schro_buffer_unref (SchroBuffer * buffer)
{
  buffer->ref_count--;
  if (buffer->ref_count == 0) {
    if (buffer->free)
      buffer->free (buffer, buffer->priv);
    g_free (buffer);
  }
}

static void
schro_buffer_free_mem (SchroBuffer * buffer, void *priv)
{
  g_free (buffer->data);
}

static void
schro_buffer_free_subbuffer (SchroBuffer * buffer, void *priv)
{
  schro_buffer_unref (buffer->parent);
}


#if 0
SchroBufferQueue *
schro_buffer_queue_new (void)
{
  return g_new0 (SchroBufferQueue, 1);
}

int
schro_buffer_queue_get_depth (SchroBufferQueue * queue)
{
  return queue->depth;
}

int
schro_buffer_queue_get_offset (SchroBufferQueue * queue)
{
  return queue->offset;
}

void
schro_buffer_queue_free (SchroBufferQueue * queue)
{
  GList *g;

  for (g = g_list_first (queue->buffers); g; g = g_list_next (g)) {
    schro_buffer_unref ((SchroBuffer *) g->data);
  }
  g_list_free (queue->buffers);
  g_free (queue);
}

void
schro_buffer_queue_push (SchroBufferQueue * queue, SchroBuffer * buffer)
{
  queue->buffers = g_list_append (queue->buffers, buffer);
  queue->depth += buffer->length;
}

SchroBuffer *
schro_buffer_queue_pull (SchroBufferQueue * queue, int length)
{
  GList *g;
  SchroBuffer *newbuffer;
  SchroBuffer *buffer;
  SchroBuffer *subbuffer;

  g_return_val_if_fail (length > 0, NULL);

  if (queue->depth < length) {
    return NULL;
  }

  SCHRO_LOG ("pulling %d, %d available", length, queue->depth);

  g = g_list_first (queue->buffers);
  buffer = g->data;

  if (buffer->length > length) {
    newbuffer = schro_buffer_new_subbuffer (buffer, 0, length);

    subbuffer = schro_buffer_new_subbuffer (buffer, length,
        buffer->length - length);
    g->data = subbuffer;
    schro_buffer_unref (buffer);
  } else {
    int offset = 0;

    newbuffer = schro_buffer_new_and_alloc (length);

    while (offset < length) {
      g = g_list_first (queue->buffers);
      buffer = g->data;

      if (buffer->length > length - offset) {
        int n = length - offset;

        oil_copy_u8 (newbuffer->data + offset, buffer->data, n);
        subbuffer = schro_buffer_new_subbuffer (buffer, n, buffer->length - n);
        g->data = subbuffer;
        schro_buffer_unref (buffer);
        offset += n;
      } else {
        oil_copy_u8 (newbuffer->data + offset, buffer->data, buffer->length);

        queue->buffers = g_list_delete_link (queue->buffers, g);
        offset += buffer->length;
      }
    }
  }

  queue->depth -= length;
  queue->offset += length;

  return newbuffer;
}

SchroBuffer *
schro_buffer_queue_peek (SchroBufferQueue * queue, int length)
{
  GList *g;
  SchroBuffer *newbuffer;
  SchroBuffer *buffer;
  int offset = 0;

  g_return_val_if_fail (length > 0, NULL);

  if (queue->depth < length) {
    return NULL;
  }

  SCHRO_LOG ("peeking %d, %d available", length, queue->depth);

  g = g_list_first (queue->buffers);
  buffer = g->data;
  if (buffer->length > length) {
    newbuffer = schro_buffer_new_subbuffer (buffer, 0, length);
  } else {
    newbuffer = schro_buffer_new_and_alloc (length);
    while (offset < length) {
      buffer = g->data;

      if (buffer->length > length - offset) {
        int n = length - offset;

        oil_copy_u8 (newbuffer->data + offset, buffer->data, n);
        offset += n;
      } else {
        oil_copy_u8 (newbuffer->data + offset, buffer->data, buffer->length);
        offset += buffer->length;
      }
      g = g_list_next (g);
    }
  }

  return newbuffer;
}
#endif

