
#ifndef HAVE_CONFIG_H
#include "config.h"
#endif

#include <caridbuffer.h>
//#include <cariddebug.h>
#include <string.h>
#include <stdlib.h>
#include <liboil/liboil.h>

static void carid_buffer_free_mem (CaridBuffer * buffer, void *);
static void carid_buffer_free_subbuffer (CaridBuffer * buffer, void *priv);


#define g_new0(type, n) memset(malloc(sizeof(type) * (n)), 0, sizeof(type) * (n))
#define g_malloc(size) malloc(size)
#define g_free(size) free(size)


CaridBuffer *
carid_buffer_new (void)
{
  CaridBuffer *buffer;

  buffer = g_new0 (CaridBuffer, 1);
  buffer->ref_count = 1;
  return buffer;
}

CaridBuffer *
carid_buffer_new_and_alloc (int size)
{
  CaridBuffer *buffer = carid_buffer_new ();

  buffer->data = g_malloc (size);
  buffer->length = size;
  buffer->free = carid_buffer_free_mem;

  return buffer;
}

CaridBuffer *
carid_buffer_new_with_data (void *data, int size)
{
  CaridBuffer *buffer = carid_buffer_new ();

  buffer->data = data;
  buffer->length = size;

  return buffer;
}

CaridBuffer *
carid_buffer_new_subbuffer (CaridBuffer * buffer, int offset, int length)
{
  CaridBuffer *subbuffer = carid_buffer_new ();

  if (buffer->parent) {
    carid_buffer_ref (buffer->parent);
    subbuffer->parent = buffer->parent;
  } else {
    carid_buffer_ref (buffer);
    subbuffer->parent = buffer;
  }
  subbuffer->data = buffer->data + offset;
  subbuffer->length = length;
  subbuffer->free = carid_buffer_free_subbuffer;

  return subbuffer;
}

CaridBuffer *
carid_buffer_ref (CaridBuffer * buffer)
{
  buffer->ref_count++;
  return buffer;
}

void
carid_buffer_unref (CaridBuffer * buffer)
{
  buffer->ref_count--;
  if (buffer->ref_count == 0) {
    if (buffer->free)
      buffer->free (buffer, buffer->priv);
    g_free (buffer);
  }
}

static void
carid_buffer_free_mem (CaridBuffer * buffer, void *priv)
{
  g_free (buffer->data);
}

static void
carid_buffer_free_subbuffer (CaridBuffer * buffer, void *priv)
{
  carid_buffer_unref (buffer->parent);
}


#if 0
CaridBufferQueue *
carid_buffer_queue_new (void)
{
  return g_new0 (CaridBufferQueue, 1);
}

int
carid_buffer_queue_get_depth (CaridBufferQueue * queue)
{
  return queue->depth;
}

int
carid_buffer_queue_get_offset (CaridBufferQueue * queue)
{
  return queue->offset;
}

void
carid_buffer_queue_free (CaridBufferQueue * queue)
{
  GList *g;

  for (g = g_list_first (queue->buffers); g; g = g_list_next (g)) {
    carid_buffer_unref ((CaridBuffer *) g->data);
  }
  g_list_free (queue->buffers);
  g_free (queue);
}

void
carid_buffer_queue_push (CaridBufferQueue * queue, CaridBuffer * buffer)
{
  queue->buffers = g_list_append (queue->buffers, buffer);
  queue->depth += buffer->length;
}

CaridBuffer *
carid_buffer_queue_pull (CaridBufferQueue * queue, int length)
{
  GList *g;
  CaridBuffer *newbuffer;
  CaridBuffer *buffer;
  CaridBuffer *subbuffer;

  g_return_val_if_fail (length > 0, NULL);

  if (queue->depth < length) {
    return NULL;
  }

  CARID_LOG ("pulling %d, %d available", length, queue->depth);

  g = g_list_first (queue->buffers);
  buffer = g->data;

  if (buffer->length > length) {
    newbuffer = carid_buffer_new_subbuffer (buffer, 0, length);

    subbuffer = carid_buffer_new_subbuffer (buffer, length,
        buffer->length - length);
    g->data = subbuffer;
    carid_buffer_unref (buffer);
  } else {
    int offset = 0;

    newbuffer = carid_buffer_new_and_alloc (length);

    while (offset < length) {
      g = g_list_first (queue->buffers);
      buffer = g->data;

      if (buffer->length > length - offset) {
        int n = length - offset;

        oil_copy_u8 (newbuffer->data + offset, buffer->data, n);
        subbuffer = carid_buffer_new_subbuffer (buffer, n, buffer->length - n);
        g->data = subbuffer;
        carid_buffer_unref (buffer);
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

CaridBuffer *
carid_buffer_queue_peek (CaridBufferQueue * queue, int length)
{
  GList *g;
  CaridBuffer *newbuffer;
  CaridBuffer *buffer;
  int offset = 0;

  g_return_val_if_fail (length > 0, NULL);

  if (queue->depth < length) {
    return NULL;
  }

  CARID_LOG ("peeking %d, %d available", length, queue->depth);

  g = g_list_first (queue->buffers);
  buffer = g->data;
  if (buffer->length > length) {
    newbuffer = carid_buffer_new_subbuffer (buffer, 0, length);
  } else {
    newbuffer = carid_buffer_new_and_alloc (length);
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

