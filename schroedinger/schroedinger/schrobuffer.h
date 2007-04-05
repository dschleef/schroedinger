
#ifndef __SCHRO_BUFFER_H__
#define __SCHRO_BUFFER_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroBuffer SchroBuffer;

struct _SchroBuffer
{
  unsigned char *data;
  int length;

  int ref_count;

  SchroBuffer *parent;

  void (*free) (SchroBuffer *, void *);
  void *priv;
};

#if 0
struct _SchroBufferQueue
{
  GList *buffers;
  int depth;
  int offset;
};
#endif

SchroBuffer *schro_buffer_new (void);
SchroBuffer *schro_buffer_new_and_alloc (int size);
SchroBuffer *schro_buffer_new_with_data (void *data, int size);
SchroBuffer *schro_buffer_new_subbuffer (SchroBuffer * buffer, int offset,
    int length);
SchroBuffer * schro_buffer_ref (SchroBuffer * buffer);
void schro_buffer_unref (SchroBuffer * buffer);

#if 0
SchroBufferQueue *schro_buffer_queue_new (void);
void schro_buffer_queue_free (SchroBufferQueue * queue);
int schro_buffer_queue_get_depth (SchroBufferQueue * queue);
int schro_buffer_queue_get_offset (SchroBufferQueue * queue);
void schro_buffer_queue_push (SchroBufferQueue * queue,
    SchroBuffer * buffer);
SchroBuffer *schro_buffer_queue_pull (SchroBufferQueue * queue, int len);
SchroBuffer *schro_buffer_queue_peek (SchroBufferQueue * queue, int len);
#endif

SCHRO_END_DECLS

#endif
