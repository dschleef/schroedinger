
#ifndef __CARID_BUFFER_H__
#define __CARID_BUFFER_H__

typedef struct _CaridBuffer CaridBuffer;

struct _CaridBuffer
{
  unsigned char *data;
  int length;

  int ref_count;

  CaridBuffer *parent;

  void (*free) (CaridBuffer *, void *);
  void *priv;
};

#if 0
struct _CaridBufferQueue
{
  GList *buffers;
  int depth;
  int offset;
};
#endif

CaridBuffer *carid_buffer_new (void);
CaridBuffer *carid_buffer_new_and_alloc (int size);
CaridBuffer *carid_buffer_new_with_data (void *data, int size);
CaridBuffer *carid_buffer_new_subbuffer (CaridBuffer * buffer, int offset,
    int length);
CaridBuffer * carid_buffer_ref (CaridBuffer * buffer);
void carid_buffer_unref (CaridBuffer * buffer);

#if 0
CaridBufferQueue *carid_buffer_queue_new (void);
void carid_buffer_queue_free (CaridBufferQueue * queue);
int carid_buffer_queue_get_depth (CaridBufferQueue * queue);
int carid_buffer_queue_get_offset (CaridBufferQueue * queue);
void carid_buffer_queue_push (CaridBufferQueue * queue,
    CaridBuffer * buffer);
CaridBuffer *carid_buffer_queue_pull (CaridBufferQueue * queue, int len);
CaridBuffer *carid_buffer_queue_peek (CaridBufferQueue * queue, int len);
#endif

#endif
