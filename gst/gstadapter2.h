
#ifndef _GST_ADAPTER2_H_
#define _GST_ADAPTER2_H_

#include <gst/gst.h>

typedef struct _GstAdapter2 GstAdapter2;
typedef struct _GstAdapter2Chunk GstAdapter2Chunk;

struct _GstAdapter2
{
  GstAdapter2Chunk *chunks;
  int n_chunks;
  int n_chunks_allocated;
  
  /* number of bytes in the adapter */
  int size;

  /* number of bytes into the first chunk that represents the start point */
  int skip;
};

struct _GstAdapter2Chunk
{
  GstBuffer *buffer;
};


GstAdapter2 *gst_adapter2_new (void);
void gst_adapter2_free (GstAdapter2 *adapter);

int gst_adapter2_available (GstAdapter2 *adapter);
int gst_adapter2_get_offset (GstAdapter2 *adapter);

void gst_adapter2_push (GstAdapter2 *adapter, GstBuffer* buf);
void gst_adapter2_clear (GstAdapter2 *adapter);
void gst_adapter2_flush (GstAdapter2 *adapter, int n);
gboolean gst_adapter_range_is_valid (GstAdapter2 *adapter,
    int offset, int n);

void gst_adapter2_copy (GstAdapter2 *adapter, void *dest, int offset,
    int size);
GstBuffer *gst_adapter2_get_buffer (GstAdapter2 *adapter, int offset,
    int *buffer_offset);
int gst_adapter2_masked_scan_uint32 (GstAdapter2 *adapter,
    guint32 pattern, guint32 mask, int offset, int n);


#endif

