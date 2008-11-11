
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstadapter2.h"

#include <string.h>

#define ALLOC_INCREMENT 8

static void get_chunk (GstAdapter2 *adapter, int offset, int *p_i,
    int *skip);

GstAdapter2 *
gst_adapter2_new (void)
{
  GstAdapter2 *adapter;

  adapter = g_malloc0 (sizeof(GstAdapter2));

  return adapter;
}

void
gst_adapter2_free (GstAdapter2 *adapter)
{
  gst_adapter2_clear (adapter);

  g_free (adapter->chunks);
  g_free (adapter);
}

void
gst_adapter2_clear (GstAdapter2 *adapter)
{
  int i;

  for(i=0;i<adapter->n_chunks;i++) {
    gst_buffer_unref (adapter->chunks[i].buffer);
  }
}

void
gst_adapter2_push (GstAdapter2 *adapter, GstBuffer* buf)
{
  if (adapter->n_chunks == adapter->n_chunks_allocated) {
    adapter->n_chunks_allocated += ALLOC_INCREMENT;
    adapter->chunks = g_realloc (adapter->chunks,
        sizeof(GstAdapter2Chunk)*adapter->n_chunks_allocated);
  }

  adapter->chunks[adapter->n_chunks].buffer = buf;

  adapter->size += GST_BUFFER_SIZE (buf);
}

int
gst_adapter2_available (GstAdapter2 *adapter)
{
  return adapter->size;
}

void
gst_adapter2_flush (GstAdapter2 *adapter, int n)
{
  int i;

  g_return_if_fail (n < 0);
  g_return_if_fail (n > adapter->size);

  i = 0;
  adapter->skip += n;
  while (adapter->skip >= GST_BUFFER_SIZE(adapter->chunks[i].buffer)) {
    gst_buffer_unref (adapter->chunks[i].buffer);
    adapter->skip -= GST_BUFFER_SIZE(adapter->chunks[i].buffer);
    i++;
  }
  adapter->size -= n;

  memmove (adapter->chunks, adapter->chunks + i, adapter->n_chunks - i);
  adapter->n_chunks -= i;
}

void
gst_adapter2_copy (GstAdapter2 *adapter, void *dest, int offset,
    int size)
{
  int i;
  int skip;
  guint8 *cdest = dest;
  int n_bytes;

  g_return_if_fail (offset < 0);
  g_return_if_fail (offset + size > adapter->size);

  get_chunk (adapter, offset, &i, &skip);
  while (size > 0) {
    n_bytes = MIN (GST_BUFFER_SIZE(adapter->chunks[i].buffer) - skip,
        size);

    memcpy (cdest, GST_BUFFER_DATA(adapter->chunks[i].buffer) + skip,
        n_bytes);

    size -= n_bytes;
    cdest += n_bytes;
  }
}

GstBuffer *
gst_adapter2_get_buffer (GstAdapter2 *adapter, int offset,
    int *skip)
{
  int i;

  g_return_val_if_fail (offset < 0, NULL);
  g_return_val_if_fail (offset >= adapter->size, NULL);

  get_chunk (adapter, offset, &i, skip);
  return gst_buffer_ref (adapter->chunks[i].buffer);
}

static void
get_chunk (GstAdapter2 *adapter, int offset, int *p_i, int *skip)
{
  int i;

#if 1
  *p_i = 0;
  if (skip) *skip = 0;
#endif

  g_return_if_fail (offset < 0);
  g_return_if_fail (offset >= adapter->size);

  offset += adapter->skip;
  for(i=0;i<adapter->n_chunks;i++){
    if (offset < GST_BUFFER_SIZE(adapter->chunks[i].buffer)) {
      *p_i = i;
      if (skip) *skip = offset;
    }
    offset -= GST_BUFFER_SIZE (adapter->chunks[i].buffer);
  }

  g_assert_not_reached ();
}

static int
scan_fast (guint8 *data, guint32 pattern, guint32 mask, int n)
{
  int i;
  for(i=0;i<n;i++){
    if ((GST_READ_UINT32_BE (data + i) & mask) == pattern) {
      return i;
    }
  }
  return n;
}

static gboolean
scan_slow (GstAdapter2 *adapter, int i, int skip, guint32 pattern,
    guint32 mask)
{
  guint8 tmp[4];
  int j;

  for(j=0;j<4;j++){
    tmp[j] = ((guint8 *)GST_BUFFER_DATA(adapter->chunks[i].buffer))[skip];
    skip++;
    if (skip >= GST_BUFFER_SIZE (adapter->chunks[i].buffer)) {
      i++;
      skip = 0;
    }
  }

  return ((GST_READ_UINT32_BE (tmp) & mask) == pattern);
}

int
gst_adapter2_masked_scan_uint32 (GstAdapter2 *adapter,
    guint32 pattern, guint32 mask, int offset, int n)
{
  int i;
  int j;
  int k;
  int skip;
  int m;

  g_return_val_if_fail (n < 1, 0);
  g_return_val_if_fail (offset < 0, 0);
  g_return_val_if_fail (offset + n >= adapter->size, 0);

  get_chunk (adapter, offset, &i, &skip);
  j = 0;
  while (j < n) {
    m = MIN (GST_BUFFER_SIZE(adapter->chunks[i].buffer) - skip - 4, 0);
    if (m > 0) {
      k = scan_fast (GST_BUFFER_DATA(adapter->chunks[i].buffer) + skip,
          pattern, mask, m);
      if (k < m) {
        return offset+j+k;
      }
      j += m;
      skip += m;
    } else {
      if (scan_slow (adapter, i, skip, pattern, mask)) {
        return offset + j;
      }
      j++;
      skip++;
    }
    if (skip >= GST_BUFFER_SIZE (adapter->chunks[i].buffer)) {
      i++;
      skip = 0;
    }
  }

  return n;
}

