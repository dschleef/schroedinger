/* Schrodinger
 * Copyright (C) 2006 David Schleef <ds@schleef.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>

#include "parsehelper.h"

void parse_helper_init (ParseHelper *ph)
{
  ph->queue = g_queue_new ();
  ph->queue_size = 0;
  ph->min_parse_offset = 0;
}

void parse_helper_flush (ParseHelper *ph)
{
  GstBuffer *buf;

  for (buf = g_queue_pop_head (ph->queue); buf != NULL; 
      buf = g_queue_pop_head (ph->queue)) {
    gst_buffer_unref (buf);
  }

  ph->queue_size = 0;
  ph->min_parse_offset = 0;
}

void parse_helper_free (ParseHelper *ph)
{
  parse_helper_flush (ph);
  g_queue_free (ph->queue);
}

void parse_helper_push (ParseHelper *ph, GstBuffer *buf)
{
  g_return_if_fail (ph != NULL);
  g_return_if_fail (ph->queue != NULL);
  g_return_if_fail (buf != NULL);

  g_queue_push_tail (ph->queue, buf);
  ph->queue_size += GST_BUFFER_SIZE(buf);
}

gboolean
parse_helper_have_next_parse_unit (ParseHelper *ph, gint *offset)
{
  const guint32 hdr_val = ('B' << 24) | ('B' << 16) | ('C' << 8) | ('D');
  guint32 collect = 0;
  gint next_offset = 0;
  GList *cur = g_list_first (ph->queue->head);

  /* Skip at least the first byte in the first buffer to avoid the 
   * first PARSE header */
  gint cur_offset = MAX (ph->min_parse_offset, 1);

  while (cur != NULL) {
    GstBuffer *buf = (GstBuffer *) cur->data;

    while (cur_offset < GST_BUFFER_SIZE (buf)) {
      collect = (collect << 8) | GST_BUFFER_DATA (buf) [cur_offset];
      if (G_UNLIKELY (collect == hdr_val)) {
        *offset = next_offset + cur_offset - 3;
        return TRUE;
      }
      cur_offset++;
    }

    next_offset += GST_BUFFER_SIZE (buf);
    cur_offset -= GST_BUFFER_SIZE (buf);
    cur = g_list_next (cur);
  }

  /* We can scan more efficiently by remembering how many bytes to skip
   * next time */
  ph->min_parse_offset = MAX (0, next_offset + cur_offset - 3);
  return FALSE;
}

gboolean
parse_helper_peek (ParseHelper *ph, guint8 *buffer, int len)
{
  GstBuffer *buf;
  int n;
  int i = 0;

  if (parse_helper_avail (ph) < len)
    return FALSE;

  while (len > 0) {
    buf = g_queue_peek_nth (ph->queue, i);
    i++;

    n = GST_BUFFER_SIZE(buf);
    if (n > len) n = len;

    memcpy (buffer, GST_BUFFER_DATA(buf), n);
    buffer += n;
    len -= n;
  }

  return TRUE;
}

gboolean
parse_helper_skip (ParseHelper *ph, int len)
{
  GstBuffer *buf;

  if (parse_helper_avail (ph) < len)
    return FALSE;
  ph->queue_size -= len;
  ph->min_parse_offset = MAX (0, ph->min_parse_offset - len);

  while (len > 0) {
    buf = g_queue_pop_head (ph->queue);

    if (GST_BUFFER_SIZE (buf) < len) {
      len -= GST_BUFFER_SIZE (buf);
    } else {
      GstBuffer *remainder;
      remainder = gst_buffer_create_sub (buf, len, GST_BUFFER_SIZE(buf) - len);
      g_queue_push_head (ph->queue, remainder);
      len = 0;
    }
    gst_buffer_unref (buf);
  }

  return TRUE;
}

GstBuffer *
parse_helper_pull (ParseHelper *ph, int len)
{
  GstBuffer *result;
  GstBuffer *buf;
  guint8 *data;

  if (parse_helper_avail (ph) < len)
    return FALSE;

  ph->queue_size -= len;
  ph->min_parse_offset = MAX (0, ph->min_parse_offset - len);

  buf = g_queue_peek_head (ph->queue);
  if (GST_BUFFER_SIZE(buf) == len) {
    buf = g_queue_pop_head (ph->queue);
    return buf;
  }
  if (GST_BUFFER_SIZE(buf) > len) {
    GstBuffer *remainder;

    buf = g_queue_pop_head (ph->queue);
    result = gst_buffer_create_sub (buf, 0, len);
    remainder = gst_buffer_create_sub (buf, len, GST_BUFFER_SIZE(buf) - len);
    gst_buffer_unref (buf);
    g_queue_push_head (ph->queue, remainder);

    return result;
  }

  result = gst_buffer_new_and_alloc (len);
  data = GST_BUFFER_DATA(result);
  while (len > 0) {
    buf = g_queue_pop_head (ph->queue);
    if (GST_BUFFER_SIZE (buf) < len) {
      memcpy (data, GST_BUFFER_DATA(buf), GST_BUFFER_SIZE(buf));
      len -= GST_BUFFER_SIZE (buf);
      data += GST_BUFFER_SIZE (buf);
    } else {
      GstBuffer *remainder;
      memcpy (data, GST_BUFFER_DATA(buf), len);
      remainder = gst_buffer_create_sub (buf, len, GST_BUFFER_SIZE(buf) - len);
      g_queue_push_head (ph->queue, remainder);
      len -= len;
      data += len;
    }
    gst_buffer_unref (buf);
  }
  
  return result;
}

gboolean parse_helper_skip_to_next_parse_unit (ParseHelper *ph, 
    gint *skipped, gint *next_offset)
{
  gint nskipped = 0;
  gint next = 0;
  gboolean found = FALSE;
    
  /* Check if we're already at the start of a parse unit */
  if (parse_helper_avail (ph) > 3) {
    guint8 header[11];

    parse_helper_peek (ph, header, 4);

    if (memcmp(header, "BBCD", 4) == 0) {
      found = TRUE;
    }
    else {
      if (parse_helper_have_next_parse_unit (ph, &nskipped)) {
        parse_helper_skip (ph, nskipped);
        found = TRUE;
      }
      else {
        /* No parse unit in the current buffer, skip everything except the
         * 3 bytes that might be the start of a new header */
        parse_helper_skip (ph, parse_helper_avail (ph) - 3);
      }
    }

    if (parse_helper_avail (ph) > 10) {
      parse_helper_peek (ph, header, 11);

      next = (header[5]<<16) | (header[6]<<8) | header[7];
    }
  }

  if (skipped)
    *skipped = nskipped;
  if (next_offset)
    *next_offset = next;
  
  return found;
}
