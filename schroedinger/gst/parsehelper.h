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
#ifndef __PARSEHELPER_H__
#define __PARSEHELPER_H__

#include <gst/gst.h>

G_BEGIN_DECLS

typedef struct ParseHelper ParseHelper;

struct ParseHelper
{
  GQueue *queue;
  gint queue_size;

  gint min_parse_offset;
};

void parse_helper_init (ParseHelper *ph);
void parse_helper_flush (ParseHelper *ph);
void parse_helper_free (ParseHelper *ph);

#define parse_helper_avail(ph) ((ph)->queue_size)

void parse_helper_push (ParseHelper *ph, GstBuffer *buffer);
GstBuffer *parse_helper_pull (ParseHelper *ph, gint len);

gboolean parse_helper_have_next_parse_unit (ParseHelper *ph, gint *offset);
gboolean parse_helper_peek (ParseHelper *ph, guint8 *buffer, gint len);
gboolean parse_helper_skip (ParseHelper *ph, gint len);
gboolean parse_helper_skip_to_next_parse_unit (ParseHelper *ph, 
    gint *skipped, gint *next_offset);

G_END_DECLS

#endif
