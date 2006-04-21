/* GStreamer
 * Copyright (C) 2005 David Schleef <ds@schleef.org>
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

#include <gst/gst.h>
#include <carid/carid.h>
#include <liboil/liboil.h>

GType gst_caridtoy_get_type (void);
GType gst_carid_enc_get_type (void);
GType gst_carid_dec_get_type (void);
GType gst_waveletvisualizer_get_type (void);

static gboolean
plugin_init (GstPlugin * plugin)
{
  carid_init();
  oil_init();

#if 0
  gst_element_register (plugin, "caridtoy", GST_RANK_NONE,
      gst_caridtoy_get_type ());
#endif
  gst_element_register (plugin, "caridenc", GST_RANK_PRIMARY,
      gst_carid_enc_get_type ());
  gst_element_register (plugin, "cariddec", GST_RANK_PRIMARY,
      gst_carid_dec_get_type ());
#if 0
  gst_element_register (plugin, "waveletvisualizer", GST_RANK_NONE,
      gst_waveletvisualizer_get_type ());
#endif

  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "carid",
    "Carid plugins",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
