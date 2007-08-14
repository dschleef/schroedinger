/* GStreamer
 * Copyright (C) <1999> Erik Walthinsen <omega@cse.ogi.edu>
 * Copyright (C) <2003> David Schleef <ds@schleef.org>
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

/*
 * This file was (probably) generated from
 * gstvideotemplate.c,v 1.18 2005/11/14 02:13:34 thomasvs Exp 
 * and
 * $Id: make_filter,v 1.8 2004/04/19 22:51:57 ds Exp $
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrotables.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_SCHRODOWNSAMPLE \
  (gst_schrodownsample_get_type())
#define GST_SCHRODOWNSAMPLE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHRODOWNSAMPLE,GstSchrodownsample))
#define GST_SCHRODOWNSAMPLE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHRODOWNSAMPLE,GstSchrodownsampleClass))
#define GST_IS_SCHRODOWNSAMPLE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHRODOWNSAMPLE))
#define GST_IS_SCHRODOWNSAMPLE_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHRODOWNSAMPLE))

typedef struct _GstSchrodownsample GstSchrodownsample;
typedef struct _GstSchrodownsampleClass GstSchrodownsampleClass;

struct _GstSchrodownsample
{
  GstBaseTransform base_transform;

  int wavelet_type;
  int level;

  SchroVideoFormat format;
  SchroParams params;

  SchroFrame *tmp_frame;
  int16_t *tmpbuf;

  int frame_number;

};

struct _GstSchrodownsampleClass
{
  GstBaseTransformClass parent_class;

};


/* GstSchrodownsample signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  ARG_0,
  ARG_WAVELET_TYPE,
  ARG_LEVEL
      /* FILL ME */
};

static void gst_schrodownsample_base_init (gpointer g_class);
static void gst_schrodownsample_class_init (gpointer g_class,
    gpointer class_data);
static void gst_schrodownsample_init (GTypeInstance * instance, gpointer g_class);

static void gst_schrodownsample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schrodownsample_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstCaps *gst_schrodownsample_transform_caps (GstBaseTransform * base_transform,
    GstPadDirection direction, GstCaps *caps);
static GstFlowReturn gst_schrodownsample_transform (GstBaseTransform * base_transform,
    GstBuffer *inbuf, GstBuffer *outbuf);
static gboolean gst_schrodownsample_get_unit_size (GstBaseTransform * base_transform,
    GstCaps *caps, guint *size);

static GstStaticPadTemplate gst_schrodownsample_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_schrodownsample_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GType
gst_schrodownsample_get_type (void)
{
  static GType compress_type = 0;

  if (!compress_type) {
    static const GTypeInfo compress_info = {
      sizeof (GstSchrodownsampleClass),
      gst_schrodownsample_base_init,
      NULL,
      gst_schrodownsample_class_init,
      NULL,
      NULL,
      sizeof (GstSchrodownsample),
      0,
      gst_schrodownsample_init,
    };

    compress_type = g_type_register_static (GST_TYPE_BASE_TRANSFORM,
        "GstSchrodownsample", &compress_info, 0);
  }
  return compress_type;
}


static void
gst_schrodownsample_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Video Filter Template",
      "Filter/Effect/Video",
      "Template for a video filter",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  //GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrodownsample_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrodownsample_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_schrodownsample_class_init (gpointer g_class, gpointer class_data)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;
  GstSchrodownsampleClass *downsample_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);
  downsample_class = GST_SCHRODOWNSAMPLE_CLASS (g_class);

  gobject_class->set_property = gst_schrodownsample_set_property;
  gobject_class->get_property = gst_schrodownsample_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_transform_class->transform = gst_schrodownsample_transform;
  base_transform_class->transform_caps = gst_schrodownsample_transform_caps;
  base_transform_class->get_unit_size = gst_schrodownsample_get_unit_size;
}

static void
gst_schrodownsample_init (GTypeInstance * instance, gpointer g_class)
{
  //GstSchrodownsample *compress = GST_SCHRODOWNSAMPLE (instance);
  //GstBaseTransform *btrans = GST_BASE_TRANSFORM (instance);

  GST_DEBUG ("gst_schrodownsample_init");
}

static void
gst_schrodownsample_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSchrodownsample *src;

  g_return_if_fail (GST_IS_SCHRODOWNSAMPLE (object));
  src = GST_SCHRODOWNSAMPLE (object);

  GST_DEBUG ("gst_schrodownsample_set_property");
  switch (prop_id) {
    case ARG_WAVELET_TYPE:
      src->wavelet_type = g_value_get_int (value);
      break;
    case ARG_LEVEL:
      src->level = g_value_get_int (value);
      break;
    default:
      break;
  }
}

static void
gst_schrodownsample_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSchrodownsample *src;

  g_return_if_fail (GST_IS_SCHRODOWNSAMPLE (object));
  src = GST_SCHRODOWNSAMPLE (object);

  switch (prop_id) {
    case ARG_WAVELET_TYPE:
      g_value_set_int (value, src->wavelet_type);
      break;
    case ARG_LEVEL:
      g_value_set_int (value, src->level);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
transform_value (GValue *dest, const GValue *src, GstPadDirection dir)
{
  g_value_init (dest, G_VALUE_TYPE (src));

  if (G_VALUE_HOLDS_INT (src)) {
    int x;

    x = g_value_get_int (src);
    if (dir == GST_PAD_SINK) {
      g_value_set_int (dest, x/2);
    } else {
      g_value_set_int (dest, x*2);
    }
  } else if (GST_VALUE_HOLDS_INT_RANGE (src)) {
    int min, max;

    min = gst_value_get_int_range_min (src);
    max = gst_value_get_int_range_max (src);

    if (dir == GST_PAD_SINK) {
      min = (min+1)/2;
      if (max == G_MAXINT) {
        max = G_MAXINT/2;
      } else {
        max = (max+1)/2;
      }
    } else {
      if (max > G_MAXINT/2) {
        max = G_MAXINT;
      } else {
        max = max*2;
      }
      if (min > G_MAXINT/2) {
        min = G_MAXINT;
      } else {
        min = min*2;
      }
    }
    gst_value_set_int_range (dest, min, max);
  } else {
    /* FIXME */
    g_warning ("case not handled");
    g_value_set_int (dest, 100);
  }
}

static GstCaps *
gst_schrodownsample_transform_caps (GstBaseTransform * base_transform,
    GstPadDirection direction, GstCaps *caps)
{
  int i;
  GstStructure *structure;
  GValue new_value = { 0 };
  const GValue *value;

  caps = gst_caps_copy (caps);

  for(i=0;i<gst_caps_get_size (caps);i++){
    structure = gst_caps_get_structure (caps, i);

    value = gst_structure_get_value (structure, "width");
    transform_value (&new_value, value, direction);
    gst_structure_set_value (structure, "width", &new_value);
    g_value_unset (&new_value);

    value = gst_structure_get_value (structure, "height");
    transform_value (&new_value, value, direction);
    gst_structure_set_value (structure, "height", &new_value);
    g_value_unset (&new_value);
  }

  return caps;
}

static gboolean
gst_schrodownsample_get_unit_size (GstBaseTransform * base_transform,
    GstCaps *caps, guint *size)
{
  int width, height;

  gst_structure_get_int (gst_caps_get_structure(caps, 0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(caps, 0),
      "height", &height);

  *size = width * height * 3 / 2;

  return TRUE;
}

static GstFlowReturn
gst_schrodownsample_transform (GstBaseTransform * base_transform,
    GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstSchrodownsample *compress;
  SchroFrame *frame;
  SchroFrame *outframe;
  int width, height;
  
  g_return_val_if_fail (GST_IS_SCHRODOWNSAMPLE (base_transform), GST_FLOW_ERROR);
  compress = GST_SCHRODOWNSAMPLE (base_transform);

  gst_structure_get_int (gst_caps_get_structure(inbuf->caps, 0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(inbuf->caps, 0),
      "height", &height);

  frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA(inbuf), width, height);
  outframe = schro_frame_new_from_data_I420 (GST_BUFFER_DATA(outbuf), width/2, height/2);
  schro_frame_downsample (outframe, frame);

  return GST_FLOW_OK;
}

