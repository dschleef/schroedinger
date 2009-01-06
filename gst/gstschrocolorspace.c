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
#include <schroedinger/schrovirtframe.h>

#define GST_TYPE_SCHROCOLORSPACE \
  (gst_schrocolorspace_get_type())
#define GST_SCHROCOLORSPACE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHROCOLORSPACE,GstSchrocolorspace))
#define GST_SCHROCOLORSPACE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHROCOLORSPACE,GstSchrocolorspaceClass))
#define GST_IS_SCHROCOLORSPACE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHROCOLORSPACE))
#define GST_IS_SCHROCOLORSPACE_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHROCOLORSPACE))

typedef struct _GstSchrocolorspace GstSchrocolorspace;
typedef struct _GstSchrocolorspaceClass GstSchrocolorspaceClass;

struct _GstSchrocolorspace
{
  GstBaseTransform base_transform;

  SchroVideoFormat format;
};

struct _GstSchrocolorspaceClass
{
  GstBaseTransformClass parent_class;

};

GType gst_schrocolorspace_get_type (void);

/* GstSchrocolorspace signals and args */
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

static void gst_schrocolorspace_base_init (gpointer g_class);
static void gst_schrocolorspace_class_init (gpointer g_class,
    gpointer class_data);
static void gst_schrocolorspace_init (GTypeInstance * instance, gpointer g_class);

static void gst_schrocolorspace_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schrocolorspace_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstCaps *gst_schrocolorspace_transform_caps (GstBaseTransform * base_transform,
    GstPadDirection direction, GstCaps *caps);
static GstFlowReturn gst_schrocolorspace_transform (GstBaseTransform * base_transform,
    GstBuffer *inbuf, GstBuffer *outbuf);
static gboolean gst_schrocolorspace_get_unit_size (GstBaseTransform * base_transform,
    GstCaps *caps, guint *size);

static GstStaticPadTemplate gst_schrocolorspace_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("{ I420, YV12, YUY2, UYVY, AYUV, v216, v210 }"))
    );

static GstStaticPadTemplate gst_schrocolorspace_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("{ I420, YV12, YUY2, UYVY, AYUV, v216, v210 }"))
    );

GType
gst_schrocolorspace_get_type (void)
{
  static GType compress_type = 0;

  if (!compress_type) {
    static const GTypeInfo compress_info = {
      sizeof (GstSchrocolorspaceClass),
      gst_schrocolorspace_base_init,
      NULL,
      gst_schrocolorspace_class_init,
      NULL,
      NULL,
      sizeof (GstSchrocolorspace),
      0,
      gst_schrocolorspace_init,
    };

    compress_type = g_type_register_static (GST_TYPE_BASE_TRANSFORM,
        "GstSchrocolorspace", &compress_info, 0);
  }
  return compress_type;
}


static void
gst_schrocolorspace_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("YCbCr format conversion",
      "Filter/Effect/Video",
      "YCbCr format conversion",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  //GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrocolorspace_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrocolorspace_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_schrocolorspace_class_init (gpointer g_class, gpointer class_data)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;
  GstSchrocolorspaceClass *colorspace_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);
  colorspace_class = GST_SCHROCOLORSPACE_CLASS (g_class);

  gobject_class->set_property = gst_schrocolorspace_set_property;
  gobject_class->get_property = gst_schrocolorspace_get_property;

#if 0
  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));
#endif

  base_transform_class->transform = gst_schrocolorspace_transform;
  base_transform_class->transform_caps = gst_schrocolorspace_transform_caps;
  base_transform_class->get_unit_size = gst_schrocolorspace_get_unit_size;
}

static void
gst_schrocolorspace_init (GTypeInstance * instance, gpointer g_class)
{
  //GstSchrocolorspace *compress = GST_SCHROCOLORSPACE (instance);
  //GstBaseTransform *btrans = GST_BASE_TRANSFORM (instance);

  GST_DEBUG ("gst_schrocolorspace_init");
}

static void
gst_schrocolorspace_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSchrocolorspace *src;

  g_return_if_fail (GST_IS_SCHROCOLORSPACE (object));
  src = GST_SCHROCOLORSPACE (object);

  GST_DEBUG ("gst_schrocolorspace_set_property");
  switch (prop_id) {
    default:
      break;
  }
}

static void
gst_schrocolorspace_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSchrocolorspace *src;

  g_return_if_fail (GST_IS_SCHROCOLORSPACE (object));
  src = GST_SCHROCOLORSPACE (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
transform_value (GValue *dest)
{
  GValue fourcc = { 0 };

  g_value_init (dest, GST_TYPE_LIST);
  g_value_init (&fourcc, GST_TYPE_FOURCC);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('I','4','2','0'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('Y','V','1','2'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('Y','U','Y','2'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('U','Y','V','Y'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('A','Y','U','V'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('v','2','1','0'));
  gst_value_list_append_value (dest, &fourcc);

  gst_value_set_fourcc (&fourcc, GST_MAKE_FOURCC('v','2','1','6'));
  gst_value_list_append_value (dest, &fourcc);

  g_value_unset (&fourcc);
}

static GstCaps *
gst_schrocolorspace_transform_caps (GstBaseTransform * base_transform,
    GstPadDirection direction, GstCaps *caps)
{
  int i;
  GstStructure *structure;
  GValue new_value = { 0 };
  const GValue *value;

  caps = gst_caps_copy (caps);

  for(i=0;i<gst_caps_get_size (caps);i++){
    structure = gst_caps_get_structure (caps, i);

    value = gst_structure_get_value (structure, "format");
    transform_value (&new_value);
    gst_structure_set_value (structure, "format", &new_value);
    g_value_unset (&new_value);
  }

  return caps;
}

static gboolean
gst_schrocolorspace_get_unit_size (GstBaseTransform * base_transform,
    GstCaps *caps, guint *size)
{
  int width, height;
  uint32_t format;

  gst_structure_get_fourcc (gst_caps_get_structure(caps, 0),
      "format", &format);
  gst_structure_get_int (gst_caps_get_structure(caps, 0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(caps, 0),
      "height", &height);

  switch (format) {
    case GST_MAKE_FOURCC('I','4','2','0'):
    case GST_MAKE_FOURCC('Y','V','1','2'):
      *size = width * height * 3 / 2;
      break;
    case GST_MAKE_FOURCC('Y','U','Y','2'):
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      *size = width * height * 2;
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      *size = width * height * 4;
      break;
    case GST_MAKE_FOURCC('v','2','1','6'):
      *size = width * height * 4;
      break;
    case GST_MAKE_FOURCC('v','2','1','0'):
      *size = ((width + 47) / 48) * 128 * height;
      break;
    default:
      g_assert_not_reached();
  }

  return TRUE;
}

static GstFlowReturn
gst_schrocolorspace_transform (GstBaseTransform * base_transform,
    GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstSchrocolorspace *compress;
  SchroFrame *out_frame;
  SchroFrame *frame;
  int width, height;
  uint32_t in_format;
  uint32_t out_format;
  SchroFrameFormat new_subsample;
  
  g_return_val_if_fail (GST_IS_SCHROCOLORSPACE (base_transform), GST_FLOW_ERROR);
  compress = GST_SCHROCOLORSPACE (base_transform);

  gst_structure_get_fourcc (gst_caps_get_structure(inbuf->caps, 0),
      "format", &in_format);
  gst_structure_get_fourcc (gst_caps_get_structure(outbuf->caps, 0),
      "format", &out_format);
  gst_structure_get_int (gst_caps_get_structure(inbuf->caps, 0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(inbuf->caps, 0),
      "height", &height);

  switch (in_format) {
    case GST_MAKE_FOURCC('I','4','2','0'):
      frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('Y','V','1','2'):
      frame = schro_frame_new_from_data_YV12 (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('Y','U','Y','2'):
      frame = schro_frame_new_from_data_YUY2 (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      frame = schro_frame_new_from_data_UYVY (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      frame = schro_frame_new_from_data_AYUV (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('v','2','1','6'):
      frame = schro_frame_new_from_data_v216 (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    case GST_MAKE_FOURCC('v','2','1','0'):
      frame = schro_frame_new_from_data_v210 (GST_BUFFER_DATA(inbuf),
          width, height);
      break;
    default:
      g_assert_not_reached();
  }

  switch (out_format) {
    case GST_MAKE_FOURCC('I','4','2','0'):
      out_frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_420;
      break;
    case GST_MAKE_FOURCC('Y','V','1','2'):
      out_frame = schro_frame_new_from_data_YV12 (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_420;
      break;
    case GST_MAKE_FOURCC('Y','U','Y','2'):
      out_frame = schro_frame_new_from_data_YUY2 (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_422;
      break;
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      out_frame = schro_frame_new_from_data_UYVY (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_422;
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      out_frame = schro_frame_new_from_data_AYUV (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_444;
      break;
    case GST_MAKE_FOURCC('v','2','1','6'):
      out_frame = schro_frame_new_from_data_v216 (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_422;
      break;
    case GST_MAKE_FOURCC('v','2','1','0'):
      out_frame = schro_frame_new_from_data_v210 (GST_BUFFER_DATA(outbuf),
          width, height);
      new_subsample = SCHRO_FRAME_FORMAT_U8_422;
      break;
    default:
      g_assert_not_reached();
  }

  frame = schro_virt_frame_new_unpack_take (frame);
  frame = schro_virt_frame_new_subsample_take (frame, new_subsample);

  switch (out_format) {
    case GST_MAKE_FOURCC('Y','U','Y','2'):
      frame = schro_virt_frame_new_pack_YUY2_take (frame);
      break;
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      frame = schro_virt_frame_new_pack_UYVY_take (frame);
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      frame = schro_virt_frame_new_pack_AYUV_take (frame);
      break;
    case GST_MAKE_FOURCC('v','2','1','6'):
      frame = schro_virt_frame_new_pack_v216_take (frame);
      break;
    case GST_MAKE_FOURCC('v','2','1','0'):
      frame = schro_virt_frame_new_pack_v210_take (frame);
      break;
    default:
      break;
  }

  schro_virt_frame_render (frame, out_frame);
  schro_frame_unref (frame);
  schro_frame_unref (out_frame);

  return GST_FLOW_OK;
}

