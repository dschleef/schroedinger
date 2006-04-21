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
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_SCHROTOY \
  (gst_schrotoy_get_type())
#define GST_SCHROTOY(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHROTOY,GstSchrotoy))
#define GST_SCHROTOY_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHROTOY,GstSchrotoyClass))
#define GST_IS_SCHROTOY(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHROTOY))
#define GST_IS_SCHROTOY_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHROTOY))

typedef struct _GstSchrotoy GstSchrotoy;
typedef struct _GstSchrotoyClass GstSchrotoyClass;

struct _GstSchrotoy
{
  GstBaseTransform base_transform;

  int wavelet_type;
  int level;

  SchroEncoder *encoder;
  SchroDecoder *decoder;
};

struct _GstSchrotoyClass
{
  GstBaseTransformClass parent_class;
};


/* GstSchrotoy signals and args */
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

static void gst_schrotoy_base_init (gpointer g_class);
static void gst_schrotoy_class_init (gpointer g_class,
    gpointer class_data);
static void gst_schrotoy_init (GTypeInstance * instance, gpointer g_class);

static void gst_schrotoy_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schrotoy_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_schrotoy_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf);

static GstStaticPadTemplate gst_schrotoy_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_schrotoy_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GType
gst_schrotoy_get_type (void)
{
  static GType compress_type = 0;

  if (!compress_type) {
    static const GTypeInfo compress_info = {
      sizeof (GstSchrotoyClass),
      gst_schrotoy_base_init,
      NULL,
      gst_schrotoy_class_init,
      NULL,
      NULL,
      sizeof (GstSchrotoy),
      0,
      gst_schrotoy_init,
    };

    compress_type = g_type_register_static (GST_TYPE_BASE_TRANSFORM,
        "GstSchrotoy", &compress_info, 0);
  }
  return compress_type;
}


static void
gst_schrotoy_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Video Filter Template",
      "Filter/Effect/Video",
      "Template for a video filter",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  //GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrotoy_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schrotoy_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_schrotoy_class_init (gpointer g_class, gpointer class_data)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gobject_class->set_property = gst_schrotoy_set_property;
  gobject_class->get_property = gst_schrotoy_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_transform_class->transform_ip = gst_schrotoy_transform_ip;
}

static void
gst_schrotoy_init (GTypeInstance * instance, gpointer g_class)
{
  GstSchrotoy *compress = GST_SCHROTOY (instance);

  GST_DEBUG ("gst_schrotoy_init");

  compress->encoder = schro_encoder_new ();
  compress->decoder = schro_decoder_new ();
}

static void
gst_schrotoy_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSchrotoy *src;

  g_return_if_fail (GST_IS_SCHROTOY (object));
  src = GST_SCHROTOY (object);

  GST_DEBUG ("gst_schrotoy_set_property");
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
gst_schrotoy_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSchrotoy *src;

  g_return_if_fail (GST_IS_SCHROTOY (object));
  src = GST_SCHROTOY (object);

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

static GstFlowReturn
gst_schrotoy_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf)
{
  GstSchrotoy *compress;
  int width;
  int height;
  SchroBuffer *input_buffer;
  SchroBuffer *encoded_buffer;
  SchroBuffer *decoded_buffer;

  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "height", &height);

  g_return_val_if_fail (GST_IS_SCHROTOY (base_transform), GST_FLOW_ERROR);
  compress = GST_SCHROTOY (base_transform);

  schro_encoder_set_size (compress->encoder, width, height);
  schro_encoder_set_wavelet_type (compress->encoder, compress->wavelet_type);

  if (1) {
  input_buffer = schro_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  schro_encoder_push_buffer (compress->encoder, input_buffer);
  encoded_buffer = schro_encoder_encode (compress->encoder);
  schro_buffer_unref (input_buffer);

#if 0
  {
    int i;
    int j;
    int n;
    int16_t *data;
    int w;
    int h;
    int x = compress->level;
    int shift;

    data = (int16_t *)encoded_buffer->data;
    n = encoded_buffer->length/2;

    w = (width + 63) & (~63);
    h = (height + 63) & (~63);

    shift=4;
    for(i=0;i<h>>shift;i++){
      for(j=0;j<w>>shift;j++){
        data[(i+(h>>shift))*w + j] += g_random_int_range (-x,x+1);
        data[i*w + j + (w>>shift)] += g_random_int_range (-x,x+1);
        data[(i+(h>>shift))*w + j + (w>>shift)] += g_random_int_range (-x,x+1);
      }
    }
  }
#endif


  decoded_buffer = schro_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  schro_decoder_set_output_buffer (compress->decoder, decoded_buffer);
  schro_decoder_decode (compress->decoder, encoded_buffer);
  } else {
    int16_t *frame_data;
    int16_t *tmp;

    frame_data = g_malloc (width*height*2);
    tmp = g_malloc (2048);
    
    oil_convert_s16_u8 (frame_data, GST_BUFFER_DATA(buf), width*height);
    schro_wavelet_transform_2d (compress->wavelet_type,
        frame_data, width*2, width, height, tmp);
    schro_wavelet_transform_2d (compress->wavelet_type,
        frame_data, width*2*2, width/2, height/2, tmp);


    schro_wavelet_inverse_transform_2d (compress->wavelet_type,
        frame_data, width*2*2, width/2, height/2, tmp);
    schro_wavelet_inverse_transform_2d (compress->wavelet_type,
        frame_data, width*2, width, height, tmp);
    oil_convert_u8_s16 (GST_BUFFER_DATA(buf), frame_data, width*height);

    g_free(frame_data);
    g_free(tmp);
  }

  return GST_FLOW_OK;
}

