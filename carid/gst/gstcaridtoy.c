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
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_CARIDTOY \
  (gst_caridtoy_get_type())
#define GST_CARIDTOY(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CARIDTOY,GstCaridtoy))
#define GST_CARIDTOY_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CARIDTOY,GstCaridtoyClass))
#define GST_IS_CARIDTOY(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CARIDTOY))
#define GST_IS_CARIDTOY_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CARIDTOY))

typedef struct _GstCaridtoy GstCaridtoy;
typedef struct _GstCaridtoyClass GstCaridtoyClass;

struct _GstCaridtoy
{
  GstBaseTransform base_transform;

  int wavelet_type;
  int level;

  CaridEncoder *encoder;
  CaridDecoder *decoder;
};

struct _GstCaridtoyClass
{
  GstBaseTransformClass parent_class;
};


/* GstCaridtoy signals and args */
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

static void gst_caridtoy_base_init (gpointer g_class);
static void gst_caridtoy_class_init (gpointer g_class,
    gpointer class_data);
static void gst_caridtoy_init (GTypeInstance * instance, gpointer g_class);

static void gst_caridtoy_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_caridtoy_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_caridtoy_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf);

static GstStaticPadTemplate gst_caridtoy_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_caridtoy_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GType
gst_caridtoy_get_type (void)
{
  static GType compress_type = 0;

  if (!compress_type) {
    static const GTypeInfo compress_info = {
      sizeof (GstCaridtoyClass),
      gst_caridtoy_base_init,
      NULL,
      gst_caridtoy_class_init,
      NULL,
      NULL,
      sizeof (GstCaridtoy),
      0,
      gst_caridtoy_init,
    };

    compress_type = g_type_register_static (GST_TYPE_BASE_TRANSFORM,
        "GstCaridtoy", &compress_info, 0);
  }
  return compress_type;
}


static void
gst_caridtoy_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Video Filter Template",
      "Filter/Effect/Video",
      "Template for a video filter",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  //GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_caridtoy_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_caridtoy_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_caridtoy_class_init (gpointer g_class, gpointer class_data)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gobject_class->set_property = gst_caridtoy_set_property;
  gobject_class->get_property = gst_caridtoy_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_transform_class->transform_ip = gst_caridtoy_transform_ip;
}

static void
gst_caridtoy_init (GTypeInstance * instance, gpointer g_class)
{
  GstCaridtoy *compress = GST_CARIDTOY (instance);

  GST_DEBUG ("gst_caridtoy_init");

  compress->encoder = carid_encoder_new ();
  compress->decoder = carid_decoder_new ();
}

static void
gst_caridtoy_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCaridtoy *src;

  g_return_if_fail (GST_IS_CARIDTOY (object));
  src = GST_CARIDTOY (object);

  GST_DEBUG ("gst_caridtoy_set_property");
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
gst_caridtoy_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstCaridtoy *src;

  g_return_if_fail (GST_IS_CARIDTOY (object));
  src = GST_CARIDTOY (object);

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
gst_caridtoy_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf)
{
  GstCaridtoy *compress;
  int width;
  int height;
  CaridBuffer *input_buffer;
  CaridBuffer *encoded_buffer;
  CaridBuffer *decoded_buffer;

  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "height", &height);

  g_return_val_if_fail (GST_IS_CARIDTOY (base_transform), GST_FLOW_ERROR);
  compress = GST_CARIDTOY (base_transform);

  carid_encoder_set_size (compress->encoder, width, height);
  carid_encoder_set_wavelet_type (compress->encoder, compress->wavelet_type);

  if (1) {
  input_buffer = carid_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  encoded_buffer = carid_encoder_encode (compress->encoder, input_buffer);
  carid_buffer_unref (input_buffer);

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


  decoded_buffer = carid_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  carid_decoder_set_output_buffer (compress->decoder, decoded_buffer);
  carid_decoder_decode (compress->decoder, encoded_buffer);
  } else {
    int16_t *frame_data;
    int16_t *tmp;

    frame_data = g_malloc (width*height*2);
    tmp = g_malloc (2048);
    
    oil_convert_s16_u8 (frame_data, GST_BUFFER_DATA(buf), width*height);
    carid_wavelet_transform_2d (compress->wavelet_type,
        frame_data, width*2, width, height, tmp);
    carid_wavelet_transform_2d (compress->wavelet_type,
        frame_data, width*2*2, width/2, height/2, tmp);


    carid_wavelet_inverse_transform_2d (compress->wavelet_type,
        frame_data, width*2*2, width/2, height/2, tmp);
    carid_wavelet_inverse_transform_2d (compress->wavelet_type,
        frame_data, width*2, width, height, tmp);
    oil_convert_u8_s16 (GST_BUFFER_DATA(buf), frame_data, width*height);

    g_free(frame_data);
    g_free(tmp);
  }

  return GST_FLOW_OK;
}

