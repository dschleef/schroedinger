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

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <string.h>
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_CARID_DEC \
  (gst_carid_dec_get_type())
#define GST_CARID_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CARID_DEC,GstCaridDec))
#define GST_CARID_DEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CARID_DEC,GstCaridDecClass))
#define GST_IS_CARID_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CARID_DEC))
#define GST_IS_CARID_DEC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CARID_DEC))

typedef struct _GstCaridDec GstCaridDec;
typedef struct _GstCaridDecClass GstCaridDecClass;

struct _GstCaridDec
{
  GstElement element;

  GstPad *sinkpad, *srcpad;

  int wavelet_type;
  int level;

  CaridDecoder *decoder;
};

struct _GstCaridDecClass
{
  GstElementClass element_class;
};


/* GstCaridDec signals and args */
enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0
};

static void gst_carid_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_carid_dec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_carid_dec_chain (GstPad *pad, GstBuffer *buf);

static GstStaticPadTemplate gst_carid_dec_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

static GstStaticPadTemplate gst_carid_dec_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GST_BOILERPLATE (GstCaridDec, gst_carid_dec, GstElement, GST_TYPE_ELEMENT);

static void
gst_carid_dec_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Dirac Decoder",
      "Codec/Decoder/Video",
      "Decode Dirac streams",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_carid_dec_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_carid_dec_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_carid_dec_class_init (GstCaridDecClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_carid_dec_set_property;
  gobject_class->get_property = gst_carid_dec_get_property;

  //element_class->change_state = gst_carid_dec_change_state;
}

static void
gst_carid_dec_init (GstCaridDec *carid_dec, GstCaridDecClass *klass)
{
  GST_DEBUG ("gst_carid_dec_init");

  carid_dec->decoder = carid_decoder_new ();

  carid_dec->sinkpad = gst_pad_new_from_static_template (&gst_carid_dec_sink_template, "sink");
  gst_pad_set_chain_function (carid_dec->sinkpad, gst_carid_dec_chain);
  //gst_pad_set_event_function (carid_dec->sinkpad, gst_carid_dec_sink_event);
  gst_element_add_pad (GST_ELEMENT(carid_dec), carid_dec->sinkpad);

  carid_dec->srcpad = gst_pad_new_from_static_template (&gst_carid_dec_src_template, "src");
  gst_element_add_pad (GST_ELEMENT(carid_dec), carid_dec->srcpad);
}

static void
gst_carid_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCaridDec *src;

  g_return_if_fail (GST_IS_CARID_DEC (object));
  src = GST_CARID_DEC (object);

  GST_DEBUG ("gst_carid_dec_set_property");
  switch (prop_id) {
    default:
      break;
  }
}

static void
gst_carid_dec_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstCaridDec *src;

  g_return_if_fail (GST_IS_CARID_DEC (object));
  src = GST_CARID_DEC (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_carid_buffer_free (CaridBuffer *buffer, void *priv)
{
  gst_buffer_unref (GST_BUFFER (priv));
}

static GstFlowReturn
gst_carid_dec_chain (GstPad *pad, GstBuffer *buf)
{
  GstCaridDec *carid_dec;
  CaridBuffer *input_buffer;
  CaridBuffer *output_buffer;
  GstBuffer *outbuf;
  GstFlowReturn ret;

  carid_dec = GST_CARID_DEC (GST_PAD_PARENT (pad));

  ret = gst_pad_alloc_buffer_and_set_caps (carid_dec->srcpad,
      GST_BUFFER_OFFSET_NONE, 1000,
      GST_PAD_CAPS (carid_dec->srcpad), &outbuf);
  if (ret != GST_FLOW_OK) {
    return ret;
  }

  input_buffer = carid_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  input_buffer->free = gst_carid_buffer_free;
  input_buffer->priv = buf;

  if (carid_decoder_is_rap (input_buffer)) {
    carid_decoder_decode (carid_dec->decoder, input_buffer);
    
    return GST_FLOW_OK;
  }

  output_buffer = carid_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  output_buffer->free = gst_carid_buffer_free;
  output_buffer->priv = buf;

  carid_decoder_set_output_buffer (carid_dec->decoder, output_buffer);

  return GST_FLOW_OK;
}


