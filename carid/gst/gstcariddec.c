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

  int n_frames;
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

static void gst_carid_dec_finalize (GObject *object);
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
  gobject_class->finalize = gst_carid_dec_finalize;

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
gst_carid_dec_finalize (GObject *object)
{
  GstCaridDec *carid_dec;

  g_return_if_fail (GST_IS_CARID_DEC (object));
  carid_dec = GST_CARID_DEC (object);

  if (carid_dec->decoder) {
    carid_decoder_free (carid_dec->decoder);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
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

static CaridBuffer *
gst_carid_wrap_gst_buffer (GstBuffer *buffer)
{
  CaridBuffer *caridbuf;

  caridbuf = carid_buffer_new_with_data (GST_BUFFER_DATA (buffer),
      GST_BUFFER_SIZE (buffer));
  caridbuf->free = gst_carid_buffer_free;
  caridbuf->priv = buffer;

  return caridbuf;
}

static void
gst_carid_frame_free (CaridFrame *frame, void *priv)
{
  gst_buffer_unref (GST_BUFFER (priv));
}

static CaridFrame *
gst_carid_wrap_frame (GstCaridDec *carid_dec, GstBuffer *buffer)
{
  CaridFrame *frame;

  frame = carid_frame_new_I420 (GST_BUFFER_DATA (buffer),
      carid_dec->decoder->params.width, carid_dec->decoder->params.height);
  frame->free = gst_carid_frame_free;
  frame->priv = buffer;

  return frame;
}

static GstFlowReturn
gst_carid_dec_chain (GstPad *pad, GstBuffer *buf)
{
  GstCaridDec *carid_dec;
  CaridBuffer *input_buffer;
  CaridFrame *frame;
  GstBuffer *outbuf;
  GstFlowReturn ret;

  carid_dec = GST_CARID_DEC (GST_PAD_PARENT (pad));

  input_buffer = gst_carid_wrap_gst_buffer (buf);

  if (carid_decoder_is_rap (input_buffer)) {
    GstCaps *caps;

    GST_DEBUG("random access point");
    carid_decoder_decode (carid_dec->decoder, input_buffer);

    caps = gst_caps_new_simple ("video/x-raw-yuv",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('I','4','2','0'),
        "width", G_TYPE_INT, carid_dec->decoder->params.width,
        "height", G_TYPE_INT, carid_dec->decoder->params.height,
        "framerate", GST_TYPE_FRACTION,
        carid_dec->decoder->params.frame_rate_numerator,
        carid_dec->decoder->params.frame_rate_denominator,
        "pixel-aspect-ratio", GST_TYPE_FRACTION,
        carid_dec->decoder->params.pixel_aspect_ratio_numerator,
        carid_dec->decoder->params.pixel_aspect_ratio_denominator,
        NULL);

    GST_DEBUG("setting caps %" GST_PTR_FORMAT, caps);

    gst_pad_set_caps (carid_dec->srcpad, caps);

    gst_caps_unref (caps);
    
    return GST_FLOW_OK;
  } else {
    int size;

    GST_DEBUG("not random access point");
    size = carid_dec->decoder->params.width * carid_dec->decoder->params.height;
    size += size/2;
    ret = gst_pad_alloc_buffer_and_set_caps (carid_dec->srcpad,
        GST_BUFFER_OFFSET_NONE, size,
        GST_PAD_CAPS (carid_dec->srcpad), &outbuf);
    if (ret != GST_FLOW_OK) {
      GST_ERROR("could not allocate buffer for pad");
      return ret;
    }

    GST_BUFFER_TIMESTAMP(outbuf) = 
      (carid_dec->n_frames *
       carid_dec->decoder->params.frame_rate_denominator * GST_SECOND)/
       carid_dec->decoder->params.frame_rate_numerator;
    GST_BUFFER_DURATION(outbuf) = 
      (carid_dec->decoder->params.frame_rate_denominator * GST_SECOND)/
       carid_dec->decoder->params.frame_rate_numerator;
    GST_BUFFER_OFFSET(outbuf) = carid_dec->n_frames;

    carid_dec->n_frames++;

    frame = gst_carid_wrap_frame (carid_dec, outbuf);

    carid_decoder_set_output_frame (carid_dec->decoder, frame);

    carid_decoder_decode (carid_dec->decoder, input_buffer);

    ret = gst_pad_push (carid_dec->srcpad, outbuf);
    
    return ret;
  }

  return GST_FLOW_OK;
}


