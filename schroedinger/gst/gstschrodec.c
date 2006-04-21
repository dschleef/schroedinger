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
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_SCHRO_DEC \
  (gst_schro_dec_get_type())
#define GST_SCHRO_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHRO_DEC,GstSchroDec))
#define GST_SCHRO_DEC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHRO_DEC,GstSchroDecClass))
#define GST_IS_SCHRO_DEC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHRO_DEC))
#define GST_IS_SCHRO_DEC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHRO_DEC))

typedef struct _GstSchroDec GstSchroDec;
typedef struct _GstSchroDecClass GstSchroDecClass;

struct _GstSchroDec
{
  GstElement element;

  GstPad *sinkpad, *srcpad;

  int wavelet_type;
  int level;

  SchroDecoder *decoder;

  int n_frames;
};

struct _GstSchroDecClass
{
  GstElementClass element_class;
};


/* GstSchroDec signals and args */
enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0
};

static void gst_schro_dec_finalize (GObject *object);
static void gst_schro_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schro_dec_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_schro_dec_chain (GstPad *pad, GstBuffer *buf);

static GstStaticPadTemplate gst_schro_dec_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

static GstStaticPadTemplate gst_schro_dec_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GST_BOILERPLATE (GstSchroDec, gst_schro_dec, GstElement, GST_TYPE_ELEMENT);

static void
gst_schro_dec_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Dirac Decoder",
      "Codec/Decoder/Video",
      "Decode Dirac streams",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_dec_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_dec_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_schro_dec_class_init (GstSchroDecClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_schro_dec_set_property;
  gobject_class->get_property = gst_schro_dec_get_property;
  gobject_class->finalize = gst_schro_dec_finalize;

  //element_class->change_state = gst_schro_dec_change_state;
}

static void
gst_schro_dec_init (GstSchroDec *schro_dec, GstSchroDecClass *klass)
{
  GST_DEBUG ("gst_schro_dec_init");

  schro_dec->decoder = schro_decoder_new ();

  schro_dec->sinkpad = gst_pad_new_from_static_template (&gst_schro_dec_sink_template, "sink");
  gst_pad_set_chain_function (schro_dec->sinkpad, gst_schro_dec_chain);
  //gst_pad_set_event_function (schro_dec->sinkpad, gst_schro_dec_sink_event);
  gst_element_add_pad (GST_ELEMENT(schro_dec), schro_dec->sinkpad);

  schro_dec->srcpad = gst_pad_new_from_static_template (&gst_schro_dec_src_template, "src");
  gst_element_add_pad (GST_ELEMENT(schro_dec), schro_dec->srcpad);
}

static void
gst_schro_dec_finalize (GObject *object)
{
  GstSchroDec *schro_dec;

  g_return_if_fail (GST_IS_SCHRO_DEC (object));
  schro_dec = GST_SCHRO_DEC (object);

  if (schro_dec->decoder) {
    schro_decoder_free (schro_dec->decoder);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_schro_dec_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSchroDec *src;

  g_return_if_fail (GST_IS_SCHRO_DEC (object));
  src = GST_SCHRO_DEC (object);

  GST_DEBUG ("gst_schro_dec_set_property");
  switch (prop_id) {
    default:
      break;
  }
}

static void
gst_schro_dec_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSchroDec *src;

  g_return_if_fail (GST_IS_SCHRO_DEC (object));
  src = GST_SCHRO_DEC (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_schro_buffer_free (SchroBuffer *buffer, void *priv)
{
  gst_buffer_unref (GST_BUFFER (priv));
}

static SchroBuffer *
gst_schro_wrap_gst_buffer (GstBuffer *buffer)
{
  SchroBuffer *schrobuf;

  schrobuf = schro_buffer_new_with_data (GST_BUFFER_DATA (buffer),
      GST_BUFFER_SIZE (buffer));
  schrobuf->free = gst_schro_buffer_free;
  schrobuf->priv = buffer;

  return schrobuf;
}

static void
gst_schro_frame_free (SchroFrame *frame, void *priv)
{
  //gst_buffer_unref (GST_BUFFER (priv));
}

static SchroFrame *
gst_schro_wrap_frame (GstSchroDec *schro_dec, GstBuffer *buffer)
{
  SchroFrame *frame;

  frame = schro_frame_new_I420 (GST_BUFFER_DATA (buffer),
      schro_dec->decoder->params.width, schro_dec->decoder->params.height);
  frame->free = gst_schro_frame_free;
  frame->priv = buffer;

  return frame;
}

static GstFlowReturn
gst_schro_dec_chain (GstPad *pad, GstBuffer *buf)
{
  GstSchroDec *schro_dec;
  SchroBuffer *input_buffer;
  SchroFrame *frame;
  GstBuffer *outbuf;
  GstFlowReturn ret;

  schro_dec = GST_SCHRO_DEC (GST_PAD_PARENT (pad));

  input_buffer = gst_schro_wrap_gst_buffer (buf);

  if (schro_decoder_is_rap (input_buffer)) {
    GstCaps *caps;

    GST_DEBUG("random access point");
    schro_decoder_decode (schro_dec->decoder, input_buffer);

    caps = gst_caps_new_simple ("video/x-raw-yuv",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('I','4','2','0'),
        "width", G_TYPE_INT, schro_dec->decoder->params.width,
        "height", G_TYPE_INT, schro_dec->decoder->params.height,
        "framerate", GST_TYPE_FRACTION,
        schro_dec->decoder->params.frame_rate_numerator,
        schro_dec->decoder->params.frame_rate_denominator,
        "pixel-aspect-ratio", GST_TYPE_FRACTION,
        schro_dec->decoder->params.pixel_aspect_ratio_numerator,
        schro_dec->decoder->params.pixel_aspect_ratio_denominator,
        NULL);

    GST_DEBUG("setting caps %" GST_PTR_FORMAT, caps);

    gst_pad_set_caps (schro_dec->srcpad, caps);

    gst_caps_unref (caps);
    
    return GST_FLOW_OK;
  } else {
    int size;

    GST_DEBUG("not random access point");
    size = schro_dec->decoder->params.width * schro_dec->decoder->params.height;
    size += size/2;
    ret = gst_pad_alloc_buffer_and_set_caps (schro_dec->srcpad,
        GST_BUFFER_OFFSET_NONE, size,
        GST_PAD_CAPS (schro_dec->srcpad), &outbuf);
    if (ret != GST_FLOW_OK) {
      GST_ERROR("could not allocate buffer for pad");
      return ret;
    }

    GST_BUFFER_TIMESTAMP(outbuf) = 
      (schro_dec->n_frames *
       schro_dec->decoder->params.frame_rate_denominator * GST_SECOND)/
       schro_dec->decoder->params.frame_rate_numerator;
    GST_BUFFER_DURATION(outbuf) = 
      (schro_dec->decoder->params.frame_rate_denominator * GST_SECOND)/
       schro_dec->decoder->params.frame_rate_numerator;
    GST_BUFFER_OFFSET(outbuf) = schro_dec->n_frames;

    schro_dec->n_frames++;

    frame = gst_schro_wrap_frame (schro_dec, outbuf);

    schro_decoder_set_output_frame (schro_dec->decoder, frame);

    schro_decoder_decode (schro_dec->decoder, input_buffer);

    ret = gst_pad_push (schro_dec->srcpad, outbuf);
    
    return ret;
  }

  return GST_FLOW_OK;
}


