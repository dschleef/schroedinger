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
#include <gst/video/video.h>
#include <string.h>
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_CARID_ENC \
  (gst_carid_enc_get_type())
#define GST_CARID_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CARID_ENC,GstCaridEnc))
#define GST_CARID_ENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CARID_ENC,GstCaridEncClass))
#define GST_IS_CARID_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CARID_ENC))
#define GST_IS_CARID_ENC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CARID_ENC))

typedef struct _GstCaridEnc GstCaridEnc;
typedef struct _GstCaridEncClass GstCaridEncClass;

struct _GstCaridEnc
{
  GstElement element;

  GstPad *sinkpad;
  GstPad *srcpad;

  /* parameters */
  int wavelet_type;
  int level;

  /* video properties */
  int width;
  int height;
  int fps_n, fps_d;
  int par_n, par_d;

  /* state */
  gboolean sent_header;
  int n_frames;
  int offset;
  int granulepos;

  CaridEncoder *encoder;
};

struct _GstCaridEncClass
{
  GstElementClass parent_class;
};


enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0,
  ARG_WAVELET_TYPE,
  ARG_LEVEL
};

static void gst_carid_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_carid_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_carid_enc_sink_setcaps (GstPad *pad, GstCaps *caps);
static gboolean gst_carid_enc_sink_event (GstPad *pad, GstEvent *event);
static GstFlowReturn gst_carid_enc_chain (GstPad *pad, GstBuffer *buf);
static GstStateChangeReturn gst_carid_enc_change_state (GstElement *element,
    GstStateChange transition);

static GstStaticPadTemplate gst_carid_enc_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_carid_enc_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

GST_BOILERPLATE (GstCaridEnc, gst_carid_enc, GstElement, GST_TYPE_ELEMENT);

static void
gst_carid_enc_base_init (gpointer g_class)
{
  static GstElementDetails carid_enc_details =
      GST_ELEMENT_DETAILS ("Dirac Encoder",
      "Coder/Encoder/Video",
      "Encode raw YUV video into Dirac stream",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_carid_enc_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_carid_enc_sink_template));

  gst_element_class_set_details (element_class, &carid_enc_details);
}

static void
gst_carid_enc_class_init (GstCaridEncClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_carid_enc_set_property;
  gobject_class->get_property = gst_carid_enc_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 2, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  gstelement_class->change_state = gst_carid_enc_change_state;
}

static void
gst_carid_enc_init (GstCaridEnc *carid_enc, GstCaridEncClass *klass)
{
  GST_DEBUG ("gst_carid_enc_init");

  carid_enc->encoder = carid_encoder_new ();
  carid_enc->wavelet_type = 2;

  carid_enc->sinkpad = gst_pad_new_from_static_template (&gst_carid_enc_sink_template, "sink");
  gst_pad_set_chain_function (carid_enc->sinkpad, gst_carid_enc_chain);
  gst_pad_set_event_function (carid_enc->sinkpad, gst_carid_enc_sink_event);
  gst_pad_set_setcaps_function (carid_enc->sinkpad, gst_carid_enc_sink_setcaps);
  gst_element_add_pad (GST_ELEMENT(carid_enc), carid_enc->sinkpad);

  carid_enc->srcpad = gst_pad_new_from_static_template (&gst_carid_enc_src_template, "src");
  gst_element_add_pad (GST_ELEMENT(carid_enc), carid_enc->srcpad);

}

static gboolean
gst_carid_enc_sink_setcaps (GstPad *pad, GstCaps *caps)
{
  GstStructure *structure;
  GstCaridEnc *carid_enc = GST_CARID_ENC (gst_pad_get_parent (pad));

  structure = gst_caps_get_structure (caps, 0);

  gst_structure_get_int (structure, "width", &carid_enc->width);
  gst_structure_get_int (structure, "height", &carid_enc->height);
  gst_structure_get_fraction (structure, "framerate", &carid_enc->fps_n,
      &carid_enc->fps_d);
  gst_structure_get_fraction (structure, "pixel-aspect-ratio",
      &carid_enc->par_n, &carid_enc->par_d);

  /* FIXME init encoder */

  carid_encoder_set_size (carid_enc->encoder, carid_enc->width,
      carid_enc->height);

  return TRUE;
}

static void
gst_carid_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCaridEnc *src;

  g_return_if_fail (GST_IS_CARID_ENC (object));
  src = GST_CARID_ENC (object);

  GST_DEBUG ("gst_carid_enc_set_property");
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
gst_carid_enc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstCaridEnc *src;

  g_return_if_fail (GST_IS_CARID_ENC (object));
  src = GST_CARID_ENC (object);

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

static gboolean
gst_carid_enc_sink_event (GstPad *pad, GstEvent *event)
{
  GstCaridEnc *carid_enc;
  gboolean ret;

  carid_enc = GST_CARID_ENC (GST_PAD_PARENT (pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      /* flush */
      ret = gst_pad_push_event (carid_enc->srcpad, event);
      break;
    default:
      ret = gst_pad_push_event (carid_enc->srcpad, event);
      break;
  }

  return ret;
}

static void
gst_carid_buffer_free (CaridBuffer *buffer, void *priv)
{
  gst_buffer_unref (GST_BUFFER(priv));
}

static GstCaps *
gst_carid_enc_set_header_on_caps (GstCaps * caps, GstBuffer * buf1)
{
  GstStructure *structure;
  GValue array = { 0 };
  GValue value = { 0 };

  caps = gst_caps_make_writable (caps);
  structure = gst_caps_get_structure (caps, 0);

  /* mark buffers */
  GST_BUFFER_FLAG_SET (buf1, GST_BUFFER_FLAG_IN_CAPS);

  /* put buffers in a fixed list */
  g_value_init (&array, GST_TYPE_ARRAY);
  g_value_init (&value, GST_TYPE_BUFFER);
  gst_value_set_buffer (&value, buf1);
  gst_value_array_append_value (&array, &value);
  g_value_unset (&value);
  gst_structure_set_value (structure, "streamheader", &array);
  g_value_unset (&array);

  return caps;
}

static GstFlowReturn
gst_carid_enc_chain (GstPad *pad, GstBuffer *buf)
{
  GstCaridEnc *carid_enc;
  CaridBuffer *input_buffer;
  CaridBuffer *encoded_buffer;
  GstBuffer *outbuf;
  GstFlowReturn ret;

  carid_enc = GST_CARID_ENC (GST_PAD_PARENT (pad));

  carid_encoder_set_wavelet_type (carid_enc->encoder, carid_enc->wavelet_type);

  if (carid_enc->sent_header == 0) {
    GstCaps *caps;

    encoded_buffer = carid_encoder_encode (carid_enc->encoder);

    GST_ERROR ("encoder produced %d bytes", encoded_buffer->length);

    outbuf = gst_buffer_new_and_alloc (encoded_buffer->length);

    memcpy (GST_BUFFER_DATA (outbuf), encoded_buffer->data,
        encoded_buffer->length);
    GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_IN_CAPS);
    GST_BUFFER_OFFSET_END (outbuf) = 0;

    caps = gst_pad_get_caps (carid_enc->srcpad);
    caps = gst_carid_enc_set_header_on_caps (caps, outbuf);

    gst_pad_set_caps (carid_enc->srcpad, caps);

    gst_buffer_set_caps (outbuf, caps);

    ret = gst_pad_push (carid_enc->srcpad, outbuf);
    if (ret!= GST_FLOW_OK) return ret;

    carid_enc->sent_header = 1;
  }

  input_buffer = carid_buffer_new_with_data (GST_BUFFER_DATA (buf),
      GST_BUFFER_SIZE (buf));
  input_buffer->free = gst_carid_buffer_free;
  input_buffer->priv = buf;

  GST_DEBUG ("pushing buffer");
  carid_encoder_push_buffer (carid_enc->encoder, input_buffer);

  while (1) {
    encoded_buffer = carid_encoder_encode (carid_enc->encoder);
    if (encoded_buffer == NULL) break;

    ret = gst_pad_alloc_buffer_and_set_caps (carid_enc->srcpad,
        GST_BUFFER_OFFSET_NONE, encoded_buffer->length,
        GST_PAD_CAPS (carid_enc->srcpad), &outbuf);
    if (ret != GST_FLOW_OK) {
      carid_buffer_unref (encoded_buffer);
      return ret;
    }

    memcpy (GST_BUFFER_DATA (outbuf), encoded_buffer->data, encoded_buffer->length);
    GST_BUFFER_OFFSET (outbuf) = carid_enc->offset;
    GST_BUFFER_OFFSET_END (outbuf) = carid_enc->granulepos;
    GST_BUFFER_TIMESTAMP (outbuf) = (carid_enc->n_frames * GST_SECOND * carid_enc->fps_d)/carid_enc->fps_n;
    GST_BUFFER_DURATION (outbuf) = (GST_SECOND * carid_enc->fps_d)/carid_enc->fps_n;

    carid_enc->offset += encoded_buffer->length;
    carid_enc->granulepos++;
    carid_enc->n_frames++;

    carid_buffer_unref (encoded_buffer);
    
    ret = gst_pad_push (carid_enc->srcpad, outbuf);

    if (ret!= GST_FLOW_OK) return ret;
  }
  return GST_FLOW_OK;
}

static GstStateChangeReturn
gst_carid_enc_change_state (GstElement *element, GstStateChange transition)
{
  GstCaridEnc *carid_enc;
  GstStateChangeReturn ret;

  carid_enc = GST_CARID_ENC (element);

  switch (transition) {
    default:
      break;
  }

  ret = parent_class->change_state (element, transition);

  switch (transition) {
    default:
      break;
  }

  return ret;
}

