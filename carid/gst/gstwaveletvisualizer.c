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
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include <string.h>
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <math.h>

#define GST_TYPE_WAVELETVISUALIZER \
  (gst_waveletvisualizer_get_type())
#define GST_WAVELETVISUALIZER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_WAVELETVISUALIZER,GstWaveletvisualizer))
#define GST_WAVELETVISUALIZER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_WAVELETVISUALIZER,GstWaveletvisualizerClass))
#define GST_IS_WAVELETVISUALIZER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_WAVELETVISUALIZER))
#define GST_IS_WAVELETVISUALIZER_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_WAVELETVISUALIZER))

typedef struct _GstWaveletvisualizerComponent GstWaveletvisualizerComponent;

typedef struct _GstWaveletvisualizer GstWaveletvisualizer;
typedef struct _GstWaveletvisualizerClass GstWaveletvisualizerClass;

struct _GstWaveletvisualizerComponent
{
  int width;
  int height;

  int rwidth;
  int rheight;

  int16_t *data;
};

struct _GstWaveletvisualizer
{
  GstPushSrc base_src;

  int wavelet_type;
  int level;

  int height;
  int width;

  GstWaveletvisualizerComponent components[3];
  int16_t *tmpdata;

  gboolean inited;

  CaridDecoder *decoder;
};

struct _GstWaveletvisualizerClass
{
  GstPushSrcClass parent_class;
};


/* GstWaveletvisualizer signals and args */
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

static GstStaticPadTemplate gst_waveletvisualizer_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstFlowReturn gst_waveletvisualizer_create (GstPushSrc * base_src, GstBuffer **buffer);
static void gst_waveletvisualizer_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_waveletvisualizer_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static gboolean gst_waveletvisualizer_setcaps (GstBaseSrc * base_src, GstCaps *caps);
static void gst_waveletvisualizer_fixate (GstPad *pad, GstCaps *caps);

GST_BOILERPLATE (GstWaveletvisualizer, gst_waveletvisualizer, GstPushSrc,
    GST_TYPE_PUSH_SRC);

static void
gst_waveletvisualizer_base_init (gpointer g_class)
{
  static GstElementDetails waveletvisualizer_details =
      GST_ELEMENT_DETAILS ("Video Filter Template",
      "Filter/Effect/Video",
      "Template for a video filter",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_waveletvisualizer_src_template));

  gst_element_class_set_details (element_class, &waveletvisualizer_details);
}

static void
gst_waveletvisualizer_class_init (GstWaveletvisualizerClass *klass)
{
  GObjectClass *gobject_class;
  GstBaseSrcClass *base_src_class;
  GstPushSrcClass *push_src_class;

  gobject_class = G_OBJECT_CLASS (klass);
  base_src_class = GST_BASE_SRC_CLASS (klass);
  push_src_class = GST_PUSH_SRC_CLASS (klass);

  gobject_class->set_property = gst_waveletvisualizer_set_property;
  gobject_class->get_property = gst_waveletvisualizer_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_src_class->set_caps = gst_waveletvisualizer_setcaps;

  push_src_class->create = gst_waveletvisualizer_create;
}

static void
gst_waveletvisualizer_init (GstWaveletvisualizer *waveletvisualizer,
    GstWaveletvisualizerClass *klass)
{
  GstPad *pad = GST_BASE_SRC_PAD (waveletvisualizer);

  GST_DEBUG ("gst_waveletvisualizer_init");

  gst_pad_set_fixatecaps_function (pad, gst_waveletvisualizer_fixate);

  waveletvisualizer->decoder = carid_decoder_new ();

  waveletvisualizer->inited = FALSE;
}

static void
gst_waveletvisualizer_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstWaveletvisualizer *src;

  g_return_if_fail (GST_IS_WAVELETVISUALIZER (object));
  src = GST_WAVELETVISUALIZER (object);

  GST_DEBUG ("gst_waveletvisualizer_set_property");
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
gst_waveletvisualizer_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstWaveletvisualizer *src;

  g_return_if_fail (GST_IS_WAVELETVISUALIZER (object));
  src = GST_WAVELETVISUALIZER (object);

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
gst_waveletvisualizer_setcaps (GstBaseSrc * base_src, GstCaps *caps)
{
  GstWaveletvisualizer *waveletvisualizer;
  gboolean res;
  int width;
  int height;
  GstStructure *structure;

  g_return_val_if_fail (GST_IS_WAVELETVISUALIZER (base_src), FALSE);
  waveletvisualizer = GST_WAVELETVISUALIZER (base_src);

  structure = gst_caps_get_structure (caps, 0);

  res = gst_structure_get_int (structure, "width", &width);
  res &= gst_structure_get_int (structure, "height", &height);

  if (!res) return FALSE;

  waveletvisualizer->height = height;
  waveletvisualizer->width = width;

  return TRUE;
}

static void
gst_waveletvisualizer_fixate (GstPad *pad, GstCaps *caps)
{
  GstStructure *structure;

  structure = gst_caps_get_structure (caps, 0);

  gst_structure_fixate_field_nearest_int (structure, "width", 320);
  gst_structure_fixate_field_nearest_int (structure, "height", 240);
  gst_structure_fixate_field_nearest_fraction (structure, "framerate", 30, 1);
}

static void do_init (GstWaveletvisualizerComponent *c, int w, int h);
static void do_update (GstWaveletvisualizerComponent *c, int dc_val);
static void do_transform (GstWaveletvisualizerComponent *c, CaridDecoder *decoder, int16_t *tmpdata, uint8_t *dest);

static GstFlowReturn
gst_waveletvisualizer_create (GstPushSrc * push_src,
    GstBuffer **buffer)
{
  GstWaveletvisualizer *wv;
  int val;
  int size;
  GstBuffer *outbuf;

  g_return_val_if_fail (GST_IS_WAVELETVISUALIZER (push_src), GST_FLOW_ERROR);
  wv = GST_WAVELETVISUALIZER (push_src);

  size = wv->width * wv->height +
    ((wv->width/2) * (wv->height/2)) *2;
  gst_pad_alloc_buffer_and_set_caps (GST_BASE_SRC_PAD (push_src),
      GST_BUFFER_OFFSET_NONE, size, GST_PAD_CAPS (GST_BASE_SRC_PAD (push_src)),
      &outbuf);

  //GST_BUFFER_TIMESTAMP (outbuf) = 0;
  //GST_BUFFER_OFFSET (outbuf) = 0;

  if (!wv->inited) {
    do_init(wv->components + 0, wv->width, wv->height);
    do_init(wv->components + 1, wv->width/2, wv->height/2);
    do_init(wv->components + 2, wv->width/2, wv->height/2);
    wv->tmpdata = malloc(wv->components[0].rwidth*wv->components[0].rheight*2);
    wv->inited = TRUE;
  }

  if (wv->wavelet_type == 0) {
    val = 128*pow(1.23,12.0);
  } else {
    val = 128;
  }

  do_update (wv->components + 0, val);
  do_update (wv->components + 1, val);
  do_update (wv->components + 2, val);

  carid_decoder_set_wavelet_type (wv->decoder, wv->wavelet_type);

  do_transform (wv->components + 0, wv->decoder, wv->tmpdata,
      GST_BUFFER_DATA (outbuf));
  do_transform (wv->components + 1, wv->decoder, wv->tmpdata,
      GST_BUFFER_DATA (outbuf) + wv->height*wv->width);
  do_transform (wv->components + 2, wv->decoder, wv->tmpdata,
      GST_BUFFER_DATA (outbuf) + wv->height*wv->width +
      wv->height*wv->width/4);

  *buffer = outbuf;

  return GST_FLOW_OK;
}


static void do_init (GstWaveletvisualizerComponent *c, int w, int h)
{
  int n;
  int i;

  c->width = w;
  c->height = h;

  c->rwidth = (w + 63) & (~63);
  c->rheight = (h + 63) & (~63);

  n = c->rwidth*c->rheight;

  c->data = malloc(n*2);
  for(i=0;i<n;i++) {
    c->data[i] = g_random_int_range(-10,10);
  }
}

static void do_update (GstWaveletvisualizerComponent *c, int dc_val)
{
  int i;
  int j;
  int16_t *data = c->data;
  int n = c->rwidth*c->rheight;

  /* let the coefficients wander a bit */
  
#if 0
  for(i=0;i<n;i++) {
    if (data[i] > 0) data[i] -= (data[i]>>5);
    if (data[i] < 0) data[i] += (-data[i]>>5);
    data[i] += g_random_int_range(-3,4);
  }
#endif
#if 0
  for(i=0;i<n;i++) {
    data[i] = g_random_int_range(-32,32);
  }
#endif
  for(i=0;i<n;i++) {
    if (data[i] > 0) data[i] -= (data[i]>>3);
    if (data[i] < 0) data[i] += (-data[i]>>3);
    data[i] += g_random_int_range(-15,16);
  }

  /* restore the DC area to mid-scale */

  for(i=0;i<c->rheight>>6;i++) {
    for(j=0;j<c->rwidth>>6;j++) {
      //data[i*width + j] = g_random_int_range(0, 255);
      data[i*c->rwidth + j] = dc_val;
    }
  }
}

static void
do_transform (GstWaveletvisualizerComponent *c, CaridDecoder *decoder,
    int16_t *tmpdata, uint8_t *dest)
{
  int n;
  CaridBuffer *encoded_buffer;
  CaridBuffer *decoded_buffer;

  n = c->rwidth*c->rheight;

  oil_memcpy(tmpdata, c->data, n*2);

  encoded_buffer = carid_buffer_new_with_data (tmpdata, c->width*c->height*2);
  carid_decoder_set_size (decoder, c->width, c->height);
  decoded_buffer = carid_buffer_new_with_data (dest,
      c->width * c->height);
  carid_decoder_set_output_buffer (decoder, decoded_buffer);
  carid_decoder_decode (decoder, encoded_buffer);
}

