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

#undef SCHRO_DISABLE_UNSTABLE_API

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

  SchroVideoFormat format;
  SchroParams params;

  SchroFrame *tmp_frame;
  int16_t *tmpbuf;

  int frame_number;

  int button_x;
  int button_y;

};

struct _GstSchrotoyClass
{
  GstBaseTransformClass parent_class;

  int test_index;

  int quants[20];
  int side0;
  int side1;
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
static gboolean
gst_schrotoy_handle_src_event (GstPad * pad, GstEvent * event);

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
  GstSchrotoyClass *toy_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);
  toy_class = GST_SCHROTOY_CLASS (g_class);

  gobject_class->set_property = gst_schrotoy_set_property;
  gobject_class->get_property = gst_schrotoy_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_transform_class->transform_ip = gst_schrotoy_transform_ip;

#define N 6
  toy_class->side0 = g_random_int_range(0,N);
  do {
    toy_class->side1 = g_random_int_range(0,N);
  } while (toy_class->side0 == toy_class->side1);
  toy_class->quants[0] = 30;
  toy_class->quants[1] = 30;
  toy_class->quants[2] = 30;
  toy_class->quants[3] = 30;
  toy_class->quants[4] = 30;
  toy_class->quants[5] = 30;
  toy_class->quants[6] = 30;
  toy_class->quants[7] = 30;
  toy_class->quants[8] = 30;
  toy_class->quants[9] = 30;
  toy_class->quants[10] = 30;
  toy_class->quants[11] = 30;
}

static void
gst_schrotoy_init (GTypeInstance * instance, gpointer g_class)
{
  GstSchrotoy *compress = GST_SCHROTOY (instance);
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (instance);

  GST_DEBUG ("gst_schrotoy_init");

  gst_pad_set_event_function (btrans->srcpad,
            GST_DEBUG_FUNCPTR (gst_schrotoy_handle_src_event));

  compress->button_x = -1;
  compress->button_y = -1;
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

static gboolean
gst_schrotoy_handle_src_event (GstPad * pad, GstEvent * event)
{
  GstSchrotoy *toy;
  const gchar *type;

  toy = GST_SCHROTOY (GST_PAD_PARENT (pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_NAVIGATION:
    {
      const GstStructure *s = gst_event_get_structure (event);

      type = gst_structure_get_string (s, "event");
      if (g_str_equal (type, "mouse-button-press")) {
        double x, y;
        gst_structure_get_double (s, "pointer_x", &x);
        gst_structure_get_double (s, "pointer_y", &y);
        toy->button_x = x;
        toy->button_y = y;
      }
    }
    default:
      break;
  }
  return gst_pad_event_default (pad, event);
}

static void
quantize_frame (SchroFrame *frame, int *quant_index_0, int *quant_index_1,
    SchroParams *params);

static void
mark_frame (SchroFrame *frame, int val)
{
  int i,j;
  uint8_t *data;
  int y,u,v;
  int x;

  for(x=0;x<8;x++){
    if ((val>>(7-x))&1) {
      y = 255; u = 0; v = 128;
    } else {
      y = 0; u = 255; v = 128;
    }
    for(j=0;j<10;j++){
      data = frame->components[0].data + frame->components[0].stride * j + 10*x;
      for(i=0;i<10;i++){
        data[i] = y;
      }
    }
    for(j=0;j<5;j++){
      data = frame->components[1].data + frame->components[1].stride * j + 5*x;
      for(i=0;i<5;i++){
        data[i] = u;
      }
    }
    for(j=0;j<5;j++){
      data = frame->components[2].data + frame->components[2].stride * j + 5*x;
      for(i=0;i<5;i++){
        data[i] = v;
      }
    }
  }

  data = frame->components[0].data + frame->components[0].stride * 50;
  for(i=0;i<frame->components[0].width;i++) {
    data[i] = 0;
  }
  for(j=50;j<100;j++) {
    data = frame->components[0].data + frame->components[0].stride * j;
    data[frame->components[0].width/2] = 0;
  }
}

static GstFlowReturn
gst_schrotoy_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf)
{
  GstSchrotoy *compress;
  GstSchrotoyClass *compress_class;
  SchroFrame *frame;
  SchroParams *params;
  int mode;
  
  g_return_val_if_fail (GST_IS_SCHROTOY (base_transform), GST_FLOW_ERROR);
  compress = GST_SCHROTOY (base_transform);
  compress_class = G_TYPE_INSTANCE_GET_CLASS (compress, GST_TYPE_SCHROTOY, GstSchrotoyClass);
  params = &compress->params;

  if (compress->format.width == 0) {
    schro_params_set_video_format (&compress->format,
        SCHRO_VIDEO_FORMAT_SD480);

    gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
        "width", &compress->format.width);
    gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
        "height", &compress->format.height);

    params->video_format = &compress->format;
    params->transform_depth = 4;
    params->wavelet_filter_index = SCHRO_WAVELET_5_3;

    schro_params_calculate_iwt_sizes (params);

    compress->tmp_frame =
      schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_S16_420,
          compress->params.iwt_luma_width,
          compress->params.iwt_luma_height);
    compress->tmpbuf = malloc (2*2048);
  }

  frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA(buf),
      params->video_format->width, params->video_format->height);

  schro_frame_convert (compress->tmp_frame, frame);
  schro_frame_iwt_transform (compress->tmp_frame, &compress->params,
      compress->tmpbuf);

  if (compress->button_x >= 0) {
    int value;
    int valid;

    valid = compress->button_y > 50;
    value = compress->button_x > params->video_format->width/2;
    GST_DEBUG("%d %d", valid, value);

    if (valid) {
      if (value) {
        if(compress_class->quants[compress_class->side0]<60) {
          compress_class->quants[compress_class->side1]--;
          compress_class->quants[compress_class->side0]++;
        }
      } else {
        if(compress_class->quants[compress_class->side1]<60) {
          compress_class->quants[compress_class->side0]--;
          compress_class->quants[compress_class->side1]++;
        }
      }
    }

    GST_ERROR("%d %d %d %d %d %d",
        compress_class->quants[0], compress_class->quants[1],
        compress_class->quants[2], compress_class->quants[3],
        compress_class->quants[4], compress_class->quants[5]);


    compress->button_x = -1;
    compress->button_y = -1;

    compress_class->side0 = g_random_int_range(0,N);
    do {
      compress_class->side1 = g_random_int_range(0,N);
    } while (compress_class->side0 == compress_class->side1);
    compress_class->test_index++;
  }

  {
    int i;
    int quant_index_0[13];
    int quant_index_1[13];

    mode = (compress_class->test_index)%4;

    for(i=0;i<13;i++) quant_index_0[i] = 0;
    for(i=0;i<13;i++) quant_index_1[i] = 0;
#if 0
    quant_index_0[0] = 16;
    quant_index_0[1] = 20;
    quant_index_0[2] = 20;
    quant_index_0[3] = 24;
    quant_index_0[4] = 20;
    quant_index_0[5] = 20;
    quant_index_0[6] = 24;
    quant_index_0[7] = 21;
    quant_index_0[8] = 21;
    quant_index_0[9] = 25;
    quant_index_0[10] = 53;
    quant_index_0[11] = 53;
    quant_index_0[12] = 57;
#endif

#if 1
    quant_index_0[1 + compress_class->side0] =
      compress_class->quants[compress_class->side0];
    quant_index_1[1 + compress_class->side1] =
      compress_class->quants[compress_class->side1];
#endif

    quantize_frame (compress->tmp_frame, quant_index_0, quant_index_1,
        &compress->params);
  }

  schro_frame_inverse_iwt_transform (compress->tmp_frame, &compress->params,
      compress->tmpbuf);
  schro_frame_convert (frame, compress->tmp_frame);

  mark_frame (frame, compress_class->test_index);

  compress->frame_number++;
  return GST_FLOW_OK;
}

static int
dequantize (int q, int quant_factor, int quant_offset)
{ 
  if (q == 0) return 0;
  if (q < 0) {
    return -((-q * quant_factor + quant_offset + 2)>>2);
  } else {
    return (q * quant_factor + quant_offset + 2)>>2;
  }
}   
  
static int
quantize (int value, int quant_factor, int quant_offset)
{ 
  unsigned int x;
    
  if (value == 0) return 0;
  if (value < 0) {
    x = (-value)<<2;
    x /= quant_factor;
    value = -x;
  } else { 
    x = value<<2;
    x /= quant_factor;
    value = x;
  }
  return value;
}

static void
quantize_frame (SchroFrame *frame, int *quant_index_0, int *quant_index_1,
    SchroParams *params)
{
  int index;
  int i,j;
  int q;
  int quant_factor;
  int quant_offset;
  int16_t *data;
  int16_t *line;
  int16_t *prev_line;
  int stride, width, height;
  int side;

  for(index=0;index<13;index++){
    for (side=0;side<2;side++) {
      int xmin, xmax;
      int position;

      position = schro_subband_get_position (index);
      schro_subband_get (frame, 0, position, params,
          &data, &stride, &width, &height);

      if (side) {
        if (quant_index_1[index] == 0) continue;
        xmin = width/2;
        xmax = width;
        quant_factor = schro_table_quant[quant_index_1[index]];
        quant_offset = schro_table_offset_1_2[quant_index_1[index]];
      } else {
        if (quant_index_0[index] == 0) continue;
        xmin = 0;
        xmax = width/2;
        quant_factor = schro_table_quant[quant_index_0[index]];
        quant_offset = schro_table_offset_1_2[quant_index_0[index]];
      }
      GST_DEBUG("side=%d index=%d quant_factor=%d %dx%d",
          side, index, quant_factor, width, height);

      if (index == 0) {
        int pred_value;

        for(j=0;j<height;j++){
          prev_line = OFFSET(data, (j-1)*stride);
          line = OFFSET(data, j*stride);
          for(i=xmin;i<xmax;i++){
            if (j>0) {
              if (i>0) {
                pred_value = schro_divide(line[i - 1] +
                    prev_line[i] + prev_line[i - 1] + 1,3);
              } else {
                pred_value = prev_line[i];
              }
            } else {
              if (i>0) {
                pred_value = line[i - 1];
              } else {
                pred_value = 0;
              }
            }
            q = quantize (line[i] - pred_value, quant_factor, quant_offset);
            line[i] = dequantize (q, quant_factor, quant_offset) + pred_value;
          }
        }
      } else {
        for(j=0;j<height;j++){
          line = OFFSET(data, j*stride);
          for(i=xmin;i<xmax;i++){
            q = quantize (line[i], quant_factor, quant_offset);
            line[i] = dequantize (q, quant_factor, quant_offset);
          }
        }
      }
    }
  }
}

