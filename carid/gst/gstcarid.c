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
#include <carid/caridwavelet.h>

#define GST_TYPE_COMPRESS \
  (gst_compress_get_type())
#define GST_COMPRESS(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_COMPRESS,GstCompress))
#define GST_COMPRESS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_COMPRESS,GstCompressClass))
#define GST_IS_COMPRESS(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_COMPRESS))
#define GST_IS_COMPRESS_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_COMPRESS))

typedef struct _GstCompress GstCompress;
typedef struct _GstCompressClass GstCompressClass;

struct _GstCompress
{
  GstBaseTransform base_transform;

  int wavelet_type;
  int level;
};

struct _GstCompressClass
{
  GstBaseTransformClass parent_class;
};


/* GstCompress signals and args */
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

static void gst_compress_base_init (gpointer g_class);
static void gst_compress_class_init (gpointer g_class,
    gpointer class_data);
static void gst_compress_init (GTypeInstance * instance, gpointer g_class);

static void gst_compress_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_compress_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_compress_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf);

static GstStaticPadTemplate gst_compress_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_compress_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

GType
gst_compress_get_type (void)
{
  static GType compress_type = 0;

  if (!compress_type) {
    static const GTypeInfo compress_info = {
      sizeof (GstCompressClass),
      gst_compress_base_init,
      NULL,
      gst_compress_class_init,
      NULL,
      NULL,
      sizeof (GstCompress),
      0,
      gst_compress_init,
    };

    compress_type = g_type_register_static (GST_TYPE_BASE_TRANSFORM,
        "GstCompress", &compress_info, 0);
  }
  return compress_type;
}


static void
gst_compress_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Video Filter Template",
      "Filter/Effect/Video",
      "Template for a video filter",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);
  //GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_compress_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_compress_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_compress_class_init (gpointer g_class, gpointer class_data)
{
  GObjectClass *gobject_class;
  GstBaseTransformClass *base_transform_class;

  gobject_class = G_OBJECT_CLASS (g_class);
  base_transform_class = GST_BASE_TRANSFORM_CLASS (g_class);

  gobject_class->set_property = gst_compress_set_property;
  gobject_class->get_property = gst_compress_get_property;

  g_object_class_install_property (gobject_class, ARG_WAVELET_TYPE,
      g_param_spec_int ("wavelet-type", "wavelet type", "wavelet type",
        0, 4, 0, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, ARG_LEVEL,
      g_param_spec_int ("level", "level", "level",
        0, 100, 0, G_PARAM_READWRITE));

  base_transform_class->transform_ip = gst_compress_transform_ip;
}

static void
gst_compress_init (GTypeInstance * instance, gpointer g_class)
{
  //GstCompress *compress = GST_COMPRESS (instance);

  GST_DEBUG ("gst_compress_init");

  /* do stuff */
}

static void
gst_compress_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCompress *src;

  g_return_if_fail (GST_IS_COMPRESS (object));
  src = GST_COMPRESS (object);

  GST_DEBUG ("gst_compress_set_property");
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
gst_compress_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstCompress *src;

  g_return_if_fail (GST_IS_COMPRESS (object));
  src = GST_COMPRESS (object);

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
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "compress", GST_RANK_NONE,
      GST_TYPE_COMPRESS);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "compress",
    "Template for a video filter",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)

void decrease_entropy (int16_t *data, int n, int level);
int estimate_entropy (int16_t *src, int n);


static GstFlowReturn
gst_compress_transform_ip (GstBaseTransform * base_transform,
    GstBuffer *buf)
{
  GstCompress *compress;
  int width;
  int height;
  int i,j;
  guint8 *data;
  int n;
  gint16 *tmp;

  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "width", &width);
  gst_structure_get_int (gst_caps_get_structure(buf->caps,0),
      "height", &height);

  g_return_val_if_fail (GST_IS_COMPRESS (base_transform), GST_FLOW_ERROR);
  compress = GST_COMPRESS (base_transform);

  data = GST_BUFFER_DATA (buf);

  for(n=1;n<width && n<height && n<=256;n<<=1);
  n>>=1;

  data += (width-n)/2 + width * (height-n)/2;

  tmp = g_malloc(n*n*2);
  for(j=0;j<n;j++) {
    for(i=0;i<n;i++) {
      tmp[i+j*n] = data[i+j*width];
    }
  }
  carid_iwt_2d (compress->wavelet_type, tmp, n, n);
  //decrease_entropy (tmp, n*n, compress->level);
  g_print("%d\n", estimate_entropy (tmp, n*n));
  //carid_iiwt_2d (compress->wavelet_type, tmp, n, n);
  for(j=0;j<n;j++) {
    for(i=0;i<n;i++) {
      int x = tmp[i+j*n];
      if (x<0) x = 0;
      if (x>255) x = 255;
      data[i+j*width] = x;
    }
  }
  g_free(tmp);

  return GST_FLOW_OK;
}

#if 0
struct component {
  int i;
  gint16 val;
};

int
entropy_compare (const void *a, const void *b)
{
  const struct component *aa = a;
  const struct component *bb = b;

  if (abs(aa->val) > abs(bb->val)) return -1;
  if (abs(aa->val) < abs(bb->val)) return 1;
  return 0;
}

void
decrease_entropy (int16_t *data, int n)
{
  struct component *index;
  int i;

  index = g_malloc(n*sizeof(struct component));
  for(i=0;i<n;i++){
    index[i].i=i;
    index[i].val=data[i];
    if ((i&0xff) >= 4 || (i>>8)>=4) data[i] = 0;
  }

  qsort (index, n, sizeof(struct component), entropy_compare);

  for(i=0;i<1000;i++){
    data[index[i].i] = index[i].val;
  }

  g_free(index);
}
#endif

void
decrease_entropy (int16_t *data, int n, int level)
{
  int i;

  if (level==0) level = 1;
#if 0
  for (i=0;i<n;i++) {
    if (abs(data[i]) < 20) data[i] = 0;
  }
#endif
  for (i=0;i<n;i++) {
    if ((i&0xff) >= 16 || (i>>8)>=16) {
      data[i] = (data[i]+level/2)/level * level;
    }
  }
}

int
estimate_entropy (int16_t *src, int n)
{
  int i;
  int sum = 0;
  int min = src[0];
  int max = src[0];
  unsigned int x;
  for(i=0;i<n;i++){
    x = abs(src[i]);
    if (x&0xffff0000) sum+=16;
    if (x&0xff00ff00) sum+=8;
    if (x&0xf0f0f0f0) sum+=4;
    if (x&0xcccccccc) sum+=2;
    if (x&0xaaaaaaaa) sum+=1;
    if (src[i] < min) min = src[i];
    if (src[i] > max) max = src[i];
  }
  g_print("range = [%d,%d]\n", min, max);
  return sum;
}

