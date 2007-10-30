/* 
 * GStreamer
 * Copyright (C) 2007 David Schleef <ds@schleef.org>
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


#define GST_CAT_DEFAULT gst_frame_store_debug
GST_DEBUG_CATEGORY_STATIC (GST_CAT_DEFAULT);

#define GST_TYPE_FRAME_STORE            (gst_frame_store_get_type())
#define GST_FRAME_STORE(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_FRAME_STORE,GstFrameStore))
#define GST_IS_FRAME_STORE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_FRAME_STORE))
#define GST_FRAME_STORE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass) ,GST_TYPE_FRAME_STORE,GstFrameStoreClass))
#define GST_IS_FRAME_STORE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass) ,GST_TYPE_FRAME_STORE))
#define GST_FRAME_STORE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj) ,GST_TYPE_FRAME_STORE,GstFrameStoreClass))
typedef struct _GstFrameStore GstFrameStore;
typedef struct _GstFrameStoreClass GstFrameStoreClass;

typedef void (*GstFrameStoreProcessFunc) (GstFrameStore *, guint8 *, guint);

struct _GstFrameStore
{
  GstElement element;

  /* < private > */
  GstPad *srcpad;
  GstPad *sinkpad;

  gboolean stepping;
  gboolean step;
  gboolean new_buffer;

  int n_frames;
  int max_frames;
  GstBuffer **frames;

  gboolean flushing;
  GMutex *mutex;
  GCond *cond;
};

struct _GstFrameStoreClass
{
  GstElementClass parent;
};

static const GstElementDetails element_details =
GST_ELEMENT_DETAILS ("FIXME",
    "Filter/Effect",
    "FIXME example filter",
    "FIXME <fixme@fixme.com>");

enum
{
  PROP_0,
  PROP_STEPPING,
  PROP_STEP
};

#define DEBUG_INIT(bla) \
  GST_DEBUG_CATEGORY_INIT (gst_frame_store_debug, "framestore", 0, "framestore element");

GST_BOILERPLATE_FULL (GstFrameStore, gst_frame_store, GstElement,
    GST_TYPE_ELEMENT, DEBUG_INIT);

static void gst_frame_store_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_frame_store_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_frame_store_chain (GstPad *pad, GstBuffer *buffer);
static gboolean gst_frame_store_sink_event (GstPad *pad, GstEvent *event);
#if 0
static gboolean gst_frame_store_set_caps (GstBaseTransform * filter,
    GstCaps *incaps, GstCaps *outcaps);
static GstFlowReturn gst_frame_store_transform (GstBaseTransform * base,
    GstBuffer * inbuf, GstBuffer *outbuf);
#endif
static void gst_frame_store_reset (GstFrameStore *filter);
static void gst_frame_store_clear (GstFrameStore *filter);
#if 0
static void gst_frame_store_task (void *element);
#endif
#if 0
static GstCaps * gst_frame_store_transform_caps (GstBaseTransform *base,
    GstPadDirection direction, GstCaps *caps);
static gboolean gst_frame_store_get_unit_size (GstBaseTransform *base,
    GstCaps *caps, guint *size);
#endif

static void gst_frame_store_step (GstFrameStore *fs);

static GstStaticPadTemplate gst_framestore_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static GstStaticPadTemplate gst_framestore_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("I420"))
    );

static void
gst_frame_store_base_init (gpointer klass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_framestore_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_framestore_sink_template));

  gst_element_class_set_details (element_class, &element_details);
}

static void
gst_frame_store_class_init (GstFrameStoreClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;
  gobject_class->set_property = gst_frame_store_set_property;
  gobject_class->get_property = gst_frame_store_get_property;

  g_object_class_install_property (gobject_class, PROP_STEPPING,
      g_param_spec_boolean ("stepping", "stepping", "stepping",
        FALSE, G_PARAM_READWRITE));
  g_object_class_install_property (gobject_class, PROP_STEP,
      g_param_spec_boolean ("step", "step", "step",
        FALSE, G_PARAM_READWRITE));
}

static void
gst_frame_store_init (GstFrameStore * filter, GstFrameStoreClass * klass)
{
  gst_element_create_all_pads (GST_ELEMENT(filter));
  filter->srcpad = gst_element_get_pad (GST_ELEMENT(filter), "src");
  filter->sinkpad = gst_element_get_pad (GST_ELEMENT(filter), "sink");

  gst_pad_set_chain_function (filter->sinkpad, gst_frame_store_chain);
  gst_pad_set_event_function (filter->sinkpad, gst_frame_store_sink_event);

  gst_frame_store_reset (filter);

  filter->n_frames = 0;
  filter->max_frames = 10;
  filter->frames = malloc(sizeof(GstBuffer*)*filter->max_frames);

  filter->cond = g_cond_new ();
  filter->mutex = g_mutex_new ();
}

static void
gst_frame_store_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstFrameStore *filter = GST_FRAME_STORE (object);

  switch (prop_id) {
    case PROP_STEPPING:
      filter->stepping = g_value_get_boolean (value);
      g_cond_signal (filter->cond);
      break;
    case PROP_STEP:
      gst_frame_store_step (filter);
#if 0
      filter->step = g_value_get_boolean (value);
      g_cond_signal (filter->cond);
#endif
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_frame_store_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstFrameStore *filter = GST_FRAME_STORE (object);

  switch (prop_id) {
    case PROP_STEPPING:
      g_value_set_boolean (value, filter->stepping);
      break;
    case PROP_STEP:
      g_value_set_boolean (value, filter->step);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

#if 0
static gboolean
gst_frame_store_set_caps (GstBaseTransform * base, GstCaps *incaps,
    GstCaps *outcaps)
{
  //GstFrameStore *filter = GST_FRAME_STORE (base);
  GstStructure *structure;

  structure = gst_caps_get_structure (incaps, 0);

  return TRUE;
}
#endif

static void
gst_frame_store_reset (GstFrameStore *filter)
{
  gst_frame_store_clear (filter);
}

static void
gst_frame_store_clear (GstFrameStore *filter)
{
  int i;
  for(i=0;i<filter->n_frames;i++){
    gst_buffer_unref (filter->frames[i]);
  }
  filter->n_frames = 0;
}

#if 0
static GstCaps *
gst_frame_store_transform_caps (GstBaseTransform *base,
    GstPadDirection direction, GstCaps *caps)
{
  return caps;
}
#endif

#if 0
static gboolean
gst_frame_store_get_unit_size (GstBaseTransform *base, GstCaps *caps,
    guint *size)
{
  int width, height;
  GstStructure *structure;

  structure = gst_caps_get_structure(caps, 0);
  gst_structure_get_int (structure, "width", &width);
  gst_structure_get_int (structure, "height", &height);

  /* FIXME bogus */
  *size = (width*height*3)/2;

  return TRUE;
}
#endif

static void
gst_frame_store_append (GstFrameStore *fs, GstBuffer *buffer)
{
  gst_buffer_ref(buffer);
  if (fs->n_frames < fs->max_frames) {
    fs->frames[fs->n_frames] = buffer;
    fs->n_frames++;
  } else {
    gst_buffer_unref(fs->frames[0]);
    memmove (fs->frames, fs->frames+1, (fs->n_frames-1)*sizeof(GstBuffer*));
    fs->frames[fs->n_frames-1] = buffer;
  }
}

static GstFlowReturn
gst_frame_store_chain (GstPad *pad, GstBuffer *buffer)
{
  GstFrameStore *fs;
  
  //fs = GST_FRAME_STORE(gst_pad_get_parent(pad));
  fs = (GstFrameStore *)(gst_pad_get_parent(pad));

  g_mutex_lock (fs->mutex);
  gst_frame_store_append (fs, buffer);
  fs->new_buffer = TRUE;
  g_cond_signal (fs->cond);
  g_mutex_unlock (fs->mutex);

  g_mutex_lock (fs->mutex);
  while(fs->new_buffer == TRUE) {
    g_cond_wait (fs->cond, fs->mutex);
  }
  g_mutex_unlock (fs->mutex);

  return GST_FLOW_OK;
}

#if 0
static void
gst_frame_store_task (void *element)
{
  GstFrameStore *fs;

  fs = GST_FRAME_STORE(element);

  g_mutex_lock (fs->mutex);
  while(fs->new_buffer == FALSE) {
    g_cond_wait (fs->cond, fs->mutex);
  }
  g_mutex_unlock (fs->mutex);

  gst_pad_push (fs->srcpad, gst_buffer_ref(buffer));
}
#endif

static gboolean
gst_frame_store_sink_event (GstPad *pad, GstEvent *event)
{
  GstFrameStore *fs;
  
  //fs = GST_FRAME_STORE(gst_pad_get_parent(pad));
  fs = (GstFrameStore *)(gst_pad_get_parent(pad));

  switch(GST_EVENT_TYPE(event)) {
    case GST_EVENT_NEWSEGMENT:
      {
        gboolean update;
        double rate;
        double applied_rate;
        GstFormat format;
        gint64 start, stop, position;

        gst_event_parse_new_segment_full (event, &update, &rate, &applied_rate,
            &format, &start, &stop, &position);

        GST_ERROR("new_segment %d %g %g %d %lld %lld %lld",
            update, rate, applied_rate, format, start, stop, position);

        gst_frame_store_clear (fs);
      }
      break;
    case GST_EVENT_FLUSH_START:
      fs->flushing = TRUE;
      GST_ERROR("flush start");
      break;
    case GST_EVENT_FLUSH_STOP:
      fs->flushing = FALSE;
      GST_ERROR("flush stop");
      break;
    default:
      break;
  }

  gst_pad_push_event (fs->srcpad, event);

  return TRUE;
}

#if 0
static GstFlowReturn
gst_frame_store_transform (GstBaseTransform * base,
    GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstFrameStore *fs = GST_FRAME_STORE (base);

  g_return_val_if_fail (gst_buffer_is_writable (outbuf), GST_FLOW_ERROR);

  memcpy (GST_BUFFER_DATA(outbuf), GST_BUFFER_DATA(inbuf),
      GST_BUFFER_SIZE(inbuf));

  if(fs->stored_buffer) {
    gst_buffer_unref (fs->stored_buffer);
  }
  fs->stored_buffer = gst_buffer_ref (inbuf);

  return GST_FLOW_OK;
}
#endif

static void gst_frame_store_step (GstFrameStore *fs)
{
  GstBuffer *buffer;
  GstEvent *event;

  GST_ERROR("ping");

  g_mutex_lock (fs->mutex);
  buffer = gst_buffer_ref(fs->frames[fs->n_frames-2]);
  g_mutex_unlock (fs->mutex);

  event = gst_event_new_flush_start ();
  gst_pad_push_event (fs->srcpad, event);
  GST_ERROR("got here");

  event = gst_event_new_flush_stop ();
  gst_pad_push_event (fs->srcpad, event);
  GST_ERROR("got here");

  event = gst_event_new_new_segment (FALSE, 1.0, GST_FORMAT_TIME,
      GST_BUFFER_TIMESTAMP(buffer), GST_CLOCK_TIME_NONE, 0ULL);
  gst_pad_push_event (fs->srcpad, event);
  GST_ERROR("got here");

  GST_BUFFER_TIMESTAMP(buffer) = GST_CLOCK_TIME_NONE;
  gst_pad_push (fs->srcpad, buffer);
  GST_ERROR("got here");
}



