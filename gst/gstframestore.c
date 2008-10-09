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

  GstBuffer **frames;
  int range_offset;
  int range_size;

  int input_frame_number;
  int frame_number;
  int pushed_frame_number;
  gboolean need_newsegment;

  gboolean flushing;
  GMutex *lock;
  GCond *cond;
  GstFlowReturn srcresult;
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
  PROP_STEP,
  PROP_FRAME_NUMBER
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
static void gst_frame_store_reset (GstFrameStore *filter);
static void gst_frame_store_clear (GstFrameStore *filter);
static GstPadLinkReturn gst_frame_store_link_src (GstPad *pad, GstPad *peer);
static void gst_frame_store_task (GstPad *pad);
static GstCaps * gst_frame_store_getcaps (GstPad *pad);
static void gst_frame_store_finalize (GObject *object);

static GstBuffer * gst_frame_store_get_frame (GstFrameStore *fs, int frame_number);
static void gst_frame_store_advance (GstFrameStore *fs);

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
  g_object_class_install_property (gobject_class, PROP_FRAME_NUMBER,
      g_param_spec_int ("frame-number", "frame number", "frame number",
        0, G_MAXINT, 0, G_PARAM_READWRITE));

  gobject_class->finalize = gst_frame_store_finalize;
}

static void
gst_frame_store_init (GstFrameStore * filter, GstFrameStoreClass * klass)
{
  gst_element_create_all_pads (GST_ELEMENT(filter));

  filter->srcpad = gst_element_get_pad (GST_ELEMENT(filter), "src");

  gst_pad_set_link_function (filter->srcpad, gst_frame_store_link_src);
  gst_pad_set_getcaps_function (filter->srcpad, gst_frame_store_getcaps);

  filter->sinkpad = gst_element_get_pad (GST_ELEMENT(filter), "sink");

  gst_pad_set_chain_function (filter->sinkpad, gst_frame_store_chain);
  gst_pad_set_event_function (filter->sinkpad, gst_frame_store_sink_event);
  gst_pad_set_getcaps_function (filter->sinkpad, gst_frame_store_getcaps);

  gst_frame_store_reset (filter);

  filter->range_offset = 0;
  filter->range_size = 10;
  filter->frames = g_malloc0(sizeof(GstBuffer*)*filter->range_size);
  filter->frame_number = 0;
  filter->pushed_frame_number = -1;
  filter->need_newsegment = TRUE;

  filter->cond = g_cond_new ();
  filter->lock = g_mutex_new ();
  filter->srcresult = GST_FLOW_WRONG_STATE;
}

static void
gst_frame_store_finalize (GObject *object)
{
  GstFrameStore *fs = GST_FRAME_STORE (object);

  g_mutex_free (fs->lock);
  g_cond_free (fs->cond);
  g_free (fs->frames);

}

static GstCaps *
gst_frame_store_getcaps (GstPad *pad)
{
  GstFrameStore *fs;
  GstPad *otherpad;
  GstCaps *caps;
  const GstCaps *tcaps;
  
  fs = GST_FRAME_STORE (gst_pad_get_parent (pad));

  otherpad = (pad == fs->srcpad ? fs->sinkpad : fs->srcpad);
  caps = gst_pad_peer_get_caps (otherpad);
  tcaps = gst_pad_get_pad_template_caps (pad);
  if (caps) {
    GstCaps *icaps;
    icaps = gst_caps_intersect (caps, tcaps);
    gst_caps_unref (caps);
    caps = icaps;
  } else {
    caps = gst_caps_copy(tcaps);
  }

  gst_object_unref (fs);

  return caps;
}


static void
gst_frame_store_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstFrameStore *fs = GST_FRAME_STORE (object);

  switch (prop_id) {
    case PROP_STEPPING:
      g_mutex_lock (fs->lock);
      fs->stepping = g_value_get_boolean (value);
      GST_DEBUG("stepping %d", fs->stepping);
      if (fs->stepping == FALSE) {
        fs->need_newsegment = TRUE;
      }
      g_cond_broadcast (fs->cond);
      g_mutex_unlock (fs->lock);
      break;
    case PROP_STEP:
      g_mutex_lock (fs->lock);
      fs->frame_number++;
      g_cond_broadcast (fs->cond);
      g_mutex_unlock (fs->lock);
      break;
    case PROP_FRAME_NUMBER:
      g_mutex_lock (fs->lock);
      fs->frame_number = g_value_get_int (value);
      g_cond_broadcast (fs->cond);
      g_mutex_unlock (fs->lock);
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
  GstFrameStore *fs = GST_FRAME_STORE (object);

  switch (prop_id) {
    case PROP_STEPPING:
      g_value_set_boolean (value, fs->stepping);
      break;
    case PROP_STEP:
      g_value_set_boolean (value, FALSE);
      break;
    case PROP_FRAME_NUMBER:
      g_value_set_int (value, fs->frame_number);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_frame_store_reset (GstFrameStore *fs)
{
  gst_frame_store_clear (fs);
}

static GstPadLinkReturn
gst_frame_store_link_src (GstPad *pad, GstPad *peer)
{
  GstPadLinkReturn result = GST_PAD_LINK_OK;
  GstFrameStore *fs;
  
  fs = GST_FRAME_STORE(gst_pad_get_parent(pad));

  if (GST_PAD_LINKFUNC (peer)) {
    result = GST_PAD_LINKFUNC(peer) (peer, pad);
  }

  if (GST_PAD_LINK_SUCCESSFUL (result)) {
    g_mutex_lock (fs->lock);
    
fs->srcresult = GST_FLOW_OK;
    if (fs->srcresult == GST_FLOW_OK) {
      gst_pad_start_task (pad, (GstTaskFunction) gst_frame_store_task,
          pad);
    } else {
      GST_DEBUG("not starting task");
      /* not starting task */
    }
    g_mutex_unlock (fs->lock);
  }

  gst_object_unref (fs);

  return result;
}


static GstFlowReturn
gst_frame_store_chain (GstPad *pad, GstBuffer *buffer)
{
  GstFrameStore *fs;
  
  fs = GST_FRAME_STORE(gst_pad_get_parent(pad));

  GST_DEBUG("chain");

  g_mutex_lock (fs->lock);
  while (fs->range_offset + fs->range_size == fs->input_frame_number) {
    GST_DEBUG("waiting for empty slot");
    g_cond_wait (fs->cond, fs->lock);
  }
  GST_DEBUG("adding %d", fs->input_frame_number);
  fs->frames[fs->input_frame_number - fs->range_offset] = buffer;
  fs->input_frame_number++;
  g_cond_broadcast (fs->cond);
  g_mutex_unlock (fs->lock);

  GST_DEBUG("chain done");

  gst_object_unref (fs);

  return GST_FLOW_OK;
}

static void
gst_frame_store_task (GstPad *pad)
{
  GstFrameStore *fs;
  GstBuffer *buffer = NULL;
  GstEvent *event = NULL;

  fs = GST_FRAME_STORE (gst_pad_get_parent (pad));

  GST_DEBUG("task");

  g_mutex_lock (fs->lock);
  while(1) {
    if (fs->stepping == FALSE || (fs->frame_number != fs->pushed_frame_number)) {
      buffer = gst_frame_store_get_frame (fs, fs->frame_number);
    }
    if (buffer) break;
    g_cond_wait (fs->cond, fs->lock);
  }
  if (fs->need_newsegment) {
    GstClock *clock;
    GstClockTime now;
    GstClockTime stream_time;

    clock = GST_ELEMENT_CLOCK (fs);
    if (clock == NULL) {
      now = 0;
      stream_time = 0;
    } else {
      now = gst_clock_get_time (GST_ELEMENT_CLOCK (fs));
      stream_time = now - GST_ELEMENT(fs)->base_time;
    }
    GST_ERROR("now %lld buffer %lld stream_time %lld",
        now, GST_BUFFER_TIMESTAMP(buffer), stream_time);
    stream_time = GST_SECOND*10;
    event = gst_event_new_new_segment (FALSE, 1.0, GST_FORMAT_TIME,
        GST_BUFFER_TIMESTAMP(buffer), -1, stream_time);
    fs->need_newsegment = FALSE;
  }
  if (fs->stepping) {
    buffer = gst_buffer_make_metadata_writable (buffer);
    GST_BUFFER_TIMESTAMP(buffer) = -1;
    GST_BUFFER_DURATION(buffer) = -1;
  }
  fs->pushed_frame_number = fs->frame_number;
  if (!fs->stepping) {
    fs->frame_number++;
  }
  if (fs->frame_number + 1 >= fs->range_offset + fs->range_size) {
    gst_frame_store_advance (fs);
  }
  g_mutex_unlock (fs->lock);

  if (event) {
    gst_pad_push_event (fs->srcpad, event);
  }

  gst_pad_push (fs->srcpad, buffer);

  GST_DEBUG("task done");

  gst_object_unref (fs);
}

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
      GST_DEBUG("flush start");
      break;
    case GST_EVENT_FLUSH_STOP:
      fs->flushing = FALSE;
      GST_DEBUG("flush stop");
      break;
    default:
      break;
  }

  gst_pad_push_event (fs->srcpad, event);

  return TRUE;
}


static GstBuffer *
gst_frame_store_get_frame (GstFrameStore *fs, int frame_number)
{
  GstBuffer *buffer;

  if (frame_number < fs->range_offset ||
      frame_number >= fs->range_offset + fs->range_size) {
    return NULL;
  }

  buffer = fs->frames[frame_number - fs->range_offset];
  if (buffer) gst_buffer_ref (buffer);

  return buffer;
}

static void
gst_frame_store_clear (GstFrameStore *fs)
{
  int i;
  for(i=0;i<fs->range_size;i++){
    if (fs->frames[i]) {
      gst_buffer_unref (fs->frames[i]);
    }
    fs->frames[i] = NULL;
  }
}

static void
gst_frame_store_advance (GstFrameStore *fs)
{
  if (fs->frames[0]) {
    gst_buffer_unref (fs->frames[0]);
  }
  memmove (fs->frames, fs->frames+1, sizeof(GstBuffer *)*(fs->range_size - 1));
  fs->frames[fs->range_size - 1] = NULL;
  fs->range_offset++;
  g_cond_broadcast (fs->cond);
}
