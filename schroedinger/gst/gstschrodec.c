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
#include <math.h>

#include "parsehelper.h"

GST_DEBUG_CATEGORY_EXTERN (schro_debug);
#define GST_CAT_DEFAULT schro_debug

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

  /* state */
  int n_frames;
  gint64 granulepos;
  GstSegment segment;
  gboolean discont;

  int bytes_per_picture;
  int fps_n;
  int fps_d;

  gboolean have_access_unit;

  int width;
  int height;

  ParseHelper parse_helper;

  gdouble proportion;
  GstClockTime earliest_time;
  GstClockTime timestamp_offset;
  gint64 granulepos_offset;
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

static const GstQueryType *gst_schro_dec_get_query_types (GstPad *pad);
static gboolean gst_schro_dec_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value);
static gboolean gst_schro_dec_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value);
static gboolean gst_schro_dec_src_query (GstPad *pad, GstQuery *query);
static gboolean gst_schro_dec_sink_query (GstPad *pad, GstQuery *query);
static gboolean gst_schro_dec_src_event (GstPad *pad, GstEvent *event);
static gboolean gst_schro_dec_sink_event (GstPad *pad, GstEvent *event);
static GstStateChangeReturn gst_schro_dec_change_state (GstElement *element,
    GstStateChange transition);
static GstFlowReturn gst_schro_dec_chain (GstPad *pad, GstBuffer *buf);
static GstFlowReturn gst_schro_dec_push_all (GstSchroDec *schro_dec, 
    gboolean at_eos);

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

  element_class->change_state = gst_schro_dec_change_state;
}

static void
gst_schro_dec_init (GstSchroDec *schro_dec, GstSchroDecClass *klass)
{
  GST_DEBUG ("gst_schro_dec_init");

  schro_dec->decoder = schro_decoder_new ();

  schro_dec->sinkpad = gst_pad_new_from_static_template (&gst_schro_dec_sink_template, "sink");
  gst_pad_set_chain_function (schro_dec->sinkpad, gst_schro_dec_chain);
  gst_pad_set_query_function (schro_dec->sinkpad, gst_schro_dec_sink_query);
  gst_pad_set_event_function (schro_dec->sinkpad, gst_schro_dec_sink_event);
  gst_element_add_pad (GST_ELEMENT(schro_dec), schro_dec->sinkpad);

  schro_dec->srcpad = gst_pad_new_from_static_template (&gst_schro_dec_src_template, "src");
  gst_pad_set_query_type_function (schro_dec->srcpad, gst_schro_dec_get_query_types);
  gst_pad_set_query_function (schro_dec->srcpad, gst_schro_dec_src_query);
  gst_pad_set_event_function (schro_dec->srcpad, gst_schro_dec_src_event);
  gst_element_add_pad (GST_ELEMENT(schro_dec), schro_dec->srcpad);

  parse_helper_init (&schro_dec->parse_helper);
}

static void
gst_schro_dec_reset (GstSchroDec *dec)
{
  GST_DEBUG("reset");
  dec->granulepos = 0;
  dec->discont = TRUE;
  dec->n_frames = 0;
  dec->timestamp_offset = GST_CLOCK_TIME_NONE;
  dec->granulepos_offset = -1;

  gst_segment_init (&dec->segment, GST_FORMAT_TIME);
  parse_helper_flush (&dec->parse_helper);
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

  parse_helper_free (&schro_dec->parse_helper);

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

#define OGG_DIRAC_GRANULE_SHIFT 30
#define OGG_DIRAC_GRANULE_LOW_MASK ((1<<OGG_DIRAC_GRANULE_SHIFT)-1)

static gint64
granulepos_to_frame (gint64 granulepos)
{
  if (granulepos == -1) return -1;

  return (granulepos >> OGG_DIRAC_GRANULE_SHIFT) +
    (granulepos & OGG_DIRAC_GRANULE_LOW_MASK);
}

static const GstQueryType *
gst_schro_dec_get_query_types (GstPad *pad)
{
  static const GstQueryType query_types[] = {
    GST_QUERY_POSITION,
    GST_QUERY_DURATION,
    GST_QUERY_CONVERT,
    0
  };

  return query_types;
}

static gboolean
gst_schro_dec_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstSchroDec *dec;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  dec = GST_SCHRO_DEC (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a decoding state */

  switch (src_format) {
    case GST_FORMAT_BYTES:
      switch (*dest_format) {
        case GST_FORMAT_DEFAULT:
          *dest_value = gst_util_uint64_scale_int (src_value, 1,
              dec->bytes_per_picture);
          break;
        case GST_FORMAT_TIME:
          /* seems like a rather silly conversion, implement me if you like */
        default:
          res = FALSE;
      }
      break;
    case GST_FORMAT_DEFAULT:
      switch (*dest_format) {
        case GST_FORMAT_TIME:
          *dest_value = gst_util_uint64_scale (src_value,
              GST_SECOND * dec->fps_d, dec->fps_n);
          break;
        case GST_FORMAT_BYTES:
          *dest_value = gst_util_uint64_scale_int (src_value,
              dec->bytes_per_picture, 1);
          break;
        default:
          res = FALSE;
      }
      break;
    default:
      res = FALSE;
      break;
  }

  gst_object_unref (dec);

  return res;
}

static gboolean
gst_schro_dec_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstSchroDec *dec;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  dec = GST_SCHRO_DEC (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a decoding state */

  switch (src_format) {
    case GST_FORMAT_DEFAULT:
      switch (*dest_format) {
        case GST_FORMAT_TIME:
          *dest_value = gst_util_uint64_scale (granulepos_to_frame (src_value),
              dec->fps_d * GST_SECOND, dec->fps_n);
          break;
        default:
          res = FALSE;
      }
      break;
    case GST_FORMAT_TIME:
      switch (*dest_format) {
        case GST_FORMAT_DEFAULT:
        {
          *dest_value = gst_util_uint64_scale (src_value,
              dec->fps_n, dec->fps_d * GST_SECOND);
          break;
        }
        default:
          res = FALSE;
          break;
      }
      break;
    default:
      res = FALSE;
      break;
  }

  gst_object_unref (dec);

  return res;
}

static gboolean
gst_schro_dec_src_query (GstPad *pad, GstQuery *query)
{
  GstSchroDec *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_DEC (gst_pad_get_parent(pad));

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_POSITION:
    {
      GstFormat format;
      gint64 time;
      gint64 value;

      gst_query_parse_position (query, &format, NULL);

      time = gst_util_uint64_scale (granulepos_to_frame (dec->granulepos),
              dec->fps_n, dec->fps_d);
      //time -= dec->segment.start;
      time += dec->segment.time;
      GST_DEBUG("query position %lld", time);
      res = gst_schro_dec_src_convert (pad, GST_FORMAT_TIME, time,
          &format, &value);
      if (!res) goto error;

      gst_query_set_position (query, format, value);
      break;
    }
    case GST_QUERY_DURATION:
      res = gst_pad_query (GST_PAD_PEER (dec->sinkpad), query);
      if (!res) goto error;
      break;
    case GST_QUERY_CONVERT:
    {
      GstFormat src_fmt, dest_fmt;
      gint64 src_val, dest_val;

      gst_query_parse_convert (query, &src_fmt, &src_val, &dest_fmt, &dest_val);
      res = gst_schro_dec_src_convert (pad, src_fmt, src_val, &dest_fmt,
          &dest_val);
      if (!res) goto error;
      gst_query_set_convert (query, src_fmt, src_val, dest_fmt, dest_val);
      break;
    }
    default:
      res = gst_pad_query_default (pad, query);
      break;
  }
done:
  gst_object_unref (dec);

  return res;
error:
  GST_DEBUG_OBJECT (dec, "query failed");
  goto done;
}

static gboolean
gst_schro_dec_sink_query (GstPad *pad, GstQuery *query)
{
  GstSchroDec *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_DEC (gst_pad_get_parent(pad));

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CONVERT:
    {
      GstFormat src_fmt, dest_fmt;
      gint64 src_val, dest_val;

      gst_query_parse_convert (query, &src_fmt, &src_val, &dest_fmt, &dest_val);
      res = gst_schro_dec_sink_convert (pad, src_fmt, src_val, &dest_fmt,
          &dest_val);
      if (!res) goto error;
      gst_query_set_convert (query, src_fmt, src_val, dest_fmt, dest_val);
      break;
    }
    default:
      res = gst_pad_query_default (pad, query);
      break;
  }
done:
  gst_object_unref (dec);

  return res;
error:
  GST_DEBUG_OBJECT (dec, "query failed");
  goto done;
}

static gboolean
gst_schro_dec_src_event (GstPad *pad, GstEvent *event)
{
  GstSchroDec *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_DEC (gst_pad_get_parent(pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
    {
      GstFormat format, tformat;
      gdouble rate;
      GstEvent *real_seek;
      GstSeekFlags flags;
      GstSeekType cur_type, stop_type;
      gint64 cur, stop;
      gint64 tcur, tstop;

      gst_event_parse_seek (event, &rate, &format, &flags, &cur_type,
          &cur, &stop_type, &stop);
      gst_event_unref (event);

      tformat = GST_FORMAT_TIME;
      res = gst_schro_dec_src_convert (pad, format, cur, &tformat, &tcur);
      if (!res) goto convert_error;
      res = gst_schro_dec_src_convert (pad, format, stop, &tformat, &tstop);
      if (!res) goto convert_error;

      real_seek = gst_event_new_seek (rate, GST_FORMAT_TIME,
          flags, cur_type, tcur, stop_type, tstop);

      res = gst_pad_push_event (dec->sinkpad, real_seek);

      break;
    }
    case GST_EVENT_QOS:
    {
      gdouble proportion;
      GstClockTimeDiff diff;
      GstClockTime timestamp;

      gst_event_parse_qos (event, &proportion, &diff, &timestamp);

      GST_OBJECT_LOCK (dec);
      dec->proportion = proportion;
      dec->earliest_time = timestamp + diff;
      GST_OBJECT_UNLOCK (dec);

      GST_DEBUG_OBJECT (dec, "got QoS %" GST_TIME_FORMAT ", %" G_GINT64_FORMAT ", %g",
          GST_TIME_ARGS(timestamp), diff, proportion);

      res = gst_pad_push_event (dec->sinkpad, event);
      break;
    }
    default:
      res = gst_pad_push_event (dec->sinkpad, event);
      break;
  }
done:
  gst_object_unref (dec);
  return res;

convert_error:
  GST_DEBUG_OBJECT (dec, "could not convert format");
  goto done;
}

static gboolean
gst_schro_dec_sink_event (GstPad *pad, GstEvent *event)
{
  GstSchroDec *dec;
  gboolean ret = FALSE;

  dec = GST_SCHRO_DEC (gst_pad_get_parent(pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_START:
      GST_ERROR("unhandled flush start");
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    case GST_EVENT_FLUSH_STOP:
      GST_ERROR("unhandled flush stop");
      gst_schro_dec_reset (dec);
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    case GST_EVENT_EOS:
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    case GST_EVENT_NEWSEGMENT:
    {
      gboolean update;
      GstFormat format;
      gdouble rate;
      gint64 start, stop, time;

      gst_event_parse_new_segment (event, &update, &rate, &format, &start,
          &stop, &time);

      if (format != GST_FORMAT_TIME)
        goto newseg_wrong_format;

      if (rate <= 0.0)
        goto newseg_wrong_rate;

      GST_ERROR("newsegment %lld %lld", start, time);
      gst_segment_set_newsegment (&dec->segment, update, rate, format,
          start, stop, time);

      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    }
    default:
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
  }
done:
  gst_object_unref (dec);
  return ret;

newseg_wrong_format:
  GST_DEBUG_OBJECT (dec, "received non TIME newsegment");
  gst_event_unref (event);
  goto done;

newseg_wrong_rate:
  GST_DEBUG_OBJECT (dec, "negative rates not supported");
  gst_event_unref (event);
  goto done;
}


static GstStateChangeReturn
gst_schro_dec_change_state (GstElement *element, GstStateChange transition)
{
  GstSchroDec *dec = GST_SCHRO_DEC (element);
  GstStateChangeReturn ret;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_schro_dec_reset (dec);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
      break;
    default:
      break;
  }

  ret = parent_class->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_schro_dec_reset (dec);
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    default:
      break;
  }

  return ret;
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

  gst_buffer_unref (GST_BUFFER (priv));
}


static SchroFrame *
gst_schro_wrap_frame (GstSchroDec *schro_dec, GstBuffer *buffer)
{
  SchroFrame *frame;

  frame = schro_frame_new_I420 (GST_BUFFER_DATA (buffer),
      schro_dec->width, schro_dec->height);
  frame->free = gst_schro_frame_free;
  frame->priv = buffer;

  return frame;
}

static GstFlowReturn
gst_schro_dec_chain (GstPad *pad, GstBuffer *buf)
{
  GstSchroDec *schro_dec;

  schro_dec = GST_SCHRO_DEC (GST_PAD_PARENT (pad));

  GST_DEBUG("timestamp offset %lld, buffer %lld gp %lld, size %d",
    schro_dec->timestamp_offset, GST_BUFFER_TIMESTAMP(buf),
    GST_BUFFER_OFFSET_END(buf), GST_BUFFER_SIZE(buf));

  if (G_UNLIKELY (GST_BUFFER_FLAG_IS_SET (buf, GST_BUFFER_FLAG_DISCONT))) {
    GST_DEBUG_OBJECT (schro_dec, "received DISCONT buffer");
    schro_decoder_reset (schro_dec->decoder);
    schro_dec->timestamp_offset = GST_CLOCK_TIME_NONE;
    schro_dec->granulepos_offset = -1;
    schro_dec->granulepos = -1;
    schro_dec->discont = TRUE;
  }

  if (!GST_CLOCK_TIME_IS_VALID(schro_dec->timestamp_offset) &&
      GST_BUFFER_TIMESTAMP_IS_VALID(buf)) {
    schro_dec->timestamp_offset = GST_BUFFER_TIMESTAMP(buf);
    GST_DEBUG("setting timestamp offset to %lld", schro_dec->timestamp_offset);
  }
  if (schro_dec->granulepos_offset == -1 &&
      GST_BUFFER_OFFSET_END(buf) != -1) {
    schro_dec->granulepos_offset = GST_BUFFER_OFFSET_END(buf);
    GST_DEBUG("setting granulepos offset to %lld", GST_BUFFER_OFFSET_END(buf));
  }

  parse_helper_push (&schro_dec->parse_helper, buf);

  return gst_schro_dec_push_all (schro_dec, FALSE);
}

#define ROUND_UP_SHIFT(x,y) (((x) + (1<<(y)) - 1)>>(y))
#define ROUND_UP_POW2(x,y) (((x) + (1<<(y)) - 1)&((~0)<<(y)))

static int
get_i420_size (int width, int height)
{
  int size;

  size = ROUND_UP_POW2(width,2) * ROUND_UP_POW2(height,1);
  size += 2 * ROUND_UP_POW2(ROUND_UP_SHIFT(width,1),2) *
    ROUND_UP_SHIFT(height,1);

  return size;
}

static GstFlowReturn
gst_schro_dec_push_all (GstSchroDec *schro_dec, gboolean at_eos)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *parse_buf;
  SchroBuffer *input_buffer;
  SchroFrame *frame;

  while (TRUE) {
    gint next;

    if (!parse_helper_skip_to_next_parse_unit (&schro_dec->parse_helper,
        NULL, &next)) {
      /* Need more data */
      return GST_FLOW_OK;
    }

    if (next == 0) {
      /* Need to wait for EOS or next parse marker */
      if (at_eos)
        next = parse_helper_avail (&schro_dec->parse_helper);
      else {
        /* Scan for next parse marker */
        if (!parse_helper_have_next_parse_unit (&schro_dec->parse_helper,
           &next))
          return GST_FLOW_OK; /* break for more data */
      }
    }

    if (parse_helper_avail (&schro_dec->parse_helper) < next)
      return GST_FLOW_OK; /* break for more data */

    GST_LOG ("Have complete parse unit of %d bytes", next);

    parse_buf = parse_helper_pull (&schro_dec->parse_helper, next);
    if (parse_buf == NULL)
      return GST_FLOW_ERROR;
    input_buffer = gst_schro_wrap_gst_buffer (parse_buf);
  
    {
      int size;
      int earliest_frame;
      GstBuffer *outbuf;
      int go = 1;
  
      schro_decoder_push (schro_dec->decoder, input_buffer);

      if (schro_dec->have_access_unit) {
        earliest_frame = gst_util_uint64_scale (
            schro_dec->earliest_time - schro_dec->segment.start,
            schro_dec->fps_n, schro_dec->fps_d * GST_SECOND);

        GST_DEBUG("earliest frame %d", earliest_frame);
        schro_decoder_set_earliest_frame (schro_dec->decoder, earliest_frame);

        schro_decoder_set_skip_ratio (schro_dec->decoder,
            1/schro_dec->proportion);
      }

      while (go) {
        int it;
        SchroVideoFormat *format;
        GstCaps *caps;

        it = schro_decoder_iterate (schro_dec->decoder);

        switch (it) {
          case SCHRO_DECODER_FIRST_ACCESS_UNIT:
            format = schro_decoder_get_video_format (schro_dec->decoder);
            schro_dec->width = format->width;
            schro_dec->height = format->height;
        
            caps = gst_caps_new_simple ("video/x-raw-yuv",
                "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('I','4','2','0'),
                "width", G_TYPE_INT, format->width,
                "height", G_TYPE_INT, format->height,
                "framerate", GST_TYPE_FRACTION,
                format->frame_rate_numerator, format->frame_rate_denominator,
                "pixel-aspect-ratio", GST_TYPE_FRACTION,
                format->aspect_ratio_numerator, format->aspect_ratio_denominator,
                NULL);
        
            GST_DEBUG("setting caps %" GST_PTR_FORMAT, caps);
        
            gst_pad_set_caps (schro_dec->srcpad, caps);
        
            schro_dec->fps_n = format->frame_rate_numerator;
            schro_dec->fps_d = format->frame_rate_denominator;
            schro_dec->bytes_per_picture = (format->width * format->height * 3) / 4;

            if (!GST_CLOCK_TIME_IS_VALID(schro_dec->timestamp_offset)) {
              schro_dec->timestamp_offset = gst_util_uint64_scale (
                  granulepos_to_frame (schro_dec->granulepos_offset),
                  schro_dec->fps_d * GST_SECOND, schro_dec->fps_n);
            }
        
            gst_caps_unref (caps);
            free (format);
        
            schro_dec->have_access_unit = TRUE;
            break;
          case SCHRO_DECODER_NEED_BITS:
            go = 0;
            break;
          case SCHRO_DECODER_NEED_FRAME:
            size = get_i420_size (schro_dec->width, schro_dec->height);
#if 0
            ret = gst_pad_alloc_buffer_and_set_caps (schro_dec->srcpad,
                GST_BUFFER_OFFSET_NONE, size,
                GST_PAD_CAPS (schro_dec->srcpad), &outbuf);
#else
            outbuf = gst_buffer_new_and_alloc (size);
            gst_buffer_set_caps(outbuf, GST_PAD_CAPS(schro_dec->srcpad));
#endif
            if (ret != GST_FLOW_OK) {
              GST_ERROR("could not allocate buffer for pad");
              return ret;
            }
            frame = gst_schro_wrap_frame (schro_dec, outbuf);
            schro_decoder_add_output_frame (schro_dec->decoder, frame);
            break;
          case SCHRO_DECODER_OK:
            frame = schro_decoder_pull (schro_dec->decoder);

            if (frame) {
              if (frame->priv) {
                outbuf = frame->priv;
          
                if (schro_dec->discont) {
                  GST_DEBUG("discont timestamp %" G_GINT64_FORMAT, 
                      GST_BUFFER_TIMESTAMP(outbuf));
                  GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DISCONT);
                  schro_dec->discont = FALSE;
                }
          
//schro_dec->timestamp_offset = 30*GST_SECOND;
                GST_BUFFER_TIMESTAMP(outbuf) = gst_util_uint64_scale (
                    schro_dec->n_frames, schro_dec->fps_d * GST_SECOND,
                    schro_dec->fps_n) +
                  schro_dec->timestamp_offset;
                GST_BUFFER_DURATION(outbuf) = gst_util_uint64_scale_int (GST_SECOND,
                   schro_dec->fps_d, schro_dec->fps_n);

                gst_buffer_ref(outbuf);
                ret = gst_pad_push (schro_dec->srcpad, outbuf);
                if (ret != GST_FLOW_OK)
                  return ret;
              } else {
                GST_DEBUG("skipped frame %d", frame->frame_number);
              }
        
              schro_dec->n_frames++;
              schro_frame_free (frame);
            }

            break;
          case SCHRO_DECODER_EOS:
            break;
          case SCHRO_DECODER_ERROR:
            /* FIXME */
            GST_ERROR ("codec error");
            break;
        }
      }
    }
  }

  return GST_FLOW_OK;
}


