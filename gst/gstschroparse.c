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
#include <gst/base/gstadapter.h>
#include <gst/video/video.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <math.h>

#define SCHRO_ENABLE_UNSTABLE_API
#include <schroedinger/schroparse.h>


GST_DEBUG_CATEGORY_EXTERN (schro_debug);
#define GST_CAT_DEFAULT schro_debug

#define GST_TYPE_SCHRO_PARSE \
  (gst_schro_parse_get_type())
#define GST_SCHRO_PARSE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHRO_PARSE,GstSchroParse))
#define GST_SCHRO_PARSE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHRO_PARSE,GstSchroParseClass))
#define GST_IS_SCHRO_PARSE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHRO_PARSE))
#define GST_IS_SCHRO_PARSE_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHRO_PARSE))

typedef struct _GstSchroParse GstSchroParse;
typedef struct _GstSchroParseClass GstSchroParseClass;

struct _GstSchroParse
{
  GstElement element;

  GstPad *sinkpad, *srcpad;
  GstAdapter *adapter;

  int wavelet_type;
  int level;

  SchroDecoder *decoder;

  /* video properties */
  int fps_n, fps_d;
  int par_n, par_d;
  int width, height;
  guint64 duration;
  GstCaps *caps;
  
  /* state */
  gboolean have_seq_header;
  int picture_number;
  GstSegment segment;
  gboolean discont;
  uint64_t granulepos_offset;
  uint64_t granulepos_low;
  uint64_t granulepos_hi;
  gint64 timestamp_offset;
  gboolean update_granulepos;

  int bytes_per_picture;
};

struct _GstSchroParseClass
{
  GstElementClass element_class;
};


/* GstSchroParse signals and args */
enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0
};

static void gst_schro_parse_finalize (GObject *object);

static const GstQueryType *gst_schro_parse_get_query_types (GstPad *pad);
static gboolean gst_schro_parse_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value);
static gboolean gst_schro_parse_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value);
static gboolean gst_schro_parse_src_query (GstPad *pad, GstQuery *query);
static gboolean gst_schro_parse_sink_query (GstPad *pad, GstQuery *query);
static gboolean gst_schro_parse_src_event (GstPad *pad, GstEvent *event);
static gboolean gst_schro_parse_sink_event (GstPad *pad, GstEvent *event);
static GstStateChangeReturn gst_schro_parse_change_state (GstElement *element,
    GstStateChange transition);
static GstFlowReturn gst_schro_parse_push_all (GstSchroParse *schro_parse, 
    gboolean at_eos);
static GstFlowReturn gst_schro_parse_chain (GstPad *pad, GstBuffer *buf);

static GstStaticPadTemplate gst_schro_parse_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

static GstStaticPadTemplate gst_schro_parse_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

GST_BOILERPLATE (GstSchroParse, gst_schro_parse, GstElement, GST_TYPE_ELEMENT);

static void
gst_schro_parse_base_init (gpointer g_class)
{
  static GstElementDetails compress_details =
      GST_ELEMENT_DETAILS ("Dirac Parser",
      "Codec/Parser/Video",
      "Parse Dirac streams",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_parse_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_parse_sink_template));

  gst_element_class_set_details (element_class, &compress_details);
}

static void
gst_schro_parse_class_init (GstSchroParseClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);

  gobject_class->finalize = gst_schro_parse_finalize;

  element_class->change_state = gst_schro_parse_change_state;
}

static void
gst_schro_parse_init (GstSchroParse *schro_parse, GstSchroParseClass *klass)
{
  GST_DEBUG ("gst_schro_parse_init");

  schro_parse->decoder = schro_decoder_new ();

  schro_parse->sinkpad = gst_pad_new_from_static_template (&gst_schro_parse_sink_template, "sink");
  gst_pad_set_chain_function (schro_parse->sinkpad, gst_schro_parse_chain);
  gst_pad_set_query_function (schro_parse->sinkpad, gst_schro_parse_sink_query);
  gst_pad_set_event_function (schro_parse->sinkpad, gst_schro_parse_sink_event);
  gst_element_add_pad (GST_ELEMENT(schro_parse), schro_parse->sinkpad);

  schro_parse->srcpad = gst_pad_new_from_static_template (&gst_schro_parse_src_template, "src");
  gst_pad_set_query_type_function (schro_parse->srcpad, gst_schro_parse_get_query_types);
  gst_pad_set_query_function (schro_parse->srcpad, gst_schro_parse_src_query);
  gst_pad_set_event_function (schro_parse->srcpad, gst_schro_parse_src_event);
  gst_element_add_pad (GST_ELEMENT(schro_parse), schro_parse->srcpad);

  schro_parse->adapter = gst_adapter_new ();
}

static void
gst_schro_parse_reset (GstSchroParse *dec)
{
  GST_DEBUG("reset");
  dec->have_seq_header = FALSE;
  dec->discont = TRUE;
  dec->picture_number = 0;
  dec->update_granulepos = TRUE;
  dec->granulepos_offset = 0;
  dec->granulepos_low = 0;
  dec->granulepos_hi = 0;
  dec->duration = GST_SECOND/30;
  dec->fps_n = 30;
  dec->fps_d = 1;
  dec->par_n = 1;
  dec->par_d = 1;
  dec->width = -1;
  dec->height = -1;

  gst_segment_init (&dec->segment, GST_FORMAT_TIME);
  gst_adapter_clear (dec->adapter);
}

static void
gst_schro_parse_finalize (GObject *object)
{
  GstSchroParse *schro_parse;

  g_return_if_fail (GST_IS_SCHRO_PARSE (object));
  schro_parse = GST_SCHRO_PARSE (object);

  if (schro_parse->decoder) {
    schro_decoder_free (schro_parse->decoder);
  }
  if (schro_parse->adapter) {
    g_object_unref (schro_parse->adapter);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

#define OGG_DIRAC_GRANULE_SHIFT 32
#define OGG_DIRAC_GRANULE_LOW_MASK ((1ULL<<OGG_DIRAC_GRANULE_SHIFT)-1)

#if 0
static gint64
granulepos_to_frame (gint64 granulepos)
{
  if (granulepos == -1) return -1;

  return (granulepos >> OGG_DIRAC_GRANULE_SHIFT) +
    (granulepos & OGG_DIRAC_GRANULE_LOW_MASK);
}
#endif

static const GstQueryType *
gst_schro_parse_get_query_types (GstPad *pad)
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
gst_schro_parse_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstSchroParse *dec;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  dec = GST_SCHRO_PARSE (gst_pad_get_parent (pad));

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
gst_schro_parse_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstSchroParse *dec;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  dec = GST_SCHRO_PARSE (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a decoding state */

  switch (src_format) {
    case GST_FORMAT_DEFAULT:
      switch (*dest_format) {
        case GST_FORMAT_TIME:
          *dest_value = gst_util_uint64_scale (src_value,
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
gst_schro_parse_src_query (GstPad *pad, GstQuery *query)
{
  GstSchroParse *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_PARSE (gst_pad_get_parent(pad));

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_POSITION:
    {
      GstFormat format;
      gint64 time;
      gint64 value;

      gst_query_parse_position (query, &format, NULL);

      time = gst_util_uint64_scale (dec->picture_number,
              dec->fps_n, dec->fps_d);
      //time -= dec->segment.start;
      time += dec->segment.time;
      GST_DEBUG("query position %lld", time);
      res = gst_schro_parse_src_convert (pad, GST_FORMAT_TIME, time,
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
      res = gst_schro_parse_src_convert (pad, src_fmt, src_val, &dest_fmt,
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
gst_schro_parse_sink_query (GstPad *pad, GstQuery *query)
{
  GstSchroParse *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_PARSE (gst_pad_get_parent(pad));

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CONVERT:
    {
      GstFormat src_fmt, dest_fmt;
      gint64 src_val, dest_val;

      gst_query_parse_convert (query, &src_fmt, &src_val, &dest_fmt, &dest_val);
      res = gst_schro_parse_sink_convert (pad, src_fmt, src_val, &dest_fmt,
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
gst_schro_parse_src_event (GstPad *pad, GstEvent *event)
{
  GstSchroParse *dec;
  gboolean res = FALSE;

  dec = GST_SCHRO_PARSE (gst_pad_get_parent(pad));

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
      res = gst_schro_parse_src_convert (pad, format, cur, &tformat, &tcur);
      if (!res) goto convert_error;
      res = gst_schro_parse_src_convert (pad, format, stop, &tformat, &tstop);
      if (!res) goto convert_error;

      real_seek = gst_event_new_seek (rate, GST_FORMAT_TIME,
          flags, cur_type, tcur, stop_type, tstop);

      res = gst_pad_push_event (dec->sinkpad, real_seek);

      break;
    }
#if 0
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

      GST_DEBUG_OBJECT (dec, "got QoS %" GST_TIME_FORMAT ", %" G_GINT64_FORMAT,
          GST_TIME_ARGS(timestamp), diff);

      res = gst_pad_push_event (dec->sinkpad, event);
      break;
    }
#endif
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
gst_schro_parse_sink_event (GstPad *pad, GstEvent *event)
{
  GstSchroParse *dec;
  gboolean ret = FALSE;

  dec = GST_SCHRO_PARSE (gst_pad_get_parent(pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_FLUSH_START:
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    case GST_EVENT_FLUSH_STOP:
      gst_schro_parse_reset (dec);
      ret = gst_pad_push_event (dec->srcpad, event);
      break;
    case GST_EVENT_EOS:
      if (gst_schro_parse_push_all (dec, FALSE) == GST_FLOW_ERROR) {
        gst_event_unref (event);
        return FALSE;
      }

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

      GST_DEBUG("newsegment %lld %lld", start, time);
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
gst_schro_parse_change_state (GstElement *element, GstStateChange transition)
{
  GstSchroParse *dec = GST_SCHRO_PARSE (element);
  GstStateChangeReturn ret;

  switch (transition) {
    case GST_STATE_CHANGE_NULL_TO_READY:
      break;
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_schro_parse_reset (dec);
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
      gst_schro_parse_reset (dec);
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    default:
      break;
  }

  return ret;
}

static void
handle_sequence_header (GstSchroParse *schro_parse, guint8 *data, int size)
{
  SchroVideoFormat video_format;
  int ret;

  ret = schro_parse_decode_sequence_header (data + 13, size - 13,
      &video_format);
  if (ret) {
    schro_parse->fps_n = video_format.frame_rate_numerator;
    schro_parse->fps_d = video_format.frame_rate_denominator;
    GST_INFO("Frame rate is %d/%d", schro_parse->fps_n,
        schro_parse->fps_d);

    schro_parse->width = video_format.width;
    schro_parse->height = video_format.height;
    GST_INFO("Frame dimensions are %d x %d\n", schro_parse->width,
        schro_parse->height);

    schro_parse->par_n = video_format.aspect_ratio_numerator;
    schro_parse->par_d = video_format.aspect_ratio_denominator;
    GST_INFO("Pixel aspect ratio is %d/%d", schro_parse->par_n,
        schro_parse->par_d);
  } else {
    GST_ERROR("Failed to get frame rate from sequence header");
  }
}

static GstFlowReturn
gst_schro_parse_push_all (GstSchroParse *schro_parse, gboolean at_eos)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *outbuf;

  while (TRUE) {
    int size;
    unsigned char header[SCHRO_PARSE_HEADER_SIZE];
    guint8 *data;
    int parse_code;
    int presentation_frame;

    if (gst_adapter_available (schro_parse->adapter) <
        SCHRO_PARSE_HEADER_SIZE) {
      /* Need more data */
      return GST_FLOW_OK;
    }
    gst_adapter_copy (schro_parse->adapter, header, 0, SCHRO_PARSE_HEADER_SIZE);

    if (memcmp (header, "BBCD", 4) != 0) {
      /* bad header or lost sync */
      /* FIXME: we should handle this */
      return GST_FLOW_ERROR;
    }

    size = GST_READ_UINT32_BE (header + 5);

    GST_LOG ("Have complete parse unit of %d bytes", size);

    if (size == 0) {
      size = 13;
    }

    if (gst_adapter_available (schro_parse->adapter) < size) {
      return GST_FLOW_OK;
    }

    outbuf = gst_adapter_take_buffer (schro_parse->adapter, size);

    data = GST_BUFFER_DATA(outbuf);
    parse_code = data[4];
    presentation_frame = schro_parse->picture_number;

    if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(parse_code)) {
      if (!schro_parse->have_seq_header) {
        handle_sequence_header (schro_parse, data, GST_BUFFER_SIZE(outbuf));
        schro_parse->have_seq_header = TRUE;
      }

      schro_parse->update_granulepos = TRUE;
    }
    if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
      if (schro_parse->update_granulepos) {
        schro_parse->granulepos_hi = schro_parse->granulepos_offset +
          presentation_frame + 1;
        schro_parse->update_granulepos = FALSE;
      }
      schro_parse->granulepos_low = schro_parse->granulepos_offset +
        presentation_frame + 1 - schro_parse->granulepos_hi;
    }
    
    GST_BUFFER_OFFSET_END (outbuf) =
      (schro_parse->granulepos_hi<<OGG_DIRAC_GRANULE_SHIFT) +
      schro_parse->granulepos_low;
    GST_BUFFER_OFFSET (outbuf) = gst_util_uint64_scale (
        (schro_parse->granulepos_hi + schro_parse->granulepos_low),
        schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
    if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
      GST_BUFFER_DURATION (outbuf) = schro_parse->duration;
      GST_BUFFER_TIMESTAMP (outbuf) =
        schro_parse->timestamp_offset + gst_util_uint64_scale (
          schro_parse->picture_number,
          schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
      schro_parse->picture_number++;
      if (!SCHRO_PARSE_CODE_IS_INTRA(parse_code)) {
        GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_DELTA_UNIT);
      } else {
        GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DELTA_UNIT);
      }
    } else {
      GST_BUFFER_DURATION (outbuf) = -1;
      GST_BUFFER_TIMESTAMP (outbuf) =
        schro_parse->timestamp_offset + gst_util_uint64_scale (
          schro_parse->picture_number,
          schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
      GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DELTA_UNIT);
    }

    if (!schro_parse->caps) {
      schro_parse->caps = gst_caps_new_simple ("video/x-dirac",
              "width", G_TYPE_INT, schro_parse->width,
              "height", G_TYPE_INT, schro_parse->height,
              "framerate", GST_TYPE_FRACTION, schro_parse->fps_n,
              schro_parse->fps_d, NULL);
      
      if (schro_parse->par_d != 1 || schro_parse->par_n != 1)
          gst_caps_set_simple (schro_parse->caps, "pixel-aspect-ratio",
	      GST_TYPE_FRACTION, schro_parse->par_n, schro_parse->par_d, NULL);

      gst_pad_set_caps (schro_parse->srcpad, schro_parse->caps);
    }
    gst_buffer_set_caps (outbuf, schro_parse->caps);

    ret = gst_pad_push (schro_parse->srcpad, outbuf);
    if (ret != GST_FLOW_OK)
      break;
  }

  return ret;
}

static GstFlowReturn
gst_schro_parse_chain (GstPad *pad, GstBuffer *buf)
{
  GstSchroParse *schro_parse;

  schro_parse = GST_SCHRO_PARSE (GST_PAD_PARENT (pad));

  if (G_UNLIKELY (GST_BUFFER_FLAG_IS_SET (buf, GST_BUFFER_FLAG_DISCONT))) {
    GST_DEBUG_OBJECT (schro_parse, "received DISCONT buffer");
    //dec->need_keyframe = TRUE;
    //dec->last_timestamp = -1;
    schro_parse->discont = TRUE;
  }

  gst_adapter_push (schro_parse->adapter, buf);

  return gst_schro_parse_push_all (schro_parse, FALSE);
}

