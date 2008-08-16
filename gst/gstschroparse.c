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
  GstAdapter *input_adapter;
  GstAdapter *output_adapter;

  int wavelet_type;
  int level;

  SchroDecoder *decoder;
  int output_format;

  /* video properties */
  int fps_n, fps_d;
  int par_n, par_d;
  int width, height;
  guint64 duration;
  GstCaps *caps;
  
  /* state */
  gboolean have_seq_header;
  gboolean have_picture;
  int buf_picture_number;
  int picture_number;
  int stored_picture_number;
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

enum {
  GST_SCHRO_PARSE_OUTPUT_DIRAC,
  GST_SCHRO_PARSE_OUTPUT_QT,
  GST_SCHRO_PARSE_OUTPUT_AVI
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
static GstFlowReturn gst_schro_parse_handle_packet_ogg (GstSchroParse *schro_parse, GstBuffer *buf);
static GstFlowReturn gst_schro_parse_handle_packet_qt (GstSchroParse *schro_parse, GstBuffer *buf);
static GstFlowReturn gst_schro_parse_handle_packet_avi (GstSchroParse *schro_parse, GstBuffer *buf);
static GstFlowReturn gst_schro_parse_push_packet_qt (GstSchroParse *schro_parse);
static GstFlowReturn gst_schro_parse_push_packet_avi (GstSchroParse *schro_parse);
static GstFlowReturn gst_schro_parse_set_output_caps (GstSchroParse *schro_parse);
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
    GST_STATIC_CAPS ("video/x-dirac;video/x-qt-part;video/x-avi-part")
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

  schro_parse->input_adapter = gst_adapter_new ();
  schro_parse->output_adapter = gst_adapter_new ();

  schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_QT;
}

static void
gst_schro_parse_reset (GstSchroParse *dec)
{
  GST_DEBUG("reset");
  dec->have_seq_header = FALSE;
  dec->discont = TRUE;
  dec->have_picture = FALSE;
  dec->picture_number = 0;
  dec->buf_picture_number = -1;
  dec->stored_picture_number = -1;
  dec->update_granulepos = TRUE;
  dec->granulepos_offset = 0;
  dec->granulepos_low = 0;
  dec->granulepos_hi = 0;
  dec->duration = GST_SECOND/30;
  dec->fps_n = 30;
  dec->fps_d = 1;
  dec->par_n = 1;
  dec->par_d = 1;
  /* FIXME */
  dec->width = 320;
  dec->height = 240;

  gst_segment_init (&dec->segment, GST_FORMAT_TIME);
  gst_adapter_clear (dec->input_adapter);
  gst_adapter_clear (dec->output_adapter);
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
  if (schro_parse->input_adapter) {
    g_object_unref (schro_parse->input_adapter);
  }
  if (schro_parse->output_adapter) {
    g_object_unref (schro_parse->output_adapter);
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

static guint64
gst_schro_parse_get_timestamp (GstSchroParse *schro_parse,
    guint32 picture_number)
{
  return schro_parse->timestamp_offset +
    gst_util_uint64_scale (picture_number,
        schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
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
    GST_DEBUG("Failed to get frame rate from sequence header");
  }
}

static GstFlowReturn
gst_schro_parse_push_all (GstSchroParse *schro_parse, gboolean at_eos)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstBuffer *outbuf;

  while (ret == GST_FLOW_OK) {
    int size;
    unsigned char header[SCHRO_PARSE_HEADER_SIZE];

    if (gst_adapter_available (schro_parse->input_adapter) <
        SCHRO_PARSE_HEADER_SIZE) {
      /* Need more data */
      return GST_FLOW_OK;
    }
    gst_adapter_copy (schro_parse->input_adapter, header, 0, SCHRO_PARSE_HEADER_SIZE);

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

    if (gst_adapter_available (schro_parse->input_adapter) < size) {
      return GST_FLOW_OK;
    }

    GST_DEBUG("dirac data unit size=%d", size);
    outbuf = gst_adapter_take_buffer (schro_parse->input_adapter, size);

    if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_DIRAC) {
      ret = gst_schro_parse_handle_packet_ogg (schro_parse, outbuf);
    } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_QT) {
      ret = gst_schro_parse_handle_packet_qt (schro_parse, outbuf);
    } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_AVI) {
      ret = gst_schro_parse_handle_packet_avi (schro_parse, outbuf);
    } else {
      ret = GST_FLOW_ERROR;
    }
  }

  return ret;
}

static GstFlowReturn
gst_schro_parse_handle_packet_ogg (GstSchroParse *schro_parse, GstBuffer *buf)
{
  guint8 *data;
  int parse_code;
  int presentation_frame;
  GstFlowReturn ret;

  data = GST_BUFFER_DATA(buf);
  parse_code = data[4];
  presentation_frame = schro_parse->picture_number;

  if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(parse_code)) {
    if (!schro_parse->have_seq_header) {
      handle_sequence_header (schro_parse, data, GST_BUFFER_SIZE(buf));
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
  
  GST_BUFFER_OFFSET_END (buf) =
    (schro_parse->granulepos_hi<<OGG_DIRAC_GRANULE_SHIFT) +
    schro_parse->granulepos_low;
  GST_BUFFER_OFFSET (buf) = gst_util_uint64_scale (
      (schro_parse->granulepos_hi + schro_parse->granulepos_low),
      schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
  if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
    GST_BUFFER_TIMESTAMP (buf) = gst_schro_parse_get_timestamp (schro_parse,
        schro_parse->picture_number);
    GST_BUFFER_DURATION (buf) = gst_schro_parse_get_timestamp (schro_parse,
          schro_parse->picture_number + 1) - GST_BUFFER_TIMESTAMP (buf);
    schro_parse->picture_number++;
    if (!SCHRO_PARSE_CODE_IS_INTRA(parse_code)) {
      GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
    } else {
      GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
    }
  } else {
    GST_BUFFER_DURATION (buf) = -1;
    GST_BUFFER_TIMESTAMP (buf) =
      schro_parse->timestamp_offset + gst_util_uint64_scale (
        schro_parse->picture_number,
        schro_parse->fps_d * GST_SECOND, schro_parse->fps_n);
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }

  gst_buffer_set_caps (buf, schro_parse->caps);

  ret = gst_pad_push (schro_parse->srcpad, buf);

  return ret;
}

static GstFlowReturn
gst_schro_parse_handle_packet_qt (GstSchroParse *schro_parse, GstBuffer *buf)
{
  guint8 *data;
  int parse_code;
  GstFlowReturn ret = GST_FLOW_OK;

  data = GST_BUFFER_DATA(buf);
  parse_code = data[4];

  if (schro_parse->have_picture) {
    schro_parse->have_picture = FALSE;
    if (SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE(parse_code)) {
      gst_adapter_push (schro_parse->output_adapter, buf);
      return gst_schro_parse_push_packet_qt (schro_parse);
    } else {
      ret = gst_schro_parse_push_packet_qt (schro_parse);
    }
  }

  if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(parse_code)) {
    if (!schro_parse->have_seq_header) {
      handle_sequence_header (schro_parse, data, GST_BUFFER_SIZE(buf));
      schro_parse->have_seq_header = TRUE;
    }
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
    schro_parse->have_picture = TRUE;
    schro_parse->buf_picture_number = GST_READ_UINT32_BE (data+13);
  }
  gst_adapter_push (schro_parse->output_adapter, buf);

  return ret;
}

static GstFlowReturn
gst_schro_parse_push_packet_qt (GstSchroParse *schro_parse)
{
  GstBuffer *buf;

  buf = gst_adapter_take_buffer (schro_parse->output_adapter,
      gst_adapter_available (schro_parse->output_adapter));

  GST_BUFFER_TIMESTAMP (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->buf_picture_number);
  GST_BUFFER_DURATION (buf) = gst_schro_parse_get_timestamp (schro_parse,
        schro_parse->buf_picture_number + 1) - GST_BUFFER_TIMESTAMP (buf);
  GST_BUFFER_OFFSET_END (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->picture_number);

  if (schro_parse->have_seq_header &&
      schro_parse->picture_number == schro_parse->buf_picture_number) {
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  } else {
    GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }
  schro_parse->have_seq_header = FALSE;

  schro_parse->picture_number++;

  gst_buffer_set_caps (buf, schro_parse->caps);

  return gst_pad_push (schro_parse->srcpad, buf);
}

static GstFlowReturn
gst_schro_parse_handle_packet_avi (GstSchroParse *schro_parse, GstBuffer *buf)
{
  guint8 *data;
  int parse_code;
  GstFlowReturn ret = GST_FLOW_OK;

  data = GST_BUFFER_DATA(buf);
  parse_code = data[4];

  while (schro_parse->have_picture) {
    schro_parse->have_picture = FALSE;
    if (SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE(parse_code)) {
      GST_DEBUG("storing");
      gst_adapter_push (schro_parse->output_adapter, buf);
      GST_DEBUG("pushing");
      return gst_schro_parse_push_packet_avi (schro_parse);
    } else {
      GST_DEBUG("pushing");
      if (schro_parse->buf_picture_number == schro_parse->picture_number ||
          schro_parse->stored_picture_number == schro_parse->picture_number) {
        ret = gst_schro_parse_push_packet_avi (schro_parse);
        /* FIXME if we get an error here, we might end up in an
         * inconsistent state. */
      } else {
        schro_parse->stored_picture_number = schro_parse->buf_picture_number;
        GST_DEBUG("OOO picture %d", schro_parse->stored_picture_number);
        /* no longer counts as a seek point */
        schro_parse->have_seq_header = FALSE;
      }
    }

    GST_DEBUG("stored %d picture %d", schro_parse->stored_picture_number,
        schro_parse->picture_number);
    if (schro_parse->stored_picture_number == schro_parse->picture_number) {
      schro_parse->have_picture = TRUE;
    }
  }

  if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(parse_code)) {
    if (!schro_parse->have_seq_header) {
      handle_sequence_header (schro_parse, data, GST_BUFFER_SIZE(buf));
      schro_parse->have_seq_header = TRUE;
    }
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
    schro_parse->have_picture = TRUE;
    schro_parse->buf_picture_number = GST_READ_UINT32_BE (data+13);
    GST_DEBUG("picture number %d", schro_parse->buf_picture_number);
  }
  GST_DEBUG("storing");
  gst_adapter_push (schro_parse->output_adapter, buf);

  GST_DEBUG("returning %d");
  return ret;
}

static GstFlowReturn
gst_schro_parse_push_packet_avi (GstSchroParse *schro_parse)
{
  GstBuffer *buf;
  int size;

  size = gst_adapter_available (schro_parse->output_adapter);
  GST_DEBUG("size %d", size);
  if (size > 0) {
    buf = gst_adapter_take_buffer (schro_parse->output_adapter, size);
  } else {
    buf = gst_buffer_new_and_alloc (0);
  }

  GST_BUFFER_TIMESTAMP (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->buf_picture_number);
  GST_BUFFER_DURATION (buf) = gst_schro_parse_get_timestamp (schro_parse,
        schro_parse->buf_picture_number + 1) - GST_BUFFER_TIMESTAMP (buf);
  GST_BUFFER_OFFSET_END (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->picture_number);

  GST_ERROR("buf_pic %d pic %d", schro_parse->buf_picture_number,
      schro_parse->picture_number);

  if (schro_parse->have_seq_header &&
      schro_parse->picture_number == schro_parse->buf_picture_number) {
    GST_ERROR("seek point on %d", schro_parse->picture_number);
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  } else {
    GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }
  schro_parse->have_seq_header = FALSE;

  schro_parse->picture_number++;

  gst_buffer_set_caps (buf, schro_parse->caps);

  return gst_pad_push (schro_parse->srcpad, buf);
}

static GstFlowReturn
gst_schro_parse_set_output_caps (GstSchroParse *schro_parse)
{
  GstCaps *caps;
  GstStructure *structure;

  GST_ERROR("set_output_caps");
  caps = gst_pad_get_allowed_caps (schro_parse->srcpad);

  if (gst_caps_is_empty (caps)) {
    gst_caps_unref (caps);
    return GST_FLOW_ERROR;
  }

  structure = gst_caps_get_structure (caps, 0);

  if (gst_structure_has_name (structure, "video/x-dirac")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_DIRAC;
  } else if (gst_structure_has_name (structure, "video/x-qt-part")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_QT;
  } else if (gst_structure_has_name (structure, "video/x-avi-part")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_AVI;
  } else {
    return GST_FLOW_ERROR;
  }

  gst_caps_unref (caps);

  if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_DIRAC) {
    schro_parse->caps = gst_caps_new_simple ("video/x-dirac",
        "width", G_TYPE_INT, schro_parse->width,
        "height", G_TYPE_INT, schro_parse->height,
        "framerate", GST_TYPE_FRACTION, schro_parse->fps_n,
        schro_parse->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, schro_parse->par_n,
        schro_parse->par_d, NULL);
  } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_QT) {
    schro_parse->caps = gst_caps_new_simple ("video/x-qt-part",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('d','r','a','c'),
        "width", G_TYPE_INT, schro_parse->width,
        "height", G_TYPE_INT, schro_parse->height,
        "framerate", GST_TYPE_FRACTION, schro_parse->fps_n,
        schro_parse->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, schro_parse->par_n,
        schro_parse->par_d, NULL);
  } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_AVI) {
    schro_parse->caps = gst_caps_new_simple ("video/x-avi-part",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('d','r','a','c'),
        "width", G_TYPE_INT, schro_parse->width,
        "height", G_TYPE_INT, schro_parse->height,
        "framerate", GST_TYPE_FRACTION, schro_parse->fps_n,
        schro_parse->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, schro_parse->par_n,
        schro_parse->par_d, NULL);
  }

  gst_pad_set_caps (schro_parse->srcpad, schro_parse->caps);

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_schro_parse_chain (GstPad *pad, GstBuffer *buf)
{
  GstSchroParse *schro_parse;
  GstFlowReturn ret;

  schro_parse = GST_SCHRO_PARSE (GST_PAD_PARENT (pad));

  if (G_UNLIKELY (GST_BUFFER_FLAG_IS_SET (buf, GST_BUFFER_FLAG_DISCONT))) {
    GST_DEBUG_OBJECT (schro_parse, "received DISCONT buffer");
    //dec->need_keyframe = TRUE;
    //dec->last_timestamp = -1;
    schro_parse->discont = TRUE;
  }

  if (!schro_parse->caps) {
    ret = gst_schro_parse_set_output_caps (schro_parse);
    if (ret != GST_FLOW_OK) {
      GST_ERROR("failed to set srcpad caps");
      return ret;
    }
  }

  gst_adapter_push (schro_parse->input_adapter, buf);

  return gst_schro_parse_push_all (schro_parse, FALSE);
}

