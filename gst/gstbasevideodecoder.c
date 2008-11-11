/* GStreamer
 * Copyright (C) 2008 David Schleef <ds@schleef.org>
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

#include "gstbasevideodecoder.h"

static void gst_base_video_decoder_finalize (GObject *object);

static gboolean gst_base_video_decoder_sink_setcaps (GstPad *pad, GstCaps *caps);
static gboolean gst_base_video_decoder_sink_event (GstPad *pad, GstEvent *event);
static GstFlowReturn gst_base_video_decoder_chain (GstPad *pad, GstBuffer *buf);
//static GstFlowReturn gst_base_video_decoder_process (GstBaseVideoDecoder *base_video_decoder);
static GstStateChangeReturn gst_base_video_decoder_change_state (GstElement *element,
    GstStateChange transition);
static const GstQueryType * gst_base_video_decoder_get_query_types (GstPad *pad);
static gboolean gst_base_video_decoder_src_query (GstPad *pad, GstQuery *query);


static GstElementClass *parent_class;

GST_BOILERPLATE (GstBaseVideoDecoder, gst_base_video_decoder, GstElement,
    GST_TYPE_ELEMENT);

static void
gst_base_video_decoder_base_init (gpointer g_class)
{

}

static void
gst_base_video_decoder_class_init (GstBaseVideoDecoderClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->finalize = gst_base_video_decoder_finalize;

  gstelement_class->change_state = gst_base_video_decoder_change_state;

  parent_class = g_type_class_peek_parent (klass);
}

static void
gst_base_video_decoder_init (GstBaseVideoDecoder *base_video_decoder,
    GstBaseVideoDecoderClass *klass)
{
  GstPadTemplate *pad_template;

  GST_DEBUG ("gst_base_video_decoder_init");

  pad_template =
    gst_element_class_get_pad_template (GST_ELEMENT_CLASS (klass), "sink");
  g_return_if_fail (pad_template != NULL);
  base_video_decoder->sinkpad = gst_pad_new_from_template (pad_template, "sink");

  gst_pad_set_chain_function (base_video_decoder->sinkpad, gst_base_video_decoder_chain);
  gst_pad_set_event_function (base_video_decoder->sinkpad, gst_base_video_decoder_sink_event);
  gst_pad_set_setcaps_function (base_video_decoder->sinkpad, gst_base_video_decoder_sink_setcaps);
  //gst_pad_set_query_function (base_video_decoder->sinkpad, gst_base_video_decoder_sink_query);
  gst_element_add_pad (GST_ELEMENT(base_video_decoder), base_video_decoder->sinkpad);

  pad_template =
    gst_element_class_get_pad_template (GST_ELEMENT_CLASS (klass), "src");
  g_return_if_fail (pad_template != NULL);
  base_video_decoder->srcpad = gst_pad_new_from_template (pad_template, "src");

  gst_pad_set_query_type_function (base_video_decoder->srcpad, gst_base_video_decoder_get_query_types);
  gst_pad_set_query_function (base_video_decoder->srcpad, gst_base_video_decoder_src_query);
  gst_element_add_pad (GST_ELEMENT(base_video_decoder), base_video_decoder->srcpad);
}

static gboolean
gst_base_video_decoder_sink_setcaps (GstPad *pad, GstCaps *caps)
{
  GstBaseVideoDecoder *base_video_decoder;
  GstBaseVideoDecoderClass *base_video_decoder_class;

  base_video_decoder = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));
  base_video_decoder_class = GST_BASE_VIDEO_DECODER_GET_CLASS(base_video_decoder);

  GST_DEBUG("setcaps");

  gst_video_format_parse_caps (caps, &base_video_decoder->format,
      &base_video_decoder->width,
      &base_video_decoder->height);
  gst_video_parse_caps_framerate (caps, &base_video_decoder->fps_n,
      &base_video_decoder->fps_d);

  base_video_decoder->par_n = 1;
  base_video_decoder->par_d = 1;
  gst_video_parse_caps_pixel_aspect_ratio (caps, &base_video_decoder->par_n,
      &base_video_decoder->par_d);

  base_video_decoder_class->set_format (base_video_decoder,
      base_video_decoder->format,
      base_video_decoder->width,
      base_video_decoder->height,
      base_video_decoder->fps_n,
      base_video_decoder->fps_d,
      base_video_decoder->par_n,
      base_video_decoder->par_d);

  base_video_decoder_class->start (base_video_decoder);

  g_object_unref (base_video_decoder);

  return TRUE;
}

static void
gst_base_video_decoder_finalize (GObject *object)
{
  GstBaseVideoDecoder *base_video_decoder;
  GstBaseVideoDecoderClass *base_video_decoder_class;

  g_return_if_fail (GST_IS_BASE_VIDEO_DECODER (object));
  base_video_decoder = GST_BASE_VIDEO_DECODER (object);
  base_video_decoder_class = GST_BASE_VIDEO_DECODER_GET_CLASS (object);

  GST_DEBUG("finalize");

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_base_video_decoder_sink_event (GstPad *pad, GstEvent *event)
{
  GstBaseVideoDecoder *base_video_decoder;
  GstBaseVideoDecoderClass *base_video_decoder_class;
  gboolean ret = FALSE;

  base_video_decoder = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));
  base_video_decoder_class = GST_BASE_VIDEO_DECODER_GET_CLASS (base_video_decoder);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      {
        GstVideoFrame *frame;

        frame = g_malloc0 (sizeof(GstVideoFrame));
        frame->presentation_frame_number = base_video_decoder->presentation_frame_number;
        frame->presentation_duration = 0;
        //frame->presentation_duration = 0;
        base_video_decoder->presentation_frame_number++;

        base_video_decoder->frames = g_list_append(base_video_decoder->frames, frame);
        base_video_decoder_class->finish (base_video_decoder, frame);

        ret = gst_pad_push_event (base_video_decoder->srcpad, event);
      }
      break;
    case GST_EVENT_NEWSEGMENT:
      {
        gboolean update;
        double rate;
        double applied_rate;
        GstFormat format;
        gint64 start;
        gint64 stop;
        gint64 position;

        gst_event_parse_new_segment_full (event, &update, &rate,
            &applied_rate, &format, &start, &stop, &position);

        if (format != GST_FORMAT_TIME)
          goto newseg_wrong_format;

        GST_DEBUG("new segment %lld %lld", start, position);

        gst_segment_set_newsegment_full (&base_video_decoder->segment, update,
            rate, applied_rate, format, start, stop, position);

        ret = gst_pad_push_event (base_video_decoder->srcpad, event);
      }
      break;
    default:
      /* FIXME this changes the order of events */
      ret = gst_pad_push_event (base_video_decoder->srcpad, event);
      break;
  }

done:
  gst_object_unref (base_video_decoder);
  return ret;

newseg_wrong_format:
  {
    GST_DEBUG_OBJECT (base_video_decoder, "received non TIME newsegment");
    gst_event_unref (event);
    goto done;
  }
}

#if 0
static gboolean
gst_base_video_decoder_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstBaseVideoDecoder *enc;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  enc = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a decoding state */

  switch (src_format) {
    case GST_FORMAT_BYTES:
      switch (*dest_format) {
#if 0
        case GST_FORMAT_DEFAULT:
          *dest_value = gst_util_uint64_scale_int (src_value, 1,
              enc->bytes_per_picture);
          break;
#endif
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
              GST_SECOND * enc->fps_d, enc->fps_n);
          break;
#if 0
        case GST_FORMAT_BYTES:
          *dest_value = gst_util_uint64_scale_int (src_value,
              enc->bytes_per_picture, 1);
          break;
#endif
        default:
          res = FALSE;
      }
      break;
    default:
      res = FALSE;
      break;
  }
}
#endif

static gboolean
gst_base_video_decoder_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{ 
  gboolean res = TRUE;
  GstBaseVideoDecoder *enc;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  } 

  enc = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a encoding state */

  GST_ERROR("src convert");
  switch (src_format) {
#if 0
    case GST_FORMAT_DEFAULT:
      switch (*dest_format) {
        case GST_FORMAT_TIME:
          *dest_value = gst_util_uint64_scale (granulepos_to_frame (src_value),
              enc->fps_d * GST_SECOND, enc->fps_n);
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
                enc->fps_n, enc->fps_d * GST_SECOND);
            break;
          }
        default:
          res = FALSE;
          break;
      }
      break;
#endif
    default:
      res = FALSE; 
      break;
  }

  gst_object_unref (enc);

  return res;
}

static const GstQueryType *
gst_base_video_decoder_get_query_types (GstPad *pad)
{
  static const GstQueryType query_types[] = {
    //GST_QUERY_POSITION,
    //GST_QUERY_DURATION,
    GST_QUERY_CONVERT,
    0
  };

  return query_types;
}

static gboolean
gst_base_video_decoder_src_query (GstPad *pad, GstQuery *query)
{
  GstBaseVideoDecoder *enc;
  gboolean res;

  enc = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));

  switch GST_QUERY_TYPE (query) {
    case GST_QUERY_CONVERT:
      {
        GstFormat src_fmt, dest_fmt;
        gint64 src_val, dest_val;

        gst_query_parse_convert (query, &src_fmt, &src_val, &dest_fmt, &dest_val);
        res = gst_base_video_decoder_src_convert (pad, src_fmt, src_val, &dest_fmt,
            &dest_val);
        if (!res) goto error;
        gst_query_set_convert (query, src_fmt, src_val, dest_fmt, dest_val);
        break;
      }
    default:
      res = gst_pad_query_default (pad, query);
  }
  gst_object_unref (enc);
  return res;

error:
  GST_DEBUG_OBJECT (enc, "query failed");
  gst_object_unref (enc);
  return res;
}

static gboolean
gst_pad_is_negotiated (GstPad *pad)
{
  GstCaps *caps;

  g_return_val_if_fail (pad != NULL, FALSE);

  caps = gst_pad_get_negotiated_caps (pad);
  if (caps) {
    gst_caps_unref (caps);
    return TRUE;
  }

  return FALSE;
}

static GstFlowReturn
gst_base_video_decoder_chain (GstPad *pad, GstBuffer *buf)
{
  GstBaseVideoDecoder *base_video_decoder;
  GstBaseVideoDecoderClass *klass;
  GstVideoFrame *frame;

  if (!gst_pad_is_negotiated (pad)) {
    return GST_FLOW_NOT_NEGOTIATED;
  }

  base_video_decoder = GST_BASE_VIDEO_DECODER (gst_pad_get_parent (pad));
  klass = GST_BASE_VIDEO_DECODER_GET_CLASS (base_video_decoder);

  if (base_video_decoder->sink_clipping) {
    gint64 start = GST_BUFFER_TIMESTAMP (buf);
    gint64 stop = start + GST_BUFFER_DURATION (buf);
    gint64 clip_start;
    gint64 clip_stop;

    if (!gst_segment_clip (&base_video_decoder->segment,
          GST_FORMAT_TIME, start, stop, &clip_start, &clip_stop)) {
      GST_DEBUG("clipping to segment dropped frame");
      goto done;
    }
  }

  frame = g_malloc0 (sizeof(GstVideoFrame));
  frame->sink_buffer = buf;
  frame->presentation_timestamp = GST_BUFFER_TIMESTAMP(buf);
  frame->presentation_duration = GST_BUFFER_DURATION(buf);
  frame->presentation_frame_number = base_video_decoder->presentation_frame_number;
  base_video_decoder->presentation_frame_number++;

  base_video_decoder->frames = g_list_append(base_video_decoder->frames, frame);

  klass->handle_frame (base_video_decoder, frame);

done:
  g_object_unref (base_video_decoder);

  return GST_FLOW_OK;
}

static GstStateChangeReturn
gst_base_video_decoder_change_state (GstElement *element, GstStateChange transition)
{
  GstBaseVideoDecoder *base_video_decoder;
  GstBaseVideoDecoderClass *base_video_decoder_class;
  GstStateChangeReturn ret;

  base_video_decoder = GST_BASE_VIDEO_DECODER (element);
  base_video_decoder_class = GST_BASE_VIDEO_DECODER_GET_CLASS (element);

  switch (transition) {
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS(parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      if (base_video_decoder_class->stop) {
        base_video_decoder_class->stop (base_video_decoder);
      }
      break;
    default:
      break;
  }

  return ret;
}

GstFlowReturn
gst_base_video_decoder_finish_frame (GstBaseVideoDecoder *base_video_decoder,
    GstVideoFrame *frame)
{
  GstFlowReturn ret;
  GstBaseVideoDecoderClass *base_video_decoder_class;

  base_video_decoder_class = GST_BASE_VIDEO_DECODER_GET_CLASS (base_video_decoder);

  frame->system_frame_number = base_video_decoder->system_frame_number;
  base_video_decoder->system_frame_number++;

  if (frame->is_sync_point) {
    base_video_decoder->distance_from_sync = 0;
    GST_BUFFER_FLAG_UNSET (frame->src_buffer, GST_BUFFER_FLAG_DELTA_UNIT);
  } else {
    GST_BUFFER_FLAG_SET (frame->src_buffer, GST_BUFFER_FLAG_DELTA_UNIT);
  }

  frame->distance_from_sync = base_video_decoder->distance_from_sync;
  base_video_decoder->distance_from_sync++;

  frame->decode_frame_number = frame->system_frame_number - 1;
  if (frame->decode_frame_number < 0) {
    frame->decode_timestamp = 0;
  } else {
    frame->decode_timestamp = gst_util_uint64_scale (frame->decode_frame_number,
        GST_SECOND * base_video_decoder->fps_d,
        base_video_decoder->fps_n);
  }

  GST_BUFFER_TIMESTAMP(frame->src_buffer) = frame->presentation_timestamp;
  GST_BUFFER_DURATION(frame->src_buffer) = frame->presentation_duration;
  GST_BUFFER_OFFSET(frame->src_buffer) = frame->decode_timestamp;

  base_video_decoder->frames = g_list_remove (base_video_decoder->frames, frame);

  if (!base_video_decoder->set_output_caps) {
    GstCaps *caps;

    if (base_video_decoder_class->get_caps) {
      caps = base_video_decoder_class->get_caps (base_video_decoder);
    } else {
      caps = gst_caps_new_simple ("video/unknown", NULL);
    }
    base_video_decoder->caps = gst_caps_ref (caps);
    gst_pad_set_caps (base_video_decoder->srcpad, caps);
    base_video_decoder->set_output_caps = TRUE;
  }

  if (base_video_decoder_class->shape_output) {
    ret = base_video_decoder_class->shape_output (base_video_decoder, frame);
  } else {
    ret = gst_pad_push (base_video_decoder->srcpad, frame->src_buffer);
  }

  g_free (frame);

  return ret;
}

int
gst_base_video_decoder_get_height (GstBaseVideoDecoder *base_video_decoder)
{
  return base_video_decoder->height;
}

int
gst_base_video_decoder_get_width (GstBaseVideoDecoder *base_video_decoder)
{
  return base_video_decoder->width;
}

GstFlowReturn
gst_base_video_decoder_end_of_stream (GstBaseVideoDecoder *base_video_decoder,
    GstBuffer *buffer)
{

  if (base_video_decoder->frames) {
    GST_ERROR("EOS with frames left over");
  }

  return gst_pad_push (base_video_decoder->srcpad, buffer);
}

