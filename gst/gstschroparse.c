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

#include "gstbasevideoparse.h"

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

typedef enum {
  GST_SCHRO_PARSE_OUTPUT_OGG,
  GST_SCHRO_PARSE_OUTPUT_QUICKTIME,
  GST_SCHRO_PARSE_OUTPUT_AVI
} GstSchroParseOutputType;

struct _GstSchroParse
{
  GstBaseVideoParse base_video_parse;

  GstPad *sinkpad, *srcpad;

  GstSchroParseOutputType output_format;

  GstBuffer *seq_header_buffer;
  
  /* state */


  gboolean have_picture;
  int buf_picture_number;
  int seq_hdr_picture_number;
  int picture_number;

  guint64 last_granulepos;

  int bytes_per_picture;
};

struct _GstSchroParseClass
{
  GstBaseVideoParseClass base_video_parse_class;
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

static gboolean gst_schro_parse_start (GstBaseVideoParse *base_video_parse);
static gboolean gst_schro_parse_stop (GstBaseVideoParse *base_video_parse);
static gboolean gst_schro_parse_reset (GstBaseVideoParse *base_video_parse);
static int gst_schro_parse_scan_for_sync (GstAdapter *adapter,
    gboolean at_eos, int offset, int n);
static gboolean gst_schro_parse_parse_data (GstBaseVideoParse *base_video_parse,
    gboolean at_eos);
static gboolean gst_schro_parse_shape_output (GstBaseVideoParse *base_video_parse, GstVideoFrame *frame);
static GstCaps * gst_schro_parse_get_caps (GstBaseVideoParse *base_video_parse);



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

GST_BOILERPLATE (GstSchroParse, gst_schro_parse, GstBaseVideoParse, GST_TYPE_BASE_VIDEO_PARSE);

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
  GstBaseVideoParseClass *base_video_parse_class;

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);
  base_video_parse_class = GST_BASE_VIDEO_PARSE_CLASS (klass);

  gobject_class->finalize = gst_schro_parse_finalize;

  base_video_parse_class->start = GST_DEBUG_FUNCPTR(gst_schro_parse_start);
  base_video_parse_class->stop = GST_DEBUG_FUNCPTR(gst_schro_parse_stop);
  base_video_parse_class->reset = GST_DEBUG_FUNCPTR(gst_schro_parse_reset);
  base_video_parse_class->parse_data = GST_DEBUG_FUNCPTR(gst_schro_parse_parse_data);
  base_video_parse_class->shape_output = GST_DEBUG_FUNCPTR(gst_schro_parse_shape_output);
  base_video_parse_class->scan_for_sync = GST_DEBUG_FUNCPTR(gst_schro_parse_scan_for_sync);
  base_video_parse_class->get_caps = GST_DEBUG_FUNCPTR(gst_schro_parse_get_caps);

}

static void
gst_schro_parse_init (GstSchroParse *schro_parse, GstSchroParseClass *klass)
{
  GstBaseVideoParse *base_video_parse = GST_BASE_VIDEO_PARSE(schro_parse);

  GST_DEBUG ("gst_schro_parse_init");

  schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_OGG;

  base_video_parse->reorder_depth = 2;
}

static gboolean
gst_schro_parse_reset (GstBaseVideoParse *base_video_parse)
{
  GstSchroParse *schro_parse;
  
  schro_parse = GST_SCHRO_PARSE(base_video_parse);

  GST_DEBUG("reset");

  return TRUE;
}

static void
gst_schro_parse_finalize (GObject *object)
{
  GstSchroParse *schro_parse;

  g_return_if_fail (GST_IS_SCHRO_PARSE (object));
  schro_parse = GST_SCHRO_PARSE (object);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_schro_parse_start (GstBaseVideoParse *base_video_parse)
{
  GstSchroParse *schro_parse = GST_SCHRO_PARSE(base_video_parse);
  GstCaps *caps;
  GstStructure *structure;

  GST_DEBUG("start");
  caps = gst_pad_get_allowed_caps (base_video_parse->srcpad);

  if (gst_caps_is_empty (caps)) {
    gst_caps_unref (caps);
    return FALSE;
  }

  structure = gst_caps_get_structure (caps, 0);

  if (gst_structure_has_name (structure, "video/x-dirac")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_OGG;
  } else if (gst_structure_has_name (structure, "video/x-qt-part")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_QUICKTIME;
  } else if (gst_structure_has_name (structure, "video/x-avi-part")) {
    schro_parse->output_format = GST_SCHRO_PARSE_OUTPUT_AVI;
  } else {
    return FALSE;
  }

  gst_caps_unref (caps);
  return TRUE;
}

static gboolean
gst_schro_parse_stop (GstBaseVideoParse *base_video_parse)
{
  return TRUE;
}

static void
parse_sequence_header (GstSchroParse *schro_parse, guint8 *data, int size)
{
  SchroVideoFormat video_format;
  int ret;
  GstVideoState *state;

  GST_DEBUG("parse_sequence_header size=%d", size);

  state = gst_base_video_parse_get_state (GST_BASE_VIDEO_PARSE(schro_parse));

  schro_parse->seq_header_buffer = gst_buffer_new_and_alloc (size);
  memcpy (GST_BUFFER_DATA (schro_parse->seq_header_buffer), data, size);

  ret = schro_parse_decode_sequence_header (data + 13, size - 13,
      &video_format);
  if (ret) {
    state->fps_n = video_format.frame_rate_numerator;
    state->fps_d = video_format.frame_rate_denominator;
    GST_DEBUG("Frame rate is %d/%d", state->fps_n,
        state->fps_d);

    state->width = video_format.width;
    state->height = video_format.height;
    GST_DEBUG("Frame dimensions are %d x %d\n", state->width,
        state->height);

    state->clean_width = video_format.clean_width;
    state->clean_height = video_format.clean_height;
    state->clean_offset_left = video_format.left_offset;
    state->clean_offset_top = video_format.top_offset;

    state->par_n = video_format.aspect_ratio_numerator;
    state->par_d = video_format.aspect_ratio_denominator;
    GST_DEBUG("Pixel aspect ratio is %d/%d", state->par_n,
        state->par_d);

    gst_base_video_parse_set_state (GST_BASE_VIDEO_PARSE (schro_parse),
        state);
  } else {
    GST_WARNING("Failed to get frame rate from sequence header");
  }

}

static int
gst_schro_parse_scan_for_sync (GstAdapter *adapter, gboolean at_eos,
    int offset, int n)
{
  int n_available = gst_adapter_available (adapter) - offset;

  if (n_available < 4) {
    if (at_eos) {
      return n_available;
    } else {
      return 0;
    }
  }

  n_available -= 3;

  return gst_adapter_masked_scan_uint32 (adapter, 0x42424344, 0xffffffff,
      offset, MIN (n, n_available - 3));
}

#ifdef unused
static gboolean
gst_schro_parse_get_packet (GstAdapter *adapter, gboolean at_eos,
    int offset, int *size)
{
  int xsize;
  unsigned char header[SCHRO_PARSE_HEADER_SIZE];
  int back;
  
  if (gst_adapter_available (adapter) - offset < SCHRO_PARSE_HEADER_SIZE) {
    /* FIXME maybe */
    return FALSE;
  }

  gst_adapter_copy_full (adapter, header, offset, SCHRO_PARSE_HEADER_SIZE);

  if (memcmp (header, "BBCD", 4) != 0) {
    return FALSE;
  }

  xsize = GST_READ_UINT32_BE (header + 5);
  if (xsize & 0xf0000000) {
    /* unreasonably large size */
    return FALSE;
  }

  back = GST_READ_UINT32_BE (header + 9);
  if (back & 0xf0000000) {
    /* unreasonably large size */
    return FALSE;
  }

  return TRUE;
}
#endif

static GstFlowReturn
gst_schro_parse_parse_data (GstBaseVideoParse *base_video_parse,
    gboolean at_eos)
{
  GstSchroParse *schro_parse;
  unsigned char header[SCHRO_PARSE_HEADER_SIZE];
  int next;
  int prev;
  int parse_code;

  GST_DEBUG("parse_data");

  schro_parse = GST_SCHRO_PARSE (base_video_parse);

  if (gst_adapter_available (base_video_parse->input_adapter) < SCHRO_PARSE_HEADER_SIZE) {
    return GST_BASE_VIDEO_PARSE_FLOW_NEED_DATA;
  }

  GST_DEBUG("available %d", gst_adapter_available (base_video_parse->input_adapter));

  gst_adapter_copy_full (base_video_parse->input_adapter, header, 0, SCHRO_PARSE_HEADER_SIZE);

  parse_code = header[4];
  next = GST_READ_UINT32_BE (header + 5);
  prev = GST_READ_UINT32_BE (header + 9);

  GST_DEBUG("%08x %02x %08x %08x",
      GST_READ_UINT32_BE (header), parse_code, next, prev);

  if (memcmp (header, "BBCD", 4) != 0 || 
      (next & 0xf0000000) || (prev & 0xf0000000)) {
    gst_base_video_parse_lost_sync (base_video_parse);
    return GST_BASE_VIDEO_PARSE_FLOW_NEED_DATA;
  }

  if (SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE (parse_code)) {
    GstVideoFrame *frame;

    if (next != 0 && next != 13) {
      GST_WARNING("next is not 0 or 13 in EOS packet (%d)", next);
    }

    gst_base_video_parse_add_to_frame (base_video_parse,
        SCHRO_PARSE_HEADER_SIZE);

    frame = gst_base_video_parse_get_frame (base_video_parse);
    frame->is_eos = TRUE;

    SCHRO_ERROR("eos");

    return gst_base_video_parse_finish_frame (base_video_parse);
  }

  if (gst_adapter_available (base_video_parse->input_adapter) < next) {
    return GST_BASE_VIDEO_PARSE_FLOW_NEED_DATA;
  }

  if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(parse_code)) {
    guint8 *data;
    
    data = g_malloc (next);

    gst_adapter_copy_full (base_video_parse->input_adapter, data, 0, next);
    parse_sequence_header (schro_parse, data, next);

    gst_base_video_parse_set_sync_point (base_video_parse);

    g_free (data);
  }

  if (SCHRO_PARSE_CODE_IS_PICTURE(parse_code)) {
    GstVideoFrame *frame;
    guint8 tmp[4];
    
    frame = gst_base_video_parse_get_frame (base_video_parse);

#if 0
    if (GST_CLOCK_TIME_IS_VALID(GST_BUFFER_TIMESTAMP(buf))) {
      frame->presentation_timestamp = GST_BUFFER_TIMESTAMP (buf);
    }
#endif

    gst_adapter_copy_full (base_video_parse->input_adapter, tmp, 13, 4);

    frame->presentation_frame_number = GST_READ_UINT32_BE (tmp);

    gst_base_video_parse_add_to_frame (base_video_parse, next);

    return gst_base_video_parse_finish_frame (base_video_parse);
  } else {
    gst_base_video_parse_add_to_frame (base_video_parse, next);
  }

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_schro_parse_shape_output_ogg (GstBaseVideoParse *base_video_parse,
    GstVideoFrame *frame)
{
  int dpn;
  int delay;
  int dist;
  int pt;
  int st;
  guint64 granulepos_hi;
  guint64 granulepos_low;
  GstBuffer *buf = frame->src_buffer;

  dpn = frame->decode_frame_number;

  pt = frame->presentation_frame_number * 2;
  st = frame->system_frame_number * 2;
  delay = pt - st + 2;
  dist = frame->distance_from_sync;

  GST_ERROR("sys %d dpn %d pt %d delay %d dist %d",
      (int)frame->system_frame_number,
      (int)frame->decode_frame_number,
      pt, delay, dist);

  granulepos_hi = (((guint64)pt - delay)<<9) | ((dist>>8));
  granulepos_low = (delay << 9) | (dist & 0xff);
  GST_DEBUG("granulepos %lld:%lld", granulepos_hi, granulepos_low);

  GST_BUFFER_OFFSET_END (buf) = (granulepos_hi << 22) | (granulepos_low);

  //GST_BUFFER_OFFSET_END(buf) = -1;

  return gst_base_video_parse_push (base_video_parse, buf);
}

static GstFlowReturn
gst_schro_parse_shape_output_quicktime (GstBaseVideoParse *base_video_parse,
    GstVideoFrame *frame)
{
  GstBuffer *buf = frame->src_buffer;
  const GstVideoState *state;

  state = gst_base_video_parse_get_state (base_video_parse);

  GST_BUFFER_TIMESTAMP (buf) = gst_video_state_get_timestamp (state,
      frame->presentation_frame_number);
  GST_BUFFER_DURATION (buf) = gst_video_state_get_timestamp (state,
      frame->presentation_frame_number + 1) - GST_BUFFER_TIMESTAMP (buf);
  GST_BUFFER_OFFSET_END (buf) = gst_video_state_get_timestamp (state,
      frame->decode_frame_number);
  GST_BUFFER_OFFSET (buf) = GST_CLOCK_TIME_NONE;

  if (frame->is_sync_point &&
      frame->presentation_frame_number == frame->system_frame_number) {
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
    GST_DEBUG("sync point");
  } else {
    GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }

  return gst_base_video_parse_push (base_video_parse, buf);
}

static GstFlowReturn
gst_schro_parse_shape_output (GstBaseVideoParse *base_video_parse,
    GstVideoFrame *frame)
{
  GstSchroParse *schro_parse;

  schro_parse = GST_SCHRO_PARSE (base_video_parse);

  switch (schro_parse->output_format) {
    case GST_SCHRO_PARSE_OUTPUT_OGG:
      return gst_schro_parse_shape_output_ogg (base_video_parse, frame);
      break;
    case GST_SCHRO_PARSE_OUTPUT_QUICKTIME:
      return gst_schro_parse_shape_output_quicktime (base_video_parse, frame);
      break;
    default:
      break;
  }

  return GST_FLOW_ERROR;
}

#if 0
static GstFlowReturn
gst_schro_parse_handle_packet_ogg (GstSchroParse *schro_parse, GstBuffer *buf)
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
      return gst_schro_parse_push_packet_ogg (schro_parse);
    } else {
      ret = gst_schro_parse_push_packet_ogg (schro_parse);
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
    if (schro_parse->have_seq_header) {
      schro_parse->seq_hdr_picture_number = GST_READ_UINT32_BE (data+13);
    }
  }
  gst_adapter_push (schro_parse->output_adapter, buf);

  return ret;
}

static GstFlowReturn
gst_schro_parse_push_packet_ogg (GstSchroParse *schro_parse)
{
  GstBuffer *buf;

  buf = gst_adapter_take_buffer (schro_parse->output_adapter,
      gst_adapter_available (schro_parse->output_adapter));

  if (schro_parse->have_seq_header) {
    schro_parse->seq_hdr_picture_number = schro_parse->picture_number;
  }
#if 0
  schro_parse->granulepos_hi = schro_parse->seq_hdr_picture_number;
  schro_parse->granulepos_low = schro_parse->picture_number - schro_parse->granulepos_hi;
#else
  {
    int dpn;
    int delay;
    int dist;
    int sys;
    int pt;
    int st;

    sys = schro_parse->picture_number;
    dpn = schro_parse->buf_picture_number;

    pt = schro_parse->buf_picture_number * 2;
    st = sys * 2;
    delay = pt - st + 2;
    dist = schro_parse->picture_number - schro_parse->seq_hdr_picture_number;

    GST_DEBUG("sys %d dpn %d pt %d delay %d dist %d",
        sys, dpn, pt, delay, dist);

    schro_parse->granulepos_hi = (((uint64_t)pt - delay)<<9) | ((dist>>8));
    schro_parse->granulepos_low = (delay << 9) | (dist & 0xff);
    GST_DEBUG("granulepos %lld:%lld",
        schro_parse->granulepos_hi, schro_parse->granulepos_low);
  }
#endif

  GST_BUFFER_OFFSET_END (buf) =
    (((uint64_t)schro_parse->granulepos_hi)<<OGG_DIRAC_GRANULE_SHIFT) +
    schro_parse->granulepos_low;
  if ((int64_t)(GST_BUFFER_OFFSET_END (buf) - schro_parse->last_granulepos) <= 0) {
    GST_DEBUG("granulepos didn't increase");
  }
  schro_parse->last_granulepos = GST_BUFFER_OFFSET_END (buf);

  GST_BUFFER_OFFSET (buf) = gst_schro_parse_get_timestamp (schro_parse,
      (schro_parse->granulepos_hi + schro_parse->granulepos_low)>>9);
  GST_BUFFER_TIMESTAMP (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->picture_number);
  GST_BUFFER_DURATION (buf) = gst_schro_parse_get_timestamp (schro_parse,
        schro_parse->picture_number + 1) - GST_BUFFER_TIMESTAMP (buf);

  if (schro_parse->have_seq_header) {
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  } else {
    GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }
  schro_parse->have_seq_header = FALSE;

  schro_parse->picture_number++;

  GST_INFO("size %d offset %lld granulepos %llu:%llu timestamp %lld duration %lld",
      GST_BUFFER_SIZE (buf),
      GST_BUFFER_OFFSET (buf),
      GST_BUFFER_OFFSET_END (buf)>>OGG_DIRAC_GRANULE_SHIFT,
      GST_BUFFER_OFFSET_END (buf)&OGG_DIRAC_GRANULE_LOW_MASK,
      GST_BUFFER_TIMESTAMP (buf),
      GST_BUFFER_DURATION (buf));

  if (!schro_parse->pushed_seq_header) {
    GST_BUFFER_OFFSET_END (schro_parse->seq_header_buffer) = -1;

    gst_pad_push (schro_parse->srcpad,
        gst_buffer_ref(schro_parse->seq_header_buffer));
    schro_parse->pushed_seq_header = TRUE;
  }

  gst_buffer_set_caps (buf, schro_parse->caps);
  return gst_pad_push (schro_parse->srcpad, buf);
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
  GST_BUFFER_OFFSET (buf) = gst_schro_parse_get_timestamp (schro_parse,
      schro_parse->picture_number - 1);
  GST_BUFFER_OFFSET_END (buf) = GST_CLOCK_TIME_NONE;

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

  GST_DEBUG("returning %d", ret);
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

  GST_DEBUG("buf_pic %d pic %d", schro_parse->buf_picture_number,
      schro_parse->picture_number);

  if (schro_parse->have_seq_header &&
      schro_parse->picture_number == schro_parse->buf_picture_number) {
    GST_DEBUG("seek point on %d", schro_parse->picture_number);
    GST_BUFFER_FLAG_UNSET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  } else {
    GST_BUFFER_FLAG_SET (buf, GST_BUFFER_FLAG_DELTA_UNIT);
  }
  schro_parse->have_seq_header = FALSE;

  schro_parse->picture_number++;

  gst_buffer_set_caps (buf, schro_parse->caps);

  return gst_pad_push (schro_parse->srcpad, buf);
}
#endif

static GstCaps *
gst_schro_parse_get_caps (GstBaseVideoParse *base_video_parse)
{
  GstCaps *caps;
  GstVideoState *state;
  GstSchroParse *schro_parse;

  schro_parse = GST_SCHRO_PARSE(base_video_parse);

  state = gst_base_video_parse_get_state (base_video_parse);

  if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_OGG) {
    caps = gst_caps_new_simple ("video/x-dirac",
        "width", G_TYPE_INT, state->width,
        "height", G_TYPE_INT, state->height,
        "framerate", GST_TYPE_FRACTION, state->fps_n,
        state->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, state->par_n,
        state->par_d,
        NULL);

    GST_BUFFER_FLAG_SET (schro_parse->seq_header_buffer, GST_BUFFER_FLAG_IN_CAPS);

    {
      GValue array = { 0 };
      GValue value = { 0 };
      GstBuffer *buf;

      g_value_init (&array, GST_TYPE_ARRAY);
      g_value_init (&value, GST_TYPE_BUFFER);
      buf = gst_buffer_copy (schro_parse->seq_header_buffer);
      gst_value_set_buffer (&value, buf);
      gst_buffer_unref (buf);
      gst_value_array_append_value (&array, &value);
      gst_structure_set_value (gst_caps_get_structure(caps,0),
          "streamheader", &array);
      g_value_unset (&value);
      g_value_unset (&array);
    }
  } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_QUICKTIME) {
    caps = gst_caps_new_simple ("video/x-qt-part",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('d','r','a','c'),
        "width", G_TYPE_INT, state->width,
        "height", G_TYPE_INT, state->height,
        "framerate", GST_TYPE_FRACTION, state->fps_n,
        state->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, state->par_n,
        state->par_d, NULL);
  } else if (schro_parse->output_format == GST_SCHRO_PARSE_OUTPUT_AVI) {
    caps = gst_caps_new_simple ("video/x-avi-part",
        "format", GST_TYPE_FOURCC, GST_MAKE_FOURCC('d','r','a','c'),
        "width", G_TYPE_INT, state->width,
        "height", G_TYPE_INT, state->height,
        "framerate", GST_TYPE_FRACTION, state->fps_n,
        state->fps_d,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, state->par_n,
        state->par_d, NULL);
  } else {
    g_assert_not_reached ();
  }

  return caps;
}


