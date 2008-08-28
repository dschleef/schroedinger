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

#ifndef _GST_BASE_VIDEO_CODER_H_
#define _GST_BASE_VIDEO_CODER_H_

#include <gst/gst.h>
#include <gst/video/video.h>

#define GST_TYPE_BASE_VIDEO_CODER \
  (gst_base_video_coder_get_type())
#define GST_BASE_VIDEO_CODER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_BASE_VIDEO_CODER,GstBaseVideoCoder))
#define GST_BASE_VIDEO_CODER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_BASE_VIDEO_CODER,GstBaseVideoCoderClass))
#define GST_BASE_VIDEO_CODER_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj),GST_TYPE_BASE_VIDEO_CODER,GstBaseVideoCoderClass))
#define GST_IS_BASE_VIDEO_CODER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_BASE_VIDEO_CODER))
#define GST_IS_BASE_VIDEO_CODER_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_BASE_VIDEO_CODER))

/**
 * GST_BASE_VIDEO_CODER_SINK_NAME:
 *
 * The name of the templates for the sink pad.
 */
#define GST_BASE_VIDEO_CODER_SINK_NAME    "sink"
/**
 * GST_BASE_VIDEO_CODER_SRC_NAME:
 *
 * The name of the templates for the source pad.
 */
#define GST_BASE_VIDEO_CODER_SRC_NAME     "src"

/**
 * GST_BASE_VIDEO_CODER_SRC_PAD:
 * @obj: base video coder instance
 *
 * Gives the pointer to the source #GstPad object of the element.
 */
#define GST_BASE_VIDEO_CODER_SRC_PAD(obj)         (GST_BASE_VIDEO_CODER_CAST (obj)->srcpad)

/**
 * GST_BASE_VIDEO_CODER_SINK_PAD:
 * @obj: base video coder instance
 *
 * Gives the pointer to the sink #GstPad object of the element.
 */
#define GST_BASE_VIDEO_CODER_SINK_PAD(obj)        (GST_BASE_VIDEO_CODER_CAST (obj)->sinkpad)


typedef struct _GstBaseVideoCoder GstBaseVideoCoder;
typedef struct _GstBaseVideoCoderClass GstBaseVideoCoderClass;
typedef struct _GstVideoFrame GstVideoFrame;

struct _GstBaseVideoCoder
{
  GstElement element;

  /*< private >*/
  GstPad *sinkpad;
  GstPad *srcpad;

  GList *frames;

  GstVideoFormat format;
  int width, height;
  int fps_n, fps_d;
  int par_n, par_d;

  gboolean sink_clipping;

  GstSegment segment;

  int bytes_per_picture;
  guint64 presentation_frame_number;
  guint64 system_frame_number;
  int distance_from_sync;

  GstCaps *caps;
  gboolean set_output_caps;
};

struct _GstBaseVideoCoderClass
{
  GstElementClass element_class;

  gboolean (*set_format) (GstBaseVideoCoder *coder, GstVideoFormat,
      int width, int height, int fps_n, int fps_d,
      int par_n, int par_d);
  gboolean (*start) (GstBaseVideoCoder *coder);
  gboolean (*stop) (GstBaseVideoCoder *coder);
  gboolean (*finish) (GstBaseVideoCoder *coder, GstVideoFrame *frame);
  gboolean (*handle_frame) (GstBaseVideoCoder *coder, GstVideoFrame *frame);
  GstFlowReturn (*shape_output) (GstBaseVideoCoder *coder, GstVideoFrame *frame);
  GstCaps *(*get_caps) (GstBaseVideoCoder *coder);

};

struct _GstVideoFrame
{
  guint64 decode_timestamp;
  guint64 presentation_timestamp;
  guint64 presentation_duration;

  gint system_frame_number;
  gint decode_frame_number;
  gint presentation_frame_number;

  int distance_from_sync;
  gboolean is_sync_point;
  gboolean is_eos;

  GstBuffer *sink_buffer;
  GstBuffer *src_buffer;

  void *coder_hook;
};

GType gst_base_video_coder_get_type (void);

int gst_base_video_coder_get_width (GstBaseVideoCoder *coder);
int gst_base_video_coder_get_height (GstBaseVideoCoder *coder);

guint64 gst_base_video_coder_get_timestamp_offset (GstBaseVideoCoder *coder);

GstVideoFrame *gst_base_video_coder_get_frame (GstBaseVideoCoder *coder,
    int frame_number);
GstFlowReturn gst_base_video_coder_finish_frame (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame);
GstFlowReturn gst_base_video_coder_end_of_stream (GstBaseVideoCoder *base_video_coder,
    GstBuffer *buffer);

#endif

