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
#include <gst/video/video.h>
#include <string.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrobitstream.h>
#include <schroedinger/schrovirtframe.h>
#include <math.h>
#include "gstbasevideocoder.h"

GST_DEBUG_CATEGORY_EXTERN (schro_debug);
#define GST_CAT_DEFAULT schro_debug

#define GST_TYPE_SCHRO_ENC \
  (gst_schro_enc_get_type())
#define GST_SCHRO_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SCHRO_ENC,GstSchroEnc))
#define GST_SCHRO_ENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SCHRO_ENC,GstSchroEncClass))
#define GST_IS_SCHRO_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SCHRO_ENC))
#define GST_IS_SCHRO_ENC_CLASS(obj) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SCHRO_ENC))

typedef struct _GstSchroEnc GstSchroEnc;
typedef struct _GstSchroEncClass GstSchroEncClass;

struct _GstSchroEnc
{
  GstBaseVideoCoder base_coder;

  GstPad *sinkpad;
  GstPad *srcpad;

  /* parameters */
  int level;

  /* video properties */
  int width;
  int height;
  int fps_n, fps_d;
  int par_n, par_d;
  GstVideoFormat format;
  GList *streamheaders;

  /* segment properties */
  gint64 segment_start;
  gint64 segment_position;

  /* state */
  gboolean got_offset;
  gboolean update_granulepos;
  uint64_t granulepos_offset;
  uint64_t granulepos_low;
  uint64_t granulepos_hi;
  gboolean started;
  gint64 timestamp_offset;
  int picture_number;

  SchroEncoder *encoder;
  SchroVideoFormat *video_format;
  GstVideoFrame *eos_frame;
};

struct _GstSchroEncClass
{
  GstBaseVideoCoderClass parent_class;
};


enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0
};

static void gst_schro_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schro_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_schro_enc_process (GstSchroEnc *schro_enc);

static gboolean gst_schro_enc_set_format (GstBaseVideoCoder *base_video_coder,
    GstVideoFormat format, int width, int height,
    int fps_n, int fps_d, int par_n, int par_d);
static gboolean gst_schro_enc_start (GstBaseVideoCoder *base_video_coder);
static gboolean gst_schro_enc_stop (GstBaseVideoCoder *base_video_coder);
static gboolean gst_schro_enc_finish (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame);
static gboolean gst_schro_enc_handle_frame (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame);
static GstFlowReturn gst_schro_enc_shape_output (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame);
static GstCaps * gst_schro_enc_get_caps (GstBaseVideoCoder *base_video_coder);

static GstStaticPadTemplate gst_schro_enc_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (
      GST_VIDEO_CAPS_YUV ("{ I420, YV12, YUY2, UYVY, AYUV }") ";"
      GST_VIDEO_CAPS_ARGB)
    );

static GstStaticPadTemplate gst_schro_enc_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

GST_BOILERPLATE (GstSchroEnc, gst_schro_enc, GstBaseVideoCoder,
    GST_TYPE_BASE_VIDEO_CODER);

static void
gst_schro_enc_base_init (gpointer g_class)
{
  static GstElementDetails schro_enc_details =
      GST_ELEMENT_DETAILS ("Dirac Encoder",
      "Codec/Encoder/Video",
      "Encode raw YUV video into Dirac stream",
      "David Schleef <ds@schleef.org>");
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_enc_src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_schro_enc_sink_template));

  gst_element_class_set_details (element_class, &schro_enc_details);
}

static void
gst_schro_enc_class_init (GstSchroEncClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseVideoCoderClass *basevideocoder_class;
  int i;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  basevideocoder_class = GST_BASE_VIDEO_CODER_CLASS (klass);

  gobject_class->set_property = gst_schro_enc_set_property;
  gobject_class->get_property = gst_schro_enc_get_property;

  for(i=0;i<schro_encoder_get_n_settings();i++){
    const SchroEncoderSetting *setting;

    setting = schro_encoder_get_setting_info (i);

    switch (setting->type) {
      case SCHRO_ENCODER_SETTING_TYPE_BOOLEAN:
        g_object_class_install_property (gobject_class, i + 1,
            g_param_spec_boolean (setting->name, setting->name, setting->name,
              setting->default_value, G_PARAM_READWRITE));
        break;
      case SCHRO_ENCODER_SETTING_TYPE_INT:
        g_object_class_install_property (gobject_class, i + 1,
            g_param_spec_int (setting->name, setting->name, setting->name,
              setting->min, setting->max, setting->default_value,
              G_PARAM_READWRITE));
        break;
      case SCHRO_ENCODER_SETTING_TYPE_ENUM:
        g_object_class_install_property (gobject_class, i + 1,
            g_param_spec_int (setting->name, setting->name, setting->name,
              setting->min, setting->max, setting->default_value,
              G_PARAM_READWRITE));
        break;
      case SCHRO_ENCODER_SETTING_TYPE_DOUBLE:
        g_object_class_install_property (gobject_class, i + 1,
            g_param_spec_double (setting->name, setting->name, setting->name,
              setting->min, setting->max, setting->default_value,
              G_PARAM_READWRITE));
        break;
      default:
        break;
    }
  }

  basevideocoder_class->set_format = GST_DEBUG_FUNCPTR(gst_schro_enc_set_format);
  basevideocoder_class->start = GST_DEBUG_FUNCPTR(gst_schro_enc_start);
  basevideocoder_class->stop = GST_DEBUG_FUNCPTR(gst_schro_enc_stop);
  basevideocoder_class->finish = GST_DEBUG_FUNCPTR(gst_schro_enc_finish);
  basevideocoder_class->handle_frame = GST_DEBUG_FUNCPTR(gst_schro_enc_handle_frame);
  basevideocoder_class->shape_output = GST_DEBUG_FUNCPTR(gst_schro_enc_shape_output);
  basevideocoder_class->get_caps = GST_DEBUG_FUNCPTR(gst_schro_enc_get_caps);
}

static void
gst_schro_enc_init (GstSchroEnc *schro_enc, GstSchroEncClass *klass)
{
  GST_DEBUG ("gst_schro_enc_init");

  schro_enc->encoder = schro_encoder_new ();
  schro_encoder_set_packet_assembly (schro_enc->encoder, TRUE);
  schro_enc->video_format =
    schro_encoder_get_video_format (schro_enc->encoder);
}

static gboolean
gst_schro_enc_set_format (GstBaseVideoCoder *base_video_coder,
    GstVideoFormat format,
    int width, int height,
    int fps_n, int fps_d,
    int par_n, int par_d)
{
  GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);

  schro_enc->format = format;
  schro_enc->width = width;
  schro_enc->height = height;
  schro_enc->fps_n = fps_n;
  schro_enc->fps_d = fps_d;
  schro_enc->par_n = par_n;
  schro_enc->par_d = par_d;

  schro_video_format_set_std_video_format (schro_enc->video_format,
      SCHRO_VIDEO_FORMAT_CUSTOM);

  switch (schro_enc->format) {
    case GST_VIDEO_FORMAT_I420:
    case GST_VIDEO_FORMAT_YV12:
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_420;
      break;
    case GST_VIDEO_FORMAT_YUY2:
    case GST_VIDEO_FORMAT_UYVY:
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_422;
      break;
    case GST_VIDEO_FORMAT_AYUV:
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_444;
      break;
    case GST_VIDEO_FORMAT_ARGB:
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_420;
      break;
    default:
      g_assert_not_reached();
  }

  schro_enc->video_format->frame_rate_numerator = schro_enc->fps_n;
  schro_enc->video_format->frame_rate_denominator = schro_enc->fps_d;

  schro_enc->video_format->width = schro_enc->width;
  schro_enc->video_format->height = schro_enc->height;
  schro_enc->video_format->clean_width = schro_enc->width;
  schro_enc->video_format->clean_height = schro_enc->height;

  schro_enc->video_format->aspect_ratio_numerator = schro_enc->par_n;
  schro_enc->video_format->aspect_ratio_denominator = schro_enc->par_d;

  schro_video_format_set_std_signal_range (schro_enc->video_format,
      SCHRO_SIGNAL_RANGE_8BIT_VIDEO);
  schro_video_format_set_std_colour_spec (schro_enc->video_format,
      SCHRO_COLOUR_SPEC_HDTV);

  schro_encoder_set_video_format (schro_enc->encoder, schro_enc->video_format);
  schro_encoder_start (schro_enc->encoder);

  return TRUE;
}

static void
gst_schro_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSchroEnc *src;

  g_return_if_fail (GST_IS_SCHRO_ENC (object));
  src = GST_SCHRO_ENC (object);

  GST_DEBUG ("gst_schro_enc_set_property");

  if (prop_id >= 1) {
    const SchroEncoderSetting *setting;
    setting = schro_encoder_get_setting_info (prop_id - 1);
    switch(G_VALUE_TYPE(value)) {
      case G_TYPE_DOUBLE:
        schro_encoder_setting_set_double (src->encoder, setting->name,
            g_value_get_double(value));
        break;
      case G_TYPE_INT:
        schro_encoder_setting_set_double (src->encoder, setting->name,
            g_value_get_int(value));
        break;
      case G_TYPE_BOOLEAN:
        schro_encoder_setting_set_double (src->encoder, setting->name,
            g_value_get_boolean(value));
        break;
    }
  }
}

static void
gst_schro_enc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstSchroEnc *src;

  g_return_if_fail (GST_IS_SCHRO_ENC (object));
  src = GST_SCHRO_ENC (object);

  if (prop_id >= 1) {
    const SchroEncoderSetting *setting;
    setting = schro_encoder_get_setting_info (prop_id - 1);
    switch(G_VALUE_TYPE(value)) {
      case G_TYPE_DOUBLE:
        g_value_set_double (value,
            schro_encoder_setting_get_double (src->encoder, setting->name));
        break;
      case G_TYPE_INT:
        g_value_set_int (value,
            schro_encoder_setting_get_double (src->encoder, setting->name));
        break;
      case G_TYPE_BOOLEAN:
        g_value_set_boolean (value,
            schro_encoder_setting_get_double (src->encoder, setting->name));
        break;
    }
  }
}

static void
gst_schro_frame_free (SchroFrame *frame, void *priv)
{
  gst_buffer_unref (GST_BUFFER(priv));
}

static SchroFrame *
gst_schro_buffer_wrap (GstSchroEnc *schro_enc, GstBuffer *buf)
{
  SchroFrame *frame;
  int width;
  int height;

  width = gst_base_video_coder_get_width (GST_BASE_VIDEO_CODER(schro_enc));
  height = gst_base_video_coder_get_height (GST_BASE_VIDEO_CODER(schro_enc));

  switch (schro_enc->format) {
    case GST_VIDEO_FORMAT_I420:
      frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_VIDEO_FORMAT_YV12:
      frame = schro_frame_new_from_data_YV12 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_VIDEO_FORMAT_YUY2:
      frame = schro_frame_new_from_data_YUY2 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_VIDEO_FORMAT_UYVY:
      frame = schro_frame_new_from_data_UYVY (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_VIDEO_FORMAT_AYUV:
      frame = schro_frame_new_from_data_AYUV (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_VIDEO_FORMAT_ARGB:
      {
        SchroFrame *rgbframe = schro_frame_new_from_data_AYUV (GST_BUFFER_DATA (buf), width, height);
        SchroFrame *vframe1;
        SchroFrame *vframe2;
        SchroFrame *vframe3;

        vframe1 = schro_virt_frame_new_unpack (rgbframe);
        vframe2 = schro_virt_frame_new_color_matrix (vframe1);
        vframe3 = schro_virt_frame_new_subsample (vframe2, SCHRO_FRAME_FORMAT_U8_420);

        frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_420,
            width, height);
        schro_virt_frame_render (vframe3, frame);
        schro_frame_unref (rgbframe);
        schro_frame_unref (vframe1);
        schro_frame_unref (vframe2);
        schro_frame_unref (vframe3);
      }
      break;
    default:
      g_assert_not_reached();
  }
  schro_frame_set_free_callback (frame, gst_schro_frame_free, buf);

  return frame;
}

static gboolean
gst_schro_enc_start (GstBaseVideoCoder *base_video_coder)
{
  //GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);

  return TRUE;
}

static gboolean
gst_schro_enc_stop (GstBaseVideoCoder *base_video_coder)
{
  GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);
  
  if (schro_enc->encoder) {
    schro_encoder_free (schro_enc->encoder);
    schro_enc->encoder = NULL;
  }

  return TRUE;
}

static gboolean
gst_schro_enc_finish (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame)
{
  GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);

  GST_DEBUG("finish");

  schro_enc->eos_frame = frame;

  schro_encoder_end_of_stream (schro_enc->encoder);
  gst_schro_enc_process (schro_enc);

  return TRUE;
}

static gboolean
gst_schro_enc_handle_frame (GstBaseVideoCoder *base_video_coder,
    GstVideoFrame *frame)
{
  GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);
  SchroFrame *schro_frame;
  GstFlowReturn ret;

  schro_frame = gst_schro_buffer_wrap (schro_enc, frame->sink_buffer);

  GST_DEBUG ("pushing frame %p", frame);
  schro_encoder_push_frame_full (schro_enc->encoder, schro_frame, frame);

  ret = gst_schro_enc_process (schro_enc);

  return ret;
}

static void
gst_caps_add_streamheader (GstCaps *caps, GList *list)
{
  GValue array = { 0 };
  GValue value = { 0 };
  GstBuffer *buf;
  GList *g;

  g_value_init (&array, GST_TYPE_ARRAY);

  for(g=g_list_first (list); g; g = g_list_next (list)) {
    g_value_init (&value, GST_TYPE_BUFFER);
    buf = gst_buffer_copy (GST_BUFFER(g->data));
    gst_value_set_buffer (&value, buf);
    gst_buffer_unref (buf);
    gst_value_array_append_value (&array, &value);
    g_value_unset (&value);
  }
  gst_structure_set_value (gst_caps_get_structure(caps,0),
      "streamheader", &array);
  g_value_unset (&array);
}

static GstCaps *
gst_schro_enc_get_caps (GstBaseVideoCoder *base_video_coder)
{
  GstSchroEnc *schro_enc = GST_SCHRO_ENC(base_video_coder);
  GstCaps *caps;

  caps = gst_caps_new_simple ("video/x-dirac",
      "width", G_TYPE_INT, base_video_coder->width,
      "height", G_TYPE_INT, base_video_coder->height,
      "framerate", GST_TYPE_FRACTION, base_video_coder->fps_n,
      base_video_coder->fps_d,
      "pixel-aspect-ratio", GST_TYPE_FRACTION, base_video_coder->par_n,
      base_video_coder->par_d,
      NULL);
  gst_caps_add_streamheader (caps, schro_enc->streamheaders);

  return caps;

}

static GstFlowReturn
gst_schro_enc_shape_output (GstBaseVideoCoder *base_video_coder,
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

  GST_DEBUG("sys %d dpn %d pt %d delay %d dist %d",
      (int)frame->system_frame_number,
      (int)frame->decode_frame_number,
      pt, delay, dist);

  granulepos_hi = (((uint64_t)pt - delay)<<9) | ((dist>>8));
  granulepos_low = (delay << 9) | (dist & 0xff);
  GST_DEBUG("granulepos %lld:%lld", granulepos_hi, granulepos_low);

  GST_BUFFER_OFFSET_END (buf) = (granulepos_hi << 22) | (granulepos_low);

  gst_buffer_set_caps (buf, base_video_coder->caps);

  return gst_pad_push (base_video_coder->srcpad, buf);
}

static GstFlowReturn
gst_schro_enc_process (GstSchroEnc *schro_enc)
{
  SchroBuffer *encoded_buffer;
  GstVideoFrame *frame;
  GstFlowReturn ret;
  int presentation_frame;
  void *voidptr;
  GstBaseVideoCoder *base_video_coder = GST_BASE_VIDEO_CODER(schro_enc);

  GST_DEBUG("process");

  while (1) {
    switch (schro_encoder_wait (schro_enc->encoder)) {
      case SCHRO_STATE_NEED_FRAME:
        return GST_FLOW_OK;
      case SCHRO_STATE_END_OF_STREAM:
        GST_DEBUG("EOS");
        return GST_FLOW_OK;
      case SCHRO_STATE_HAVE_BUFFER:
        voidptr = NULL;
        encoded_buffer = schro_encoder_pull_full (schro_enc->encoder,
            &presentation_frame, &voidptr);
        frame = voidptr;
        if (encoded_buffer == NULL) {
          GST_DEBUG("encoder_pull returned NULL");
          /* FIXME This shouldn't happen */
          return GST_FLOW_ERROR;
        }

        if (voidptr == NULL) {
          GST_DEBUG("got eos");
          frame = schro_enc->eos_frame;
        }

        if (schro_enc->streamheaders == NULL) {
          int size = GST_READ_UINT32_BE (encoded_buffer->data + 5);
          GstBuffer *buffer;

          buffer = gst_buffer_new_and_alloc (size);
          memcpy (GST_BUFFER_DATA(buffer), encoded_buffer->data, size);

          GST_BUFFER_FLAG_SET (buffer, GST_BUFFER_FLAG_IN_CAPS);
          schro_enc->streamheaders = g_list_append (NULL, buffer);
        }

        if (SCHRO_PARSE_CODE_IS_SEQ_HEADER (encoded_buffer->data[4])) {
          frame->is_sync_point = TRUE;
        }
        
        frame->src_buffer = gst_buffer_new_and_alloc (encoded_buffer->length);
        memcpy (GST_BUFFER_DATA (frame->src_buffer), encoded_buffer->data,
            encoded_buffer->length);
        schro_buffer_unref (encoded_buffer);

        ret = gst_base_video_coder_finish_frame (base_video_coder, frame);

        if (ret != GST_FLOW_OK) {
          GST_DEBUG("pad_push returned %d", ret);
          return ret;
        }
        break;
      case SCHRO_STATE_AGAIN:
        break;
    }
  }
  return GST_FLOW_OK;
}


