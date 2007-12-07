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
#include <math.h>

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
  GstElement element;

  GstPad *sinkpad;
  GstPad *srcpad;

  /* parameters */
  int level;

  /* video properties */
  int width;
  int height;
  int fps_n, fps_d;
  int par_n, par_d;
  guint64 duration;
  uint32_t fourcc;

  /* segment properties */
  gint64 segment_start;
  gint64 segment_position;

  /* state */
  gboolean got_offset;
  uint64_t granulepos_offset;
  uint64_t granulepos_low;
  uint64_t granulepos_hi;
  gboolean started;

  SchroEncoder *encoder;
  SchroVideoFormat *video_format;
};

struct _GstSchroEncClass
{
  GstElementClass parent_class;
};


enum
{
  LAST_SIGNAL
};

enum
{
  ARG_0
};

static void gst_schro_enc_finalize (GObject *object);
static void gst_schro_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_schro_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_schro_enc_sink_setcaps (GstPad *pad, GstCaps *caps);
static gboolean gst_schro_enc_sink_event (GstPad *pad, GstEvent *event);
static GstFlowReturn gst_schro_enc_chain (GstPad *pad, GstBuffer *buf);
static GstFlowReturn gst_schro_enc_process (GstSchroEnc *schro_enc);
static GstStateChangeReturn gst_schro_enc_change_state (GstElement *element,
    GstStateChange transition);
static const GstQueryType * gst_schro_enc_get_query_types (GstPad *pad);
static gboolean gst_schro_enc_src_query (GstPad *pad, GstQuery *query);

static GstStaticPadTemplate gst_schro_enc_sink_template =
    GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_YUV ("{ I420, YV12, YUY2, UYVY, AYUV }"))
    );

static GstStaticPadTemplate gst_schro_enc_src_template =
    GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-dirac")
    );

GST_BOILERPLATE (GstSchroEnc, gst_schro_enc, GstElement, GST_TYPE_ELEMENT);

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
  int i;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);

  gobject_class->set_property = gst_schro_enc_set_property;
  gobject_class->get_property = gst_schro_enc_get_property;
  gobject_class->finalize = gst_schro_enc_finalize;

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

  gstelement_class->change_state = gst_schro_enc_change_state;
}

static void
gst_schro_enc_init (GstSchroEnc *schro_enc, GstSchroEncClass *klass)
{
  GST_DEBUG ("gst_schro_enc_init");

  schro_enc->encoder = schro_encoder_new ();
  schro_enc->video_format =
    schro_encoder_get_video_format (schro_enc->encoder);

  schro_enc->sinkpad = gst_pad_new_from_static_template (&gst_schro_enc_sink_template, "sink");
  gst_pad_set_chain_function (schro_enc->sinkpad, gst_schro_enc_chain);
  gst_pad_set_event_function (schro_enc->sinkpad, gst_schro_enc_sink_event);
  gst_pad_set_setcaps_function (schro_enc->sinkpad, gst_schro_enc_sink_setcaps);
  //gst_pad_set_query_function (schro_enc->sinkpad, gst_schro_enc_sink_query);
  gst_element_add_pad (GST_ELEMENT(schro_enc), schro_enc->sinkpad);

  schro_enc->srcpad = gst_pad_new_from_static_template (&gst_schro_enc_src_template, "src");
  gst_pad_set_query_type_function (schro_enc->srcpad, gst_schro_enc_get_query_types);
  gst_pad_set_query_function (schro_enc->srcpad, gst_schro_enc_src_query);
  gst_element_add_pad (GST_ELEMENT(schro_enc), schro_enc->srcpad);
}

static gboolean
gst_schro_enc_sink_setcaps (GstPad *pad, GstCaps *caps)
{
  GstStructure *structure;
  GstSchroEnc *schro_enc = GST_SCHRO_ENC (gst_pad_get_parent (pad));

  structure = gst_caps_get_structure (caps, 0);

  gst_structure_get_fourcc (structure, "format", &schro_enc->fourcc);
  gst_structure_get_int (structure, "width", &schro_enc->width);
  gst_structure_get_int (structure, "height", &schro_enc->height);
  gst_structure_get_fraction (structure, "framerate", &schro_enc->fps_n,
      &schro_enc->fps_d);
  schro_enc->par_n = 1;
  schro_enc->par_d = 1;
  gst_structure_get_fraction (structure, "pixel-aspect-ratio",
      &schro_enc->par_n, &schro_enc->par_d);

  schro_video_format_set_std_video_format (schro_enc->video_format,
      SCHRO_VIDEO_FORMAT_CUSTOM);

  switch (schro_enc->fourcc) {
    case GST_MAKE_FOURCC('I','4','2','0'):
    case GST_MAKE_FOURCC('Y','V','1','2'):
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_420;
      break;
    case GST_MAKE_FOURCC('Y','U','Y','2'):
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_422;
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      schro_enc->video_format->chroma_format = SCHRO_CHROMA_444;
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

  schro_encoder_set_video_format (schro_enc->encoder, schro_enc->video_format);

#if 0
  schro_encoder_use_perceptual_weighting (schro_enc->encoder,
      SCHRO_ENCODER_PERCEPTUAL_MOO, 3.0);
#endif

  schro_enc->duration = gst_util_uint64_scale_int (GST_SECOND,
          schro_enc->fps_d, schro_enc->fps_n);

  gst_object_unref (GST_OBJECT(schro_enc));

  return TRUE;
}

static void
gst_schro_enc_finalize (GObject *object)
{
  GstSchroEnc *schro_enc;

  g_return_if_fail (GST_IS_SCHRO_ENC (object));
  schro_enc = GST_SCHRO_ENC (object);

  if (schro_enc->encoder) {
    schro_encoder_free (schro_enc->encoder);
    schro_enc->encoder = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
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

static gboolean
gst_schro_enc_sink_event (GstPad *pad, GstEvent *event)
{
  GstSchroEnc *schro_enc;
  gboolean ret;

  schro_enc = GST_SCHRO_ENC (GST_PAD_PARENT (pad));

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_EOS:
      schro_encoder_end_of_stream (schro_enc->encoder);
      gst_schro_enc_process (schro_enc);
      ret = gst_pad_push_event (schro_enc->srcpad, event);
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

        GST_DEBUG("new segment %lld %lld", start, position);
        schro_enc->segment_start = start;
        schro_enc->segment_position = position;

        ret = gst_pad_push_event (schro_enc->srcpad, event);
      }
      break;
    default:
      ret = gst_pad_push_event (schro_enc->srcpad, event);
      break;
  }

  return ret;
}

static void
gst_schro_frame_free (SchroFrame *frame, void *priv)
{
  gst_buffer_unref (GST_BUFFER(priv));
}

static SchroFrame *
gst_schro_buffer_wrap (GstSchroEnc *schro_enc, GstBuffer *buf,
    int width, int height)
{
  SchroFrame *frame;

  switch (schro_enc->fourcc) {
    case GST_MAKE_FOURCC('I','4','2','0'):
      frame = schro_frame_new_from_data_I420 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_MAKE_FOURCC('Y','V','1','2'):
      frame = schro_frame_new_from_data_YV12 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_MAKE_FOURCC('Y','U','Y','2'):
      frame = schro_frame_new_from_data_YUY2 (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_MAKE_FOURCC('U','Y','V','Y'):
      frame = schro_frame_new_from_data_UYVY (GST_BUFFER_DATA (buf), width, height);
      break;
    case GST_MAKE_FOURCC('A','Y','U','V'):
      frame = schro_frame_new_from_data_AYUV (GST_BUFFER_DATA (buf), width, height);
      break;
    default:
      g_assert_not_reached();
  }
  schro_frame_set_free_callback (frame, gst_schro_frame_free, buf);

  return frame;
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
gst_schro_enc_get_query_types (GstPad *pad)
{
  static const GstQueryType query_types[] = {
    //GST_QUERY_POSITION,
    //GST_QUERY_DURATION,
    GST_QUERY_CONVERT,
    0
  };

  return query_types;
}

#if 0
static gboolean
gst_schro_enc_sink_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{
  gboolean res = TRUE;
  GstSchroEnc *enc;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  }

  enc = GST_SCHRO_ENC (gst_pad_get_parent (pad));

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
gst_schro_enc_src_convert (GstPad *pad,
    GstFormat src_format, gint64 src_value,
    GstFormat * dest_format, gint64 *dest_value)
{ 
  gboolean res = TRUE;
  GstSchroEnc *enc;

  if (src_format == *dest_format) {
    *dest_value = src_value;
    return TRUE;
  } 

  enc = GST_SCHRO_ENC (gst_pad_get_parent (pad));

  /* FIXME: check if we are in a encoding state */

  switch (src_format) {
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
    default:
      res = FALSE; 
      break;
  }

  gst_object_unref (enc);

  return res;
}

static gboolean
gst_schro_enc_src_query (GstPad *pad, GstQuery *query)
{
  GstSchroEnc *enc;
  gboolean res;

  enc = GST_SCHRO_ENC (gst_pad_get_parent (pad));

  switch GST_QUERY_TYPE (query) {
    case GST_QUERY_CONVERT:
      {
        GstFormat src_fmt, dest_fmt;
        gint64 src_val, dest_val;

        gst_query_parse_convert (query, &src_fmt, &src_val, &dest_fmt, &dest_val);
        res = gst_schro_enc_src_convert (pad, src_fmt, src_val, &dest_fmt,
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

static GstFlowReturn
gst_schro_enc_chain (GstPad *pad, GstBuffer *buf)
{
  GstSchroEnc *schro_enc;
  SchroFrame *frame;
  GstFlowReturn ret;

  schro_enc = GST_SCHRO_ENC (gst_pad_get_parent (pad));

  if (GST_BUFFER_TIMESTAMP (buf) < schro_enc->segment_start) {
    GST_DEBUG("dropping early buffer");
    return GST_FLOW_OK;
  }
  if (!schro_enc->got_offset) {
    schro_enc->granulepos_offset =
      gst_util_uint64_scale (GST_BUFFER_TIMESTAMP(buf), schro_enc->fps_n,
          GST_SECOND * schro_enc->fps_d);

    GST_DEBUG("using granulepos offset %lld", schro_enc->granulepos_offset);
    schro_enc->granulepos_hi = 0;
    schro_enc->got_offset = TRUE;
  }
  if (!schro_enc->started) {
    schro_encoder_start (schro_enc->encoder);
    schro_enc->started = TRUE;
  }

  frame = gst_schro_buffer_wrap (schro_enc, buf, schro_enc->width, schro_enc->height);

  GST_DEBUG ("pushing frame");
  schro_encoder_push_frame (schro_enc->encoder, frame);

  ret = gst_schro_enc_process (schro_enc);

  gst_object_unref (schro_enc);

  return ret;
}

static GstFlowReturn
gst_schro_enc_process (GstSchroEnc *schro_enc)
{
  SchroBuffer *encoded_buffer;
  GstBuffer *outbuf;
  GstFlowReturn ret;
  int presentation_frame;

  while (1) {
    switch (schro_encoder_wait (schro_enc->encoder)) {
      case SCHRO_STATE_NEED_FRAME:
      case SCHRO_STATE_END_OF_STREAM:
        return GST_FLOW_OK;
      case SCHRO_STATE_HAVE_BUFFER:
        encoded_buffer = schro_encoder_pull (schro_enc->encoder,
            &presentation_frame);
        if (encoded_buffer == NULL) {
          GST_ERROR("encoder_pull returned NULL");
          /* FIXME This shouldn't happen */
          return GST_FLOW_ERROR;
        }

        if (schro_decoder_is_access_unit (encoded_buffer)) {
          schro_enc->granulepos_hi = schro_enc->granulepos_offset +
            presentation_frame + 1;
        }

        schro_enc->granulepos_low = schro_enc->granulepos_offset +
          presentation_frame + 1 - schro_enc->granulepos_hi;

        outbuf = gst_buffer_new_and_alloc (encoded_buffer->length);
        memcpy (GST_BUFFER_DATA (outbuf), encoded_buffer->data,
            encoded_buffer->length);
        gst_buffer_set_caps (outbuf, gst_pad_get_caps(schro_enc->srcpad));

        GST_BUFFER_OFFSET_END (outbuf) =
          (schro_enc->granulepos_hi<<30) + schro_enc->granulepos_low;
        GST_BUFFER_OFFSET (outbuf) = gst_util_uint64_scale (
            (schro_enc->granulepos_hi + schro_enc->granulepos_low),
            schro_enc->fps_d * GST_SECOND, schro_enc->fps_n);

        GST_BUFFER_TIMESTAMP (outbuf) = gst_util_uint64_scale (
            (schro_enc->granulepos_hi + schro_enc->granulepos_low),
            schro_enc->fps_d * GST_SECOND, schro_enc->fps_n);
        if (schro_decoder_is_picture (encoded_buffer)) {
          GST_BUFFER_DURATION (outbuf) = schro_enc->duration;
        } else {
          GST_BUFFER_DURATION (outbuf) = 0;
        }

        GST_INFO("size %d offset %lld granulepos %llu:%llu timestamp %lld duration %lld",
            GST_BUFFER_SIZE (outbuf),
            GST_BUFFER_OFFSET (outbuf),
            GST_BUFFER_OFFSET_END (outbuf)>>30,
            GST_BUFFER_OFFSET_END (outbuf)&((1<<30) - 1),
            GST_BUFFER_TIMESTAMP (outbuf),
            GST_BUFFER_DURATION (outbuf));

        if (schro_decoder_is_intra (encoded_buffer)) {
          GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DELTA_UNIT);
        }

        schro_buffer_unref (encoded_buffer);
        
        ret = gst_pad_push (schro_enc->srcpad, outbuf);

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

static GstStateChangeReturn
gst_schro_enc_change_state (GstElement *element, GstStateChange transition)
{
  GstSchroEnc *schro_enc;
  GstStateChangeReturn ret;

  schro_enc = GST_SCHRO_ENC (element);

  switch (transition) {
    default:
      break;
  }

  ret = parent_class->change_state (element, transition);

  switch (transition) {
    default:
      break;
  }

  return ret;
}

