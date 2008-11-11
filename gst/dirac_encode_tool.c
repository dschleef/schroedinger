
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include "run_pipeline.h"

#include <string.h>

#define GETTEXT_PACKAGE NULL

int verbose;

int use_schro;
int use_dirac_research;
int use_theora;
int use_h264;
int use_h263;

int use_vorbis;
int use_mp2;
int use_aac;
int use_mp3;
int use_raw_audio;

int use_ogg;
int use_mpeg_ts;
int use_quicktime;

int use_deinterlace;
int use_downsample;
int use_HD720;
int use_HD1080;
int use_SD;
int use_SDwide;
int use_SIF;
int use_SIFwide;

/*
 * Notes:
 *
 * ogg/schro:
 *   - error in encode
 *   - seeking causes desync
 * ogg/dirac-research
 *   - encode error
 *   - seeking causes desync
 * ogg/theora
 *   - encode error
 *
 * quicktime/schro
 *   - doesn't play
 * quicktime/dirac-research
 *   - broken
 *
 * quicktime/aac
 *   - ?
 * quicktime/mp3
 *   - ?
 * quicktime/mp2
 *   - fails to negotiate
 *
 * mpeg-ts/schro/mp3
 *   - "clock problem"
 *   - duration doesn't work
 * mpeg-ts/dirac-research
 *   - bad duration
 *
 * mpeg-ts/mp3
 *   - ok
 * mpeg-ts/aac
 *   - not supported
 * mpeg-ts/mp2
 *   - ok
 *  
 */

int have_element (const char *element_name);
void audio_handoff (GstElement *element, GstBuffer *buffer, gpointer user_data);
void video_handoff (GstElement *element, GstBuffer *buffer, gpointer user_data);

static GOptionEntry entries[] = 
{
  { "verbose", 'v', 0, G_OPTION_ARG_NONE, &verbose, "Be verbose", NULL },

  { "schro", 0, 0, G_OPTION_ARG_NONE, &use_schro, "Use Schroedinger encoder", NULL },
  { "dirac-research", 0, 0, G_OPTION_ARG_NONE, &use_dirac_research, "Use Dirac research encoder", NULL },
  { "theora", 0, 0, G_OPTION_ARG_NONE, &use_theora, "Use Theora encoder", NULL },
  { "h264", 0, 0, G_OPTION_ARG_NONE, &use_h264, "Use H.264 video", NULL },
  { "h263", 0, 0, G_OPTION_ARG_NONE, &use_h263, "Use H.263 video", NULL },

  { "ogg", 0, 0, G_OPTION_ARG_NONE, &use_ogg, "Create Ogg file", NULL },
  { "mpeg-ts", 0, 0, G_OPTION_ARG_NONE, &use_mpeg_ts, "Create MPEG TS file", NULL },
  { "quicktime", 0, 0, G_OPTION_ARG_NONE, &use_quicktime, "Create QuickTime file", NULL },

  { "mp2", 0, 0, G_OPTION_ARG_NONE, &use_mp2, "Use MPEG 1 Layer 2 (\"mp2\") audio", NULL },
  { "mp3", 0, 0, G_OPTION_ARG_NONE, &use_mp3, "Use MPEG 1 Layer 3 (\"mp3\") audio", NULL },
  { "aac", 0, 0, G_OPTION_ARG_NONE, &use_aac, "Use MPEG 4 AAC (\"mp4\") audio", NULL },
  { "vorbis", 0, 0, G_OPTION_ARG_NONE, &use_vorbis, "Use Vorbis audio", NULL },
  { "raw-audio", 0, 0, G_OPTION_ARG_NONE, &use_raw_audio, "Use uncompressed audio", NULL },

  { "deinterlace", 0, 0, G_OPTION_ARG_NONE, &use_deinterlace, "Deinterlace video", NULL },
  { "downsample", 0, 0, G_OPTION_ARG_NONE, &use_downsample, "Downsample video", NULL },

  { "HD1080", 0, 0, G_OPTION_ARG_NONE, &use_HD1080, "resize to HD1080", NULL },
  { "HD720", 0, 0, G_OPTION_ARG_NONE, &use_HD720, "resize to HD720", NULL },
  { "SD", 0, 0, G_OPTION_ARG_NONE, &use_SD, "resize to Standard Definition", NULL },
  { "SDwide", 0, 0, G_OPTION_ARG_NONE, &use_SDwide, "resize to wide SD", NULL },
  { "SIF", 0, 0, G_OPTION_ARG_NONE, &use_SIF, "resize to SIF", NULL },
  { "SIFwide", 0, 0, G_OPTION_ARG_NONE, &use_SIFwide, "resize to SIFwide", NULL },

  { NULL }
};

int
main (int argc, char *argv[])
{
  GError *error = NULL;
  GOptionContext *context;
  gchar *input_filename;
  gchar *output_filename;
  GstElement *pipeline;
  GstElement *e;
  gboolean res;
  GString *pipe_desc;
  const gchar *extension;

  if (!g_thread_supported ()) g_thread_init(NULL);

  context = g_option_context_new ("- encode Dirac video");
  g_option_context_add_main_entries (context, entries, GETTEXT_PACKAGE);
  g_option_context_add_group (context, gst_init_get_option_group ());
  if (!g_option_context_parse (context, &argc, &argv, &error)) {
    g_print ("option parsing failed: %s\n", error->message);
    exit (1);
  }

  if (argc < 2) {
    g_print ("expected filename\n");
    exit(1);
  }

  input_filename = argv[1];

  error = NULL;

  pipe_desc = g_string_new ("");

  g_string_append (pipe_desc, "filesrc name=src ! decodebin name=dec ");

  if (!(use_ogg || use_mpeg_ts || use_quicktime)) {
    use_ogg = TRUE;
  }
  if (use_ogg) {
    g_string_append (pipe_desc, "oggmux name=mux ! filesink name=sink ");
    extension = "ogv";
  } else if (use_mpeg_ts) {
    g_string_append (pipe_desc, "mpegtsmux name=mux ! filesink name=sink ");
    extension = "ts";
  } else if (use_quicktime) {
    g_string_append (pipe_desc, "qtmux name=mux ! filesink name=sink ");
    extension = "mov";
  } else {
    g_print("unknown container\n");
    exit (1);
  }
  output_filename = g_strdup_printf ("out.%s", extension);

  if (!(use_schro || use_dirac_research || use_theora || use_h264 || use_h263)) {
    use_schro = TRUE;
  }
  g_string_append (pipe_desc, "dec.! identity name=viden ! queue max-size-bytes=0 max-size-time=0 ! ffmpegcolorspace ! ");

  if (use_deinterlace) {
    g_string_append (pipe_desc, "deinterlace ! ");
  }

  if (use_HD720) {
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=1280,height=720,pixel-aspect-ratio=1/1 ! ");
  } else if (use_HD1080) {
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=1920,height=1080,pixel-aspect-ratio=1/1 ! ");
  } else if (use_SDwide) {
    //g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=704,height=360,pixel-aspect-ratio=10/11 ! ");
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=640,height=360,pixel-aspect-ratio=1/1 ! ");
  } else if (use_SD) {
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=704,height=480,pixel-aspect-ratio=10/11 ! ");
  } else if (use_SIF) {
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=320,height=240,pixel-aspect-ratio=1/1 ! ");
  } else if (use_SIFwide) {
    g_string_append (pipe_desc, "schroscale ! video/x-raw-yuv,width=320,height=180,pixel-aspect-ratio=1/1 ! ");
  } else {
    /* nothing */
  }

  if (use_schro) {
    if (use_quicktime) {
      g_string_append (pipe_desc, "schroenc name=venc ! schroparse !mux. ");
    } else {
      //g_string_append (pipe_desc, "schroenc name=venc !mux. ");
      g_string_append (pipe_desc, "schroenc name=venc ! schroparse !mux. ");
    }
  } else if (use_dirac_research) {
    g_string_append (pipe_desc, "diracenc name=venc ! schroparse !mux. ");
  } else if (use_theora) {
    g_string_append (pipe_desc, "theoraenc name=venc !mux. ");
  } else if (use_h264) {
    g_string_append (pipe_desc, "x264enc name=venc !mux. ");
  } else if (use_h263) {
    g_string_append (pipe_desc, "ffenc_h263 name=venc !mux. ");
  } else {
    g_print("unknown video encoder\n");
    exit (1);
  }

  if (!(use_vorbis || use_mp2 || use_mp3 || use_aac || use_raw_audio)) {
    if (use_ogg) {
      use_vorbis = TRUE;
    } else if (use_quicktime || use_mpeg_ts) {
      if (have_element ("faac")) {
        use_aac = TRUE;
      } else if (have_element ("lame")) {
        use_mp3 = TRUE;
      } else if (have_element ("ffenc_mp2")) {
        use_mp2 = TRUE;
      } else {
        use_raw_audio = TRUE;
      }
    } else {
      g_print ("unable to select audio encoder\n");
    }
  }

  g_string_append (pipe_desc,
      "dec.! identity name=aiden ! queue max-size-bytes=0 max-size-time=0 ! audioconvert ! ");
  if (use_vorbis) {
    g_string_append (pipe_desc,
        "vorbisenc name=aenc !mux. ");
  } else if (use_mp2) {
    g_string_append (pipe_desc,
        "ffenc_mp2 name=aenc !mux. ");
  } else if (use_mp3) {
    g_string_append (pipe_desc,
        "lame name=aenc !mux. ");
  } else if (use_aac) {
    /* use LC profile in AAC */
    g_string_append (pipe_desc,
        "faac profile=2 name=aenc !mux. ");
  } else if (use_raw_audio) {
    g_string_append (pipe_desc,
        "identity name=aenc !mux.audio_00 ");
  } else {
    g_print("unknown audio encoder\n");
    exit (1);
  }

  g_print ("pipeline: %s\n", pipe_desc->str);

  pipeline = (GstElement *) gst_parse_launch (pipe_desc->str, &error);
  g_string_free (pipe_desc, FALSE);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "src");
  g_assert (e != NULL);
  g_object_set (e, "location", input_filename, NULL);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "sink");
  g_assert (e != NULL);
  g_object_set (e, "location", output_filename, NULL);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "aiden");
  g_assert (e != NULL);
  g_object_set (e, "signal-handoffs", TRUE, NULL);
  g_signal_connect (e, "handoff", G_CALLBACK(audio_handoff), NULL);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "viden");
  g_assert (e != NULL);
  g_object_set (e, "signal-handoffs", TRUE, NULL);
  g_signal_connect (e, "handoff", G_CALLBACK(video_handoff), NULL);

  e = gst_bin_get_by_name (GST_BIN(pipeline), "venc");
  g_assert (e != NULL);
  if (use_schro) {
    g_object_set (e, "au-distance", 30, NULL);
    g_object_set (e, "rate-control", 4, NULL);
    g_object_set (e, "quality", 6.0, NULL);
  }

  res = run_pipeline (pipeline);
  if (!res) {
    g_print ("pipeline failed to run\n");
  }

  g_object_unref (pipeline);

  exit(0);
}

int
have_element (const gchar *element_name)
{
  GstPluginFeature *feature;

  feature = gst_default_registry_find_feature (element_name, GST_TYPE_ELEMENT_FACTORY);
  if (feature) {
    g_object_unref (feature);
    return TRUE;
  }
  return FALSE;
}


gboolean have_first_audio_timestamp;
gboolean have_first_video_timestamp;
GstClockTime first_audio_timestamp;
GstClockTime first_video_timestamp;

void
check_first_timestamps (void)
{
  if (have_first_video_timestamp && have_first_audio_timestamp) {
    if (first_video_timestamp != first_audio_timestamp) {
      g_print ("broken first timestamps %lld != %lld\n",
          first_video_timestamp, first_audio_timestamp);
    }
  }
}

void
audio_handoff (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  if (!have_first_audio_timestamp) {
    first_audio_timestamp = GST_BUFFER_TIMESTAMP (buffer);
    have_first_audio_timestamp = TRUE;
    g_print ("first audio timestamp %lld\n", GST_BUFFER_TIMESTAMP (buffer));

    check_first_timestamps ();
  }
}

void
video_handoff (GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  if (!have_first_video_timestamp) {
    first_video_timestamp = GST_BUFFER_TIMESTAMP (buffer);
    have_first_video_timestamp = TRUE;
    g_print ("first video timestamp %lld\n", GST_BUFFER_TIMESTAMP (buffer));

    check_first_timestamps ();
  }
}

