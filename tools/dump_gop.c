
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrobitstream.h>
#include <schroedinger/schrounpack.h>
#include <glib.h>
#include <string.h>

#define DIRAC_COMPAT 1

#if 0
/* Used for checking bitstream bugs */
#define MARKER() \
do { \
  g_print("  marker: %d\n", schro_unpack_decode_uint(&unpack)); \
}while(0)
#else
#define MARKER()
#endif

static void fakesink_handoff (GstElement *fakesink, GstBuffer *buffer,
    GstPad *pad, gpointer p_pipeline);
static void event_loop(GstElement *pipeline);

gboolean raw = FALSE;
char *fn = "output.ogg";

static GOptionEntry entries[] = 
{
    { "raw", 'r', 0, G_OPTION_ARG_NONE, &raw, "File is raw Dirac stream", NULL },
    { NULL }
};

int
main (int argc, char *argv[])
{
  GstElement *pipeline;
  GstElement *fakesink;
  GstElement *filesrc;
  GError *error = NULL;
  GOptionContext *context;

  if (!g_thread_supported ()) g_thread_init(NULL);

  context = g_option_context_new ("dump_packets");
  g_option_context_add_main_entries (context, entries, NULL);
  g_option_context_add_group (context, gst_init_get_option_group ());
  g_option_context_parse (context, &argc, &argv, &error);
  g_option_context_free (context);
  if (argc > 1) {
    fn = argv[1];
  }

  gst_init(NULL,NULL);

  if (raw) {
    pipeline = gst_parse_launch("filesrc ! schroparse ! video/x-dirac ! fakesink", NULL);
  } else {
    pipeline = gst_parse_launch("filesrc ! oggdemux ! video/x-dirac ! fakesink", NULL);
  }

  fakesink = gst_bin_get_by_name (GST_BIN(pipeline), "fakesink0");
  g_assert(fakesink != NULL);

  g_object_set (G_OBJECT(fakesink), "signal-handoffs", TRUE, NULL);

  g_signal_connect (G_OBJECT(fakesink), "handoff",
      G_CALLBACK(fakesink_handoff), pipeline);

  filesrc = gst_bin_get_by_name (GST_BIN(pipeline), "filesrc0");
  g_assert(filesrc != NULL);

  g_object_set (G_OBJECT(filesrc), "location", fn, NULL);

  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  event_loop(pipeline);

  gst_element_set_state (pipeline, GST_STATE_NULL);

  return 0;
}

static void
dump_hex (guint8 *data, int length, char *prefix)
{
  int i;
  for(i=0;i<length;i++){
    if ((i&0xf) == 0) {
      g_print("%s0x%04x: ", prefix, i);
    }
    g_print("%02x ", data[i]);
    if ((i&0xf) == 0xf) {
      g_print("\n");
    }
  }
  if ((i&0xf) != 0xf) {
    g_print("\n");
  }
}

static void
fakesink_handoff (GstElement *fakesink, GstBuffer *buffer, GstPad *pad,
    gpointer p_pipeline)
{
  guint8 *data;
  SchroUnpack unpack;
  const char *parse_code;
  int next;
  int prev;
  int size;

  data = GST_BUFFER_DATA(buffer);
  size = GST_BUFFER_SIZE(buffer);

  if (memcmp (data, "KW-DIRAC", 8) == 0) {
    g_print("KW-DIRAC header\n");
    return;
  }
  if (memcmp (data, "BBCD", 4) != 0) {
    g_print("non-Dirac packet\n");
    dump_hex (data, MIN(size, 100), "  ");
    return;
  }

  switch (data[4]) {
    case SCHRO_PARSE_CODE_SEQUENCE_HEADER:
      parse_code = "access unit header";
      break;
    case SCHRO_PARSE_CODE_AUXILIARY_DATA:
      parse_code = "auxiliary data";
      break;
    case SCHRO_PARSE_CODE_INTRA_REF:
      parse_code = "intra ref";
      break;
    case SCHRO_PARSE_CODE_INTRA_NON_REF:
      parse_code = "intra non-ref";
      break;
    case SCHRO_PARSE_CODE_INTER_REF_1:
      parse_code = "inter ref 1";
      break;
    case SCHRO_PARSE_CODE_INTER_REF_2:
      parse_code = "inter ref 2";
      break;
    case SCHRO_PARSE_CODE_INTER_NON_REF_1:
      parse_code = "inter non-ref 1";
      break;
    case SCHRO_PARSE_CODE_INTER_NON_REF_2:
      parse_code = "inter non-ref 2";
      break;
    case SCHRO_PARSE_CODE_END_OF_SEQUENCE:
      parse_code = "end of sequence";
      break;
    case SCHRO_PARSE_CODE_LD_INTRA_REF:
      parse_code = "low-delay intra ref";
      break;
    case SCHRO_PARSE_CODE_LD_INTRA_NON_REF:
      parse_code = "low-delay intra non-ref";
      break;
    case SCHRO_PARSE_CODE_INTRA_REF_NOARITH:
      parse_code = "intra ref noarith";
      break;
    case SCHRO_PARSE_CODE_INTRA_NON_REF_NOARITH:
      parse_code = "intra non-ref noarith";
      break;
    case SCHRO_PARSE_CODE_INTER_REF_1_NOARITH:
      parse_code = "inter ref 1 noarith";
      break;
    case SCHRO_PARSE_CODE_INTER_REF_2_NOARITH:
      parse_code = "inter ref 2 noarith";
      break;
    case SCHRO_PARSE_CODE_INTER_NON_REF_1_NOARITH:
      parse_code = "inter non-ref 1 noarith";
      break;
    case SCHRO_PARSE_CODE_INTER_NON_REF_2_NOARITH:
      parse_code = "inter non-ref 2 noarith";
      break;
    default:
      parse_code = "unknown";
      break;
  }
  schro_unpack_init_with_data (&unpack, data + 5, size - 5, 1);

  next = schro_unpack_decode_bits (&unpack, 32);
  prev = schro_unpack_decode_bits (&unpack, 32);

  if (data[4] == SCHRO_PARSE_CODE_SEQUENCE_HEADER) {
    g_print("AU\n");
  } else if (SCHRO_PARSE_CODE_IS_PICTURE(data[4])) {
    int num_refs = SCHRO_PARSE_CODE_NUM_REFS(data[4]);
    int pic_num;
    int n;
    int i;

    schro_unpack_byte_sync(&unpack);
    pic_num = schro_unpack_decode_bits(&unpack, 32);
    g_print("%5d: %c ", pic_num,
        (SCHRO_PARSE_CODE_IS_REFERENCE(data[4])) ? 'R' : ' ');

    switch (num_refs) {
      case 0:
        g_print(" I             ");
        break;
      case 1:
        g_print(" P %5d       ",
            pic_num + schro_unpack_decode_sint(&unpack));
        break;
      case 2:
        g_print(" B %5d %5d",
            pic_num + schro_unpack_decode_sint(&unpack),
            pic_num + schro_unpack_decode_sint(&unpack));
        break;
    }
    if (SCHRO_PARSE_CODE_IS_REFERENCE(data[4])) {
      n = schro_unpack_decode_uint(&unpack);
      g_print(" retire:");
      if (n == 0) {
        g_print("      ");
      }
      for(i=0;i<n;i++){
        g_print(" %5d",
            pic_num + schro_unpack_decode_sint(&unpack));
      }
    } else {
      g_print("               ");
    }
    g_print("  %d\n", next);
  } else if (data[4] == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
  }

  schro_unpack_byte_sync (&unpack);
}

static void
event_loop(GstElement *pipeline)
{
  GstBus *bus;
  GstMessage *message = NULL;

  bus = gst_element_get_bus (GST_ELEMENT(pipeline));

  while (TRUE) {
    message = gst_bus_poll (bus, GST_MESSAGE_ANY, -1);
    
    switch(message->type) {
      case GST_MESSAGE_EOS:
        gst_message_unref (message);
        return;
      case GST_MESSAGE_WARNING:
      case GST_MESSAGE_ERROR:
        {
          GError *error = NULL;
          gchar *debug;

          gst_message_parse_error (message, &error, &debug);
          gst_object_default_error (GST_MESSAGE_SRC(message), error, debug);
          gst_message_unref(message);
          g_error_free (error);
          g_free (debug);
          return;
        }
      default:
        gst_message_unref(message);
        break;
    }
  }
}

