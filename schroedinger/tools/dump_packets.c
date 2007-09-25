
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrobitstream.h>
#include <glib.h>
#include <string.h>

#define DIRAC_COMPAT 1


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
  SchroBits *bits;
  SchroBuffer *buf;
  const char *parse_code;
  int next;
  int prev;

  data = GST_BUFFER_DATA(buffer);

  if (memcmp (data, "KW-DIRAC", 8) == 0) {
    g_print("KW-DIRAC header\n");
    return;
  }
  if (memcmp (data, "BBCD", 4) != 0) {
    g_print("non-Dirac packet\n");
    dump_hex (data, MIN(GST_BUFFER_SIZE(buffer), 100), "  ");
    return;
  }

  switch (data[4]) {
    case SCHRO_PARSE_CODE_ACCESS_UNIT:
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
    case SCHRO_PARSE_CODE_END_SEQUENCE:
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
  g_print("Parse code: %s (0x%02x)\n", parse_code, data[4]);
  
  buf = schro_buffer_new_with_data (data + 5, GST_BUFFER_SIZE(buffer) - 5);
  bits = schro_bits_new();
  schro_bits_decode_init (bits, buf);

  next = schro_bits_decode_bits (bits, 32);
  prev = schro_bits_decode_bits (bits, 32);

  g_print("  offset to next: %d\n", next);
  g_print("  offset to prev: %d\n", prev);

  if (data[4] == SCHRO_PARSE_CODE_ACCESS_UNIT) {
    int bit;

    /* parse parameters */
    g_print("  version.major: %d\n", schro_bits_decode_uint(bits));
    g_print("  version.minor: %d\n", schro_bits_decode_uint(bits));
    g_print("  profile: %d\n", schro_bits_decode_uint(bits));
    g_print("  level: %d\n", schro_bits_decode_uint(bits));

    /* sequence parameters */
    g_print("  video_format: %d\n", schro_bits_decode_uint(bits));

    bit = schro_bits_decode_bit(bits);
    g_print("  custom dimensions flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      g_print("    width: %d\n", schro_bits_decode_uint(bits));
      g_print("    height: %d\n", schro_bits_decode_uint(bits));
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  chroma format flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      g_print("    chroma format: %d\n", schro_bits_decode_uint(bits));
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  video depth flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      g_print("    video depth: %d\n", schro_bits_decode_uint(bits));
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  scan format flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      bit = schro_bits_decode_bit(bits);
      g_print("    interlaced source: %s\n", bit ? "yes" : "no");
      if (bit) {
        bit = schro_bits_decode_bit(bits);
        g_print("      field dominance flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          bit = schro_bits_decode_bit(bits);
          g_print("      top field first: %s\n", bit ? "yes" : "no");
        }
        bit = schro_bits_decode_bit(bits);
        g_print("      field interleaving flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          bit = schro_bits_decode_bit(bits);
          g_print("      sequential fields: %s\n", bit ? "yes" : "no");
        }
      }
    }
    
    bit = schro_bits_decode_bit(bits);
    g_print("  frame rate flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_bits_decode_uint(bits);
      g_print("    frame rate index: %d\n", index);
      if (index == 0) {
        g_print("      frame rate numerator: %d\n",
            schro_bits_decode_uint(bits));
        g_print("      frame rate demoninator: %d\n",
            schro_bits_decode_uint(bits));
      }
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  aspect ratio flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_bits_decode_uint(bits);
      g_print("    aspect ratio index: %d\n", index);
      if (index == 0) {
        g_print("      aspect ratio numerator: %d\n",
            schro_bits_decode_uint(bits));
        g_print("      aspect ratio demoninator: %d\n",
            schro_bits_decode_uint(bits));
      }
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  clean area flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      g_print("    clean width: %d\n", schro_bits_decode_uint(bits));
      g_print("    clean height: %d\n", schro_bits_decode_uint(bits));
      g_print("    left offset: %d\n", schro_bits_decode_uint(bits));
      g_print("    top offset: %d\n", schro_bits_decode_uint(bits));
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  signal range flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_bits_decode_uint(bits);
      g_print("    signal range index: %d\n", index);
      if (index == 0) {
        g_print("      luma offset: %d\n", schro_bits_decode_uint(bits));
        g_print("      luma excursion: %d\n", schro_bits_decode_uint(bits));
        g_print("      chroma offset: %d\n", schro_bits_decode_uint(bits));
        g_print("      chroma excursion: %d\n", schro_bits_decode_uint(bits));
      }
    }

    bit = schro_bits_decode_bit(bits);
    g_print("  colour spec flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_bits_decode_uint(bits);
      g_print("    colour spec index: %d\n", index);
      if (index == 0) {
        bit = schro_bits_decode_bit(bits);
        g_print("      colour primaries flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          g_print("        colour primaries: %d\n",
              schro_bits_decode_uint(bits));
        }
        bit = schro_bits_decode_bit(bits);
        g_print("      colour matrix flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          g_print("        colour matrix: %d\n", schro_bits_decode_uint(bits));
        }
        bit = schro_bits_decode_bit(bits);
        g_print("      transfer function flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          g_print("        transfer function: %d\n",
              schro_bits_decode_uint(bits));
        }
      }
    }
  } else if (SCHRO_PARSE_CODE_IS_PICTURE(data[4])) {
    int num_refs = SCHRO_PARSE_CODE_NUM_REFS(data[4]);
    int bit;
    int n;
    int i;
    int lowdelay = SCHRO_PARSE_CODE_IS_LOW_DELAY(data[4]);

    g_print("  num refs: %d\n", num_refs);

    schro_bits_sync(bits);
    g_print("  picture_number: %u\n", schro_bits_decode_bits(bits, 32));
    if (num_refs > 0) {
      g_print("  ref1_offset: %d\n", schro_bits_decode_sint(bits));
    }
    if (num_refs > 1) {
      g_print("  ref2_offset: %d\n", schro_bits_decode_sint(bits));
    }
    if (!lowdelay) {
      n = schro_bits_decode_uint(bits);
      g_print("  n retire: %d\n", n);
      for(i=0;i<n;i++){
        g_print("    %d: %d\n", i, schro_bits_decode_sint(bits));
      }
    }

    if (num_refs > 0) {
      schro_bits_sync(bits);
      bit = schro_bits_decode_bit(bits);
      g_print("  block parameters flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        int index = schro_bits_decode_uint(bits);
        g_print("    block parameters index: %d\n", index);
        if (index == 0) {
          g_print("    luma block width: %d\n", schro_bits_decode_uint(bits));
          g_print("    luma block height: %d\n", schro_bits_decode_uint(bits));
          g_print("    horiz luma block sep: %d\n", schro_bits_decode_uint(bits));
          g_print("    vert luma block sep: %d\n", schro_bits_decode_uint(bits));
        }
      }

      bit = schro_bits_decode_bit(bits);
      g_print("  motion vector precision flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        g_print("  motion vector precision bits: %d\n",
            schro_bits_decode_uint(bits));
      }
      
      bit = schro_bits_decode_bit(bits);
      g_print("  using global motion flag: %s\n", bit ? "yes" : "no");

      if (bit) {
        for(i=0;i<num_refs;i++){
          g_print("    global motion ref%d:\n", i+1);
          bit = schro_bits_decode_bit(bits);
          g_print("      non-zero pan/tilt flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("        pan %d\n", schro_bits_decode_sint(bits));
            g_print("        tilt %d\n", schro_bits_decode_sint(bits));
          }
          bit = schro_bits_decode_bit(bits);
          g_print("      non-zero zoom rot shear flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("       exponent %d\n", schro_bits_decode_uint(bits));
            g_print("       A11 %d\n", schro_bits_decode_sint(bits));
            g_print("       A12 %d\n", schro_bits_decode_sint(bits));
            g_print("       A21 %d\n", schro_bits_decode_sint(bits));
            g_print("       A22 %d\n", schro_bits_decode_sint(bits));
          }
          bit = schro_bits_decode_bit(bits);
          g_print("     non-zero perspective flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("       exponent %d\n", schro_bits_decode_uint(bits));
            g_print("       perspective_x %d\n", schro_bits_decode_sint(bits));
            g_print("       perspective_y %d\n", schro_bits_decode_sint(bits));
          }
        }
      }

      bit = schro_bits_decode_bit(bits);
      g_print("  picture prediction mode flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        g_print("  picture prediction mode: %d\n",
            schro_bits_decode_uint(bits));
      }
      bit = schro_bits_decode_bit(bits);
      g_print("  non-default weights flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        g_print("  picture weight precision: %d\n",
            schro_bits_decode_uint(bits));
        for(i=0;i<num_refs;i++){
          g_print("  picture weight ref%d: %d\n", i+1,
              schro_bits_decode_sint(bits));
        }
      }

      schro_bits_sync(bits);
      n = schro_bits_decode_uint(bits);
      g_print("  superblock split data length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      n = schro_bits_decode_uint(bits);
      g_print("  prediction modes data length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      n = schro_bits_decode_uint(bits);
      g_print("  vector data (ref1,x) length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      n = schro_bits_decode_uint(bits);
      g_print("  vector data (ref1,y) length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      if (num_refs>1) {
        n = schro_bits_decode_uint(bits);
        g_print("  vector data (ref2,x) length: %d\n", n);
        schro_bits_sync (bits);
        schro_bits_skip (bits, n);

        n = schro_bits_decode_uint(bits);
        g_print("  vector data (ref2,y) length: %d\n", n);
        schro_bits_sync (bits);
        schro_bits_skip (bits, n);
      }

      n = schro_bits_decode_uint(bits);
      g_print("  DC data (y) length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      n = schro_bits_decode_uint(bits);
      g_print("  DC data (u) length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

      n = schro_bits_decode_uint(bits);
      g_print("  DC data (v) length: %d\n", n);
      schro_bits_sync (bits);
      schro_bits_skip (bits, n);

    }

    schro_bits_sync (bits);
    if (num_refs == 0) {
      bit = 0;
    } else {
      bit = schro_bits_decode_bit (bits);
      g_print("  zero residual: %s\n", bit ? "yes" : "no");
    }
    if (!bit) {
      int depth;
      int j;

      bit = schro_bits_decode_bit (bits);
      g_print("  non-default wavelet flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        g_print("    wavelet index: %d\n", schro_bits_decode_uint(bits));
      }

      bit = schro_bits_decode_bit (bits);
      g_print("  wavelet depth flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        depth = schro_bits_decode_uint(bits);
        g_print("    transform depth: %d\n", depth);
      } else {
        /* FIXME */
        depth = 4;
      }

      if (!lowdelay) {
        bit = schro_bits_decode_bit (bits);
        g_print("  spatial partition flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          bit = schro_bits_decode_bit (bits);
          g_print("    non-default partition flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            for(i=0;i<depth+1;i++){
              g_print("      number of codeblocks depth=%d\n", i);
              g_print("        horizontal codeblocks: %d\n",
                  schro_bits_decode_uint(bits));
              g_print("        vertical codeblocks: %d\n",
                  schro_bits_decode_uint(bits));
            }
          }
          g_print("    codeblock mode index: %d\n", schro_bits_decode_uint(bits));
        }

        schro_bits_sync (bits);
        for(j=0;j<3;j++){
          g_print("  component %d:\n",j);
          g_print("    comp subband  length  quantiser_index\n");
          for(i=0;i<1+depth*3;i++){
            int length;

            if(bits->error) {
              g_print("    PAST END\n");
              continue;
            }

            if (i!=0) schro_bits_sync(bits);
            length = schro_bits_decode_uint(bits);
            if (length > 0) {
              g_print("    %4d %4d:   %6d    %3d\n", j, i, length,
                  schro_bits_decode_uint(bits));
              schro_bits_sync(bits);
              schro_bits_skip (bits, length);
            } else {
              g_print("    %4d %4d:   %6d\n", j, i, length);
            }
          }
        }
      } else {
        int slice_width_exp;
        int slice_height_exp;
        int slice_bytes_numerator;
        int slice_bytes_denominator;

        slice_width_exp = schro_bits_decode_uint(bits);
        slice_height_exp = schro_bits_decode_uint(bits);
        slice_bytes_numerator = schro_bits_decode_uint(bits);
        slice_bytes_denominator = schro_bits_decode_uint(bits);

        g_print("  slice_width_exp: %d\n", slice_width_exp);
        g_print("  slice_height_exp: %d\n", slice_height_exp);
        g_print("  slice_bytes_numerator: %d\n", slice_bytes_numerator);
        g_print("  slice_bytes_denominator: %d\n", slice_bytes_denominator);

        bit = schro_bits_decode_bit (bits);
        g_print("  encode_quant_matrix: %s\n", bit ? "yes" : "no");
        if (bit) {
          for(i=0;i<1+depth*3;i++){
            g_print("    %2d: %d\n", i, schro_bits_decode_uint(bits));
          }
        }

        bit = schro_bits_decode_bit (bits);
        g_print("  encode_quant_offsets: %s\n", bit ? "yes" : "no");
        if (bit) {
          g_print("    luma_offset: %d\n", schro_bits_decode_sint(bits));
          g_print("    chroma1_offset: %d\n", schro_bits_decode_sint(bits));
          g_print("    chroma2_offset: %d\n", schro_bits_decode_sint(bits));
        }

        schro_bits_sync (bits);
      }
    }
  } else if (data[4] == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
    int length = next - 14;
    g_print("  code: %d\n", data[13]);
    g_print("  string: %.*s\n", length, data + 14);

    schro_bits_skip (bits, 4 + 4 + length);
  }

  schro_bits_sync (bits);

  g_print("offset %d\n", schro_bits_get_offset (bits));
  dump_hex (bits->buffer->data + schro_bits_get_offset (bits),
      MIN(bits->buffer->length - schro_bits_get_offset (bits), 100), "  ");

  schro_bits_free (bits);
  schro_buffer_unref (buf);
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

