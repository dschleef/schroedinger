
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <schroedinger/schro.h>
#include <glib.h>
#include <string.h>



static void fakesink_handoff (GstElement *fakesink, GstBuffer *buffer,
    GstPad *pad, gpointer p_pipeline);
static void event_loop(GstElement *pipeline);

int
main (int argc, char *argv[])
{
  GstElement *pipeline;
  GstElement *fakesink;
  GstElement *filesrc;

  gst_init(NULL,NULL);

  pipeline = gst_parse_launch("filesrc ! oggdemux ! video/x-dirac ! fakesink", NULL);

  fakesink = gst_bin_get_by_name (GST_BIN(pipeline), "fakesink0");
  g_assert(fakesink != NULL);

  g_object_set (G_OBJECT(fakesink), "signal-handoffs", TRUE, NULL);

  g_signal_connect (G_OBJECT(fakesink), "handoff",
      G_CALLBACK(fakesink_handoff), pipeline);

  filesrc = gst_bin_get_by_name (GST_BIN(pipeline), "filesrc0");
  g_assert(filesrc != NULL);

  g_object_set (G_OBJECT(filesrc), "location", "../output.ogg", NULL);

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

  data = GST_BUFFER_DATA(buffer);

  if (memcmp (data, "KW-DIRAC", 7) == 0) {
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
    default:
      parse_code = "unknown";
      break;
  }
  g_print("Parse code: %s (0x%02x)\n", parse_code, data[4]);
  
  buf = schro_buffer_new_with_data (data + 5, GST_BUFFER_SIZE(buffer) - 5);
  bits = schro_bits_new();
  schro_bits_decode_init (bits, buf);

  {
    int next;
    int prev;

    next = schro_bits_decode_bits (bits, 24);
    prev = schro_bits_decode_bits (bits, 24);

    g_print("  offset to next: %d\n", next);
    g_print("  offset to prev: %d\n", prev);
  }

  if (data[4] == SCHRO_PARSE_CODE_ACCESS_UNIT) {
    int bit;

    g_print("  au_picture_number: %u\n", schro_bits_decode_bits(bits, 32));
    g_print("  version.major: %d\n", schro_bits_decode_uint(bits));
    g_print("  version.minor: %d\n", schro_bits_decode_uint(bits));
    g_print("  profile: %d\n", schro_bits_decode_uint(bits));
    g_print("  level: %d\n", schro_bits_decode_uint(bits));

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

    g_print("  num refs: %d\n", num_refs);

    schro_bits_sync(bits);
    g_print("  picture_number: %u\n", schro_bits_decode_bits(bits, 32));
    if (num_refs > 0) {
      g_print("  ref1_offset: %d\n", schro_bits_decode_sint(bits));
    }
    if (num_refs > 1) {
      g_print("  ref2_offset: %d\n", schro_bits_decode_sint(bits));
    }
    n = schro_bits_decode_uint(bits);
    g_print("  n retire: %d\n", n);
    for(i=0;i<n;i++){
      g_print("    %d: %d\n", i, schro_bits_decode_sint(bits));
    }

    if (num_refs > 1) {
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
        bit = schro_bits_decode_bit(bits);
        g_print("  global motion only: %s\n", bit ? "yes" : "no");

        for(i=0;i<num_refs;i++){
          g_print("  global motion ref%d:\n", i+1);
          bit = schro_bits_decode_bit(bits);
          g_print("  non-zero pan/tilt flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("    pan %d\n", schro_bits_decode_sint(bits));
            g_print("    tilt %d\n", schro_bits_decode_sint(bits));
          }
          bit = schro_bits_decode_bit(bits);
          g_print("  non-zero zoom rot shear flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("    exponent %d\n", schro_bits_decode_uint(bits));
            g_print("    A11 %d\n", schro_bits_decode_sint(bits));
            g_print("    A12 %d\n", schro_bits_decode_sint(bits));
            g_print("    A21 %d\n", schro_bits_decode_sint(bits));
            g_print("    A22 %d\n", schro_bits_decode_sint(bits));
          }
          bit = schro_bits_decode_bit(bits);
          g_print("  non-zero perspective flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            g_print("    exponent %d\n", schro_bits_decode_uint(bits));
            g_print("    perspective_x %d\n", schro_bits_decode_sint(bits));
            g_print("    perspective_y %d\n", schro_bits_decode_sint(bits));
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
              schro_bits_decode_uint(bits));
        }
      }

      schro_bits_sync (bits);
      n = schro_bits_decode_uint(bits);
      g_print("  block data length: %d\n", n);
      schro_bits_sync (bits);
      bits->offset += n*8;
    }

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

      for(j=0;j<3;j++){
        g_print("  component %d:\n",j);
        for(i=0;i<1+depth*3;i++){
          int length;

          if(bits->offset > bits->buffer->length * 8) {
            g_print("    PAST END\n");
            continue;
          }

          schro_bits_sync(bits);
          g_print("    subband %d:\n", i);
          length = schro_bits_decode_uint(bits);
          g_print("      length: %d\n", length);
          if (length) {
            g_print("      quantiser index: %d\n", schro_bits_decode_uint(bits));
          }
          schro_bits_sync(bits);
          bits->offset += length * 8;
        }
      }
    }
  }

  schro_bits_sync (bits);

  g_print("offset %d\n", bits->offset);
  dump_hex (bits->buffer->data + bits->offset/8,
      MIN(bits->buffer->length - bits->offset/8, 100), "  ");

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

