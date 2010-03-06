
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrobitstream.h>
#include <schroedinger/schrounpack.h>
#include <string.h>
#include <stdio.h>

#if 0
/* Used for checking bitstream bugs */
#define MARKER() \
do { \
  g_print("  marker: %d\n", schro_unpack_decode_uint(&unpack)); \
}while(0)
#else
#define MARKER()
#endif

static void handle_packet(unsigned char *data, int size);

const char *fn = "output.drc";

int
main (int argc, char *argv[])
{
  FILE *file;

  if (argc > 1) {
    fn = argv[1];
  }

  file = fopen (fn, "r");
  if (file == NULL) {
    printf("cannot open %s\n", fn);
    return 1;
  }

  while (1) {
    unsigned char *packet;
    unsigned char header[13];
    int n;
    int size;

    n = fread (header, 1, 13, file);
    if (n == 0) {
      return 0;
    }
    if (n < 13) {
      printf("truncated header\n");
      return 1;
    }

    if (header[0] != 'B' || header[1] != 'B' || header[2] != 'C' ||
        header[3] != 'D') {
      printf("expected BBCD header\n");
      return 1;
    }

    size = (header[5]<<24) | (header[6]<<16) | (header[7]<<8) | (header[8]);
    if (size == 0) {
      size = 13;
    }
    if (size < 13) {
      printf("packet too small (%d < 13)\n", size);
      return 1;
    }
    if (size > 16*1024*1024) {
      printf("packet too large? (%d > 16777216)\n", size);
      return 1;
    }

    packet = malloc (size);
    memcpy (packet, header, 13);
    n = fread (packet + 13, 1, size - 13, file);
    if (n < size - 13) {
      printf("truncated packet (%d < %d)\n", n, size-13);
      exit(1);
    }

    handle_packet (packet, size);
        
    free(packet);
  }

  return 0;
}

static void
dump_hex (const unsigned char *data, int length, const char *prefix)
{
  int i;
  for(i=0;i<length;i++){
    if ((i&0xf) == 0) {
      printf("%s0x%04x: ", prefix, i);
    }
    printf("%02x ", data[i]);
    if ((i&0xf) == 0xf) {
      printf("\n");
    }
  }
  if ((i&0xf) != 0xf) {
    printf("\n");
  }
}

static void
handle_packet(unsigned char *data, int size)
{
  SchroUnpack unpack;
  const char *parse_code;
  int next;
  int prev;

  if (memcmp (data, "KW-DIRAC", 8) == 0) {
    printf("KW-DIRAC header\n");
    return;
  }
  if (memcmp (data, "BBCD", 4) != 0) {
    printf("non-Dirac packet\n");
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
    case SCHRO_PARSE_CODE_PADDING:
      parse_code = "padding";
      break;
    default:
      parse_code = "unknown";
      break;
  }
  printf("Parse code: %s (0x%02x)\n", parse_code, data[4]);
  
  schro_unpack_init_with_data (&unpack, data + 5, size - 5, 1);

  next = schro_unpack_decode_bits (&unpack, 32);
  prev = schro_unpack_decode_bits (&unpack, 32);

  printf("  offset to next: %d\n", next);
  printf("  offset to prev: %d\n", prev);

  if (data[4] == SCHRO_PARSE_CODE_SEQUENCE_HEADER) {
    int bit;

    /* parse parameters */
    printf("  version.major: %d\n", schro_unpack_decode_uint(&unpack));
    printf("  version.minor: %d\n", schro_unpack_decode_uint(&unpack));
    printf("  profile: %d\n", schro_unpack_decode_uint(&unpack));
    printf("  level: %d\n", schro_unpack_decode_uint(&unpack));

    /* sequence parameters */
    printf("  video_format: %d\n", schro_unpack_decode_uint(&unpack));

    bit = schro_unpack_decode_bit(&unpack);
    printf("  custom dimensions flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      printf("    width: %d\n", schro_unpack_decode_uint(&unpack));
      printf("    height: %d\n", schro_unpack_decode_uint(&unpack));
    }

    bit = schro_unpack_decode_bit(&unpack);
    printf("  chroma format flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      printf("    chroma format: %d\n", schro_unpack_decode_uint(&unpack));
    }

    bit = schro_unpack_decode_bit(&unpack);
    printf("  custom_scan_format_flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      printf("    source sampling: %d\n", schro_unpack_decode_uint(&unpack));
    }
    
    MARKER();

    bit = schro_unpack_decode_bit(&unpack);
    printf("  frame rate flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_unpack_decode_uint(&unpack);
      printf("    frame rate index: %d\n", index);
      if (index == 0) {
        printf("      frame rate numerator: %d\n",
            schro_unpack_decode_uint(&unpack));
        printf("      frame rate denominator: %d\n",
            schro_unpack_decode_uint(&unpack));
      }
    }

    MARKER();

    bit = schro_unpack_decode_bit(&unpack);
    printf("  aspect ratio flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_unpack_decode_uint(&unpack);
      printf("    aspect ratio index: %d\n", index);
      if (index == 0) {
        printf("      aspect ratio numerator: %d\n",
            schro_unpack_decode_uint(&unpack));
        printf("      aspect ratio denominator: %d\n",
            schro_unpack_decode_uint(&unpack));
      }
    }

    MARKER();

    bit = schro_unpack_decode_bit(&unpack);
    printf("  clean area flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      printf("    clean width: %d\n", schro_unpack_decode_uint(&unpack));
      printf("    clean height: %d\n", schro_unpack_decode_uint(&unpack));
      printf("    left offset: %d\n", schro_unpack_decode_uint(&unpack));
      printf("    top offset: %d\n", schro_unpack_decode_uint(&unpack));
    }

    MARKER();

    bit = schro_unpack_decode_bit(&unpack);
    printf("  signal range flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_unpack_decode_uint(&unpack);
      printf("    signal range index: %d\n", index);
      if (index == 0) {
        printf("      luma offset: %d\n", schro_unpack_decode_uint(&unpack));
        printf("      luma excursion: %d\n", schro_unpack_decode_uint(&unpack));
        printf("      chroma offset: %d\n", schro_unpack_decode_uint(&unpack));
        printf("      chroma excursion: %d\n", schro_unpack_decode_uint(&unpack));
      }
    }

    MARKER();

    bit = schro_unpack_decode_bit(&unpack);
    printf("  colour spec flag: %s\n", bit ? "yes" : "no");
    if (bit) {
      int index = schro_unpack_decode_uint(&unpack);
      printf("    colour spec index: %d\n", index);
      if (index == 0) {
        bit = schro_unpack_decode_bit(&unpack);
        printf("      colour primaries flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          printf("        colour primaries: %d\n",
              schro_unpack_decode_uint(&unpack));
        }
        bit = schro_unpack_decode_bit(&unpack);
        printf("      colour matrix flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          printf("        colour matrix: %d\n", schro_unpack_decode_uint(&unpack));
        }
        bit = schro_unpack_decode_bit(&unpack);
        printf("      transfer function flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          printf("        transfer function: %d\n",
              schro_unpack_decode_uint(&unpack));
        }
      }
    }

    printf("  interlaced_coding: %d\n", schro_unpack_decode_uint(&unpack));

    MARKER();

  } else if (SCHRO_PARSE_CODE_IS_PICTURE(data[4])) {
    int num_refs = SCHRO_PARSE_CODE_NUM_REFS(data[4]);
    int bit;
    int n;
    int i;
    int lowdelay = SCHRO_PARSE_CODE_IS_LOW_DELAY(data[4]);

    printf("  num refs: %d\n", num_refs);

    schro_unpack_byte_sync(&unpack);
    printf("  picture_number: %u\n", schro_unpack_decode_bits(&unpack, 32));
    if (num_refs > 0) {
      printf("  ref1_offset: %d\n", schro_unpack_decode_sint(&unpack));
    }
    if (num_refs > 1) {
      printf("  ref2_offset: %d\n", schro_unpack_decode_sint(&unpack));
    }
    if (SCHRO_PARSE_CODE_IS_REFERENCE(data[4])) {
      int r = schro_unpack_decode_sint(&unpack);
      if (r == 0) {
        printf("  retire: none\n");
      } else {
        printf("  retire: %d\n", r);
      }
    }

    if (num_refs > 0) {
      int index;

      schro_unpack_byte_sync(&unpack);
      index = schro_unpack_decode_uint(&unpack);
      
      printf("  block parameters index: %d\n", index);
      if (index == 0) {
        printf("    luma block width: %d\n", schro_unpack_decode_uint(&unpack));
        printf("    luma block height: %d\n", schro_unpack_decode_uint(&unpack));
        printf("    horiz luma block sep: %d\n", schro_unpack_decode_uint(&unpack));
        printf("    vert luma block sep: %d\n", schro_unpack_decode_uint(&unpack));
      }
      
      MARKER();

      printf("  motion vector precision bits: %d\n", schro_unpack_decode_uint(&unpack));
      
      MARKER();

      bit = schro_unpack_decode_bit(&unpack);
      printf("  using global motion flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        for(i=0;i<num_refs;i++){
          printf("    global motion ref%d:\n", i+1);
          bit = schro_unpack_decode_bit(&unpack);
          printf("      non-zero pan/tilt flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            printf("        pan %d\n", schro_unpack_decode_sint(&unpack));
            printf("        tilt %d\n", schro_unpack_decode_sint(&unpack));
          }
          bit = schro_unpack_decode_bit(&unpack);
          printf("      non-zero zoom rot shear flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            printf("       exponent %d\n", schro_unpack_decode_uint(&unpack));
            printf("       A11 %d\n", schro_unpack_decode_sint(&unpack));
            printf("       A12 %d\n", schro_unpack_decode_sint(&unpack));
            printf("       A21 %d\n", schro_unpack_decode_sint(&unpack));
            printf("       A22 %d\n", schro_unpack_decode_sint(&unpack));
          }
          bit = schro_unpack_decode_bit(&unpack);
          printf("     non-zero perspective flag: %s\n", bit ? "yes" : "no");
          if (bit) {
            printf("       exponent %d\n", schro_unpack_decode_uint(&unpack));
            printf("       perspective_x %d\n", schro_unpack_decode_sint(&unpack));
            printf("       perspective_y %d\n", schro_unpack_decode_sint(&unpack));
          }
        }
      }

      MARKER();

      printf("  picture prediction mode: %d\n", schro_unpack_decode_uint(&unpack));

      bit = schro_unpack_decode_bit(&unpack);
      printf("  non-default weights flag: %s\n", bit ? "yes" : "no");
      if (bit) {
        printf("  picture weight precision: %d\n",
            schro_unpack_decode_uint(&unpack));
        for(i=0;i<num_refs;i++){
          printf("  picture weight ref%d: %d\n", i+1,
              schro_unpack_decode_sint(&unpack));
        }
      }

      MARKER();

      schro_unpack_byte_sync(&unpack);

      n = schro_unpack_decode_uint(&unpack);
      printf("  superblock split data length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      n = schro_unpack_decode_uint(&unpack);
      printf("  prediction modes data length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      n = schro_unpack_decode_uint(&unpack);
      printf("  vector data (ref1,x) length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      n = schro_unpack_decode_uint(&unpack);
      printf("  vector data (ref1,y) length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      if (num_refs>1) {
        n = schro_unpack_decode_uint(&unpack);
        printf("  vector data (ref2,x) length: %d\n", n);
        schro_unpack_byte_sync (&unpack);
        schro_unpack_skip_bits (&unpack, n*8);

        n = schro_unpack_decode_uint(&unpack);
        printf("  vector data (ref2,y) length: %d\n", n);
        schro_unpack_byte_sync (&unpack);
        schro_unpack_skip_bits (&unpack, n*8);
      }

      n = schro_unpack_decode_uint(&unpack);
      printf("  DC data (y) length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      n = schro_unpack_decode_uint(&unpack);
      printf("  DC data (u) length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

      n = schro_unpack_decode_uint(&unpack);
      printf("  DC data (v) length: %d\n", n);
      schro_unpack_byte_sync (&unpack);
      schro_unpack_skip_bits (&unpack, n*8);

    }

    schro_unpack_byte_sync (&unpack);
    if (num_refs == 0) {
      bit = 0;
    } else {
      bit = schro_unpack_decode_bit (&unpack);
      printf("  zero residual: %s\n", bit ? "yes" : "no");
    }
    if (!bit) {
      int depth;
      int j;

      printf("  wavelet index: %d\n", schro_unpack_decode_uint(&unpack));

      depth = schro_unpack_decode_uint(&unpack);
      printf("  transform depth: %d\n", depth);

      if (!lowdelay) {
        bit = schro_unpack_decode_bit (&unpack);
        printf("    spatial partition flag: %s\n", bit ? "yes" : "no");
        if (bit) {
          for(i=0;i<depth+1;i++){
            printf("      number of codeblocks depth=%d\n", i);
            printf("        horizontal codeblocks: %d\n",
                schro_unpack_decode_uint(&unpack));
            printf("        vertical codeblocks: %d\n",
                schro_unpack_decode_uint(&unpack));
          }
          printf("    codeblock mode index: %d\n", schro_unpack_decode_uint(&unpack));
        }

        schro_unpack_byte_sync (&unpack);
        for(j=0;j<3;j++){
          printf("  component %d:\n",j);
          printf("    comp subband  length  quantiser_index\n");
          for(i=0;i<1+depth*3;i++){
            int length;

            if(unpack.overrun) {
              printf("    PAST END\n");
              continue;
            }

            length = schro_unpack_decode_uint(&unpack);
            if (length > 0) {
              printf("    %4d %4d:   %6d    %3d\n", j, i, length,
                  schro_unpack_decode_uint(&unpack));
              schro_unpack_byte_sync(&unpack);
              schro_unpack_skip_bits (&unpack, length*8);
            } else {
              printf("    %4d %4d:   %6d\n", j, i, length);
              schro_unpack_byte_sync(&unpack);
            }
          }
        }
      } else {
        int slice_x;
        int slice_y;
        int slice_bytes_numerator;
        int slice_bytes_denominator;

        slice_x = schro_unpack_decode_uint(&unpack);
        slice_y = schro_unpack_decode_uint(&unpack);
        slice_bytes_numerator = schro_unpack_decode_uint(&unpack);
        slice_bytes_denominator = schro_unpack_decode_uint(&unpack);

        printf("  n_horiz_slices: %d\n", slice_x);
        printf("  n_horiz_slices: %d\n", slice_y);
        printf("  slice_bytes_numerator: %d\n", slice_bytes_numerator);
        printf("  slice_bytes_denominator: %d\n", slice_bytes_denominator);

        bit = schro_unpack_decode_bit (&unpack);
        printf("  encode_quant_matrix: %s\n", bit ? "yes" : "no");
        if (bit) {
          for(i=0;i<1+depth*3;i++){
            printf("    %2d: %d\n", i, schro_unpack_decode_uint(&unpack));
          }
        }

        schro_unpack_byte_sync (&unpack);
      }
    }
  } else if (data[4] == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
    int length = next - 14;
    int code = data[13];
    switch (code) {
      case 0:
        printf("  code: 0 (invalid)\n");
        break;
      case 1:
        printf("  code: 1 (encoder implementation/version)\n");
        printf("  string: %.*s\n", length, data + 14);
        break;
      case 2:
        printf("  code: 2 (SMPTE 12M timecode)\n");
        break;
      case 3:
        {
          int i;
          printf("  code: 3 (MD5 checksum)\n");
          printf("  checksum: ");
          for(i=0;i<16;i++){
            printf("%02x", data[14+i]);
          }
          printf("\n");
        }
        break;
      case 4:
        {
          int bitrate;
          printf("  code: %d (bitrate)\n", code);
          bitrate = (data[14]<<24);
          bitrate |= (data[15]<<16);
          bitrate |= (data[16]<<8);
          bitrate |= (data[17]<<0);
          printf("  bitrate: %d\n", bitrate);
        }
        break;
      default:
        printf("  code: %d (unknown)\n", code);
        dump_hex (data + 14, length, "    ");
        break;
    }

    schro_unpack_skip_bits (&unpack, (1 + length)*8);
  } else if (data[4] == SCHRO_PARSE_CODE_PADDING) {
    int length = next - 13;
    schro_unpack_skip_bits (&unpack, length*8);
  }

  schro_unpack_byte_sync (&unpack);

  printf("offset %d\n", schro_unpack_get_bits_read (&unpack)/8);
  dump_hex (unpack.data, MIN(data + size - unpack.data, 100), "  ");
}

