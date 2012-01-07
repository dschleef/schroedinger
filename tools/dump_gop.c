
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
  printf("  marker: %d\n", schro_unpack_decode_uint(&unpack)); \
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
handle_packet (unsigned char *data, int size)
{
  SchroUnpack unpack;
  const char *parse_code SCHRO_UNUSED;
  int next;
  int prev SCHRO_UNUSED;

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
    default:
      parse_code = "unknown";
      break;
  }
  schro_unpack_init_with_data (&unpack, data + 5, size - 5, 1);

  next = schro_unpack_decode_bits (&unpack, 32);
  prev = schro_unpack_decode_bits (&unpack, 32);

  if (data[4] == SCHRO_PARSE_CODE_SEQUENCE_HEADER) {
    printf("AU\n");
    printf("pictur: ");
    printf("    ");
    printf("  ");
    printf("  ref1 ");
    printf("  ref2 ");
    printf("retire ");
    printf("     size \n");
  } else if (SCHRO_PARSE_CODE_IS_PICTURE(data[4])) {
    int num_refs = SCHRO_PARSE_CODE_NUM_REFS(data[4]);
    int pic_num;
    char ref_chars[3] = { 'I', 'P', 'B' };

    schro_unpack_byte_sync(&unpack);
    pic_num = schro_unpack_decode_bits(&unpack, 32);
    printf("%6d: ", pic_num);
    if (SCHRO_PARSE_CODE_IS_REFERENCE(data[4])) {
      printf("ref ");
    } else {
      printf("    ");
    }

    printf("%c ", ref_chars[num_refs]);

    if (num_refs > 0) {
      printf("%6d ", pic_num + schro_unpack_decode_sint(&unpack));
    } else {
      printf("       ");
    }
    if (num_refs > 1) {
      printf("%6d ", pic_num + schro_unpack_decode_sint(&unpack));
    } else {
      printf("       ");
    }

    if (SCHRO_PARSE_CODE_IS_REFERENCE(data[4])) {
      int r = schro_unpack_decode_sint(&unpack);
      if (r == 0) {
        printf("  none ");
      } else {
        printf("%6d ", pic_num + r);
      }
    } else {
      printf("       ");
    }
    printf(" %8d\n", next);
  } else if (data[4] == SCHRO_PARSE_CODE_AUXILIARY_DATA) {
  }

  schro_unpack_byte_sync (&unpack);
}

