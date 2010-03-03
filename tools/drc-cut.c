
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrobitstream.h>
#include <schroedinger/schrounpack.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

int parse_packet (FILE *file, unsigned char **p_data, int *p_size);
int write_packet (FILE *outfile, unsigned char *packet, int size);

const char *fn = "output.drc";

int
main (int argc, char *argv[])
{
  unsigned char *packet;
  unsigned char *seq_header_packet = NULL;
  int seq_header_size = 0;
  SchroPictureNumber start_picture;
  SchroPictureNumber end_picture;
  SchroPictureNumber pic_num;
  int size;
  FILE *file;
  FILE *outfile;
  int ret;
  int in_segment = FALSE;

  if (argc < 5) {
    fprintf(stderr, "dirac_cut infile.drc outfile.drc start_picture end_picture\n");
    return 1;
  }

  file = fopen (argv[1], "r");
  if (file == NULL) {
    printf("cannot open %s for reading: %s\n", argv[1], strerror(errno));
    return 1;
  }

  outfile = fopen (argv[2], "w");
  if (outfile == NULL) {
    printf("cannot open %s for writing: %s\n", argv[2], strerror(errno));
    return 1;
  }

  start_picture = strtoul (argv[3], NULL, 0);
  end_picture = strtoul (argv[4], NULL, 0);

  while (1) {
    ret = parse_packet (file, &packet, &size);
    if (!ret) {
      exit(1);
    }

    if (seq_header_packet == NULL && SCHRO_PARSE_CODE_IS_SEQ_HEADER(packet[4])) {
      seq_header_packet = packet;
      seq_header_size = size;
      continue;
    }

    if (SCHRO_PARSE_CODE_IS_PICTURE(packet[4])) {
      pic_num = (packet[13] << 24) | (packet[14]<<16) |
        (packet[15]<<8) | packet[16];
      printf("got picture %d\n", pic_num);
    } else {
      pic_num = 0;
    }

    if (in_segment) {
      if (SCHRO_PARSE_CODE_IS_PICTURE(packet[4]) && pic_num >= end_picture) {
#if 0
        unsigned char eos_packet[13];

        in_segment = FALSE;

        eos_packet[0] = 'B'
        eos_packet[1] = 'B'
        eos_packet[2] = 'C'
        eos_packet[3] = 'D'
        eos_packet[4] = SCHRO_PARSE_CODE_EOS;

        write_packet (outfile, eos_packet, 13);
#endif

        exit(0);
      }
    } else {
      if (SCHRO_PARSE_CODE_IS_PICTURE(packet[4])) {
        if (pic_num >= start_picture) {
          printf("pushing seq header\n");
          write_packet (outfile, seq_header_packet, seq_header_size);
          in_segment = TRUE;
        }
      }
    }

    if (in_segment) {
      if (SCHRO_PARSE_CODE_IS_PICTURE(packet[4])) {
        printf("pushing picture %d (was %d)\n", pic_num - start_picture, pic_num);
        pic_num -= start_picture;

        packet[13] = (pic_num>>24)&0xff;
        packet[14] = (pic_num>>16)&0xff;
        packet[15] = (pic_num>>8)&0xff;
        packet[16] = (pic_num>>0)&0xff;
      } else {
        printf("pushing non-picture\n");
      }

      write_packet (outfile, packet, size);
    }

    free(packet);
  }

  return 0;
}

unsigned int last_offset;

int
write_packet (FILE *outfile, unsigned char *packet, int size)
{
  int ret;

  if (packet[4] == SCHRO_PARSE_CODE_END_OF_SEQUENCE) {
    packet[5] = 0;
    packet[6] = 0;
    packet[7] = 0;
    packet[8] = 0;
  } else {
    packet[5] = (size>>24)&0xff;
    packet[6] = (size>>16)&0xff;
    packet[7] = (size>>8)&0xff;
    packet[8] = (size>>0)&0xff;
  }

  packet[9] = (last_offset>>24)&0xff;
  packet[10] = (last_offset>>16)&0xff;
  packet[11] = (last_offset>>8)&0xff;
  packet[12] = (last_offset>>0)&0xff;

  last_offset = size;

  ret = fwrite (packet, 1, size, outfile);

  if (ret != size) return FALSE;
  return TRUE;
}

int
parse_packet (FILE *file, unsigned char **p_data, int *p_size)
{
  unsigned char *packet;
  unsigned char header[13];
  int n;
  int size;

  n = fread (header, 1, 13, file);
  if (n == 0) {
    *p_data = NULL;
    *p_size = 0;
    return 1;
  }
  if (n < 13) {
    printf("truncated header\n");
    return 0;
  }

  if (header[0] != 'B' || header[1] != 'B' || header[2] != 'C' ||
      header[3] != 'D') {
    return 0;
  }

  size = (header[5]<<24) | (header[6]<<16) | (header[7]<<8) | (header[8]);
  if (size == 0) {
    size = 13;
  }
  if (size < 13) {
    return 0;
  }
  if (size > 16*1024*1024) {
    printf("packet too large? (%d > 16777216)\n", size);
    return 0;
  }

  packet = malloc (size);
  memcpy (packet, header, 13);
  n = fread (packet + 13, 1, size - 13, file);
  if (n < size - 13) {
    free (packet);
    return 0;
  }

  *p_data = packet;
  *p_size = size;
  return 1;
}

