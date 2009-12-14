
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "dirac_parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define READ_SIZE 4096

unsigned char data[READ_SIZE];

int parse_packet (FILE *file, unsigned char **p_data, int *p_size);

int
main (int argc, char *argv[])
{
  FILE *file;
  int ret;
  DiracSequenceHeader header = {0};
  int size;
  unsigned char *packet;
  int n;
  int i;

  if (argc < 2) {
    fprintf(stderr, "parse_header infile.drc\n");
    return 1;
  }

  file = fopen (argv[1], "r");
  if (file == NULL) {
    printf("cannot open %s for reading: %s\n", argv[1], strerror(errno));
    return 1;
  }

  n = fread (data, 1, 4096, file);
  if (n < 13) {
    printf("file too short\n");
    exit(1);
  }

  for(i=0;i<=n - 13;i++){
    packet = data + i;
    if (packet[0] == 'B' && packet[1] == 'B' && packet[2] == 'C' &&
        packet[3] == 'D') {
      size = (packet[5]<<24) | (packet[6]<<16) | (packet[7]<<8) | (packet[8]);
      if (size == 0) {
        size = 13;
      }

      if (i + size > n) {
        continue;
      }

      if (SCHRO_PARSE_CODE_IS_SEQ_HEADER(packet[4])) {
        ret = dirac_sequence_header_parse (&header, packet + 13, size - 13);
        
        if (!ret) {
          printf("bad header\n");
          exit(1);
        }

#define PRINT(ack) printf( #ack ": %d\n", header. ack );
        PRINT(major_version);
        PRINT(minor_version);
        PRINT(profile);
        PRINT(level);
        PRINT(index);
        PRINT(width);
        PRINT(height);
        PRINT(chroma_format);
        PRINT(interlaced);
        PRINT(top_field_first);
        PRINT(frame_rate_numerator);
        PRINT(frame_rate_denominator);
        PRINT(aspect_ratio_numerator);
        PRINT(aspect_ratio_denominator);
        PRINT(clean_width);
        PRINT(clean_height);
        PRINT(left_offset);
        PRINT(top_offset);
        PRINT(luma_offset);
        PRINT(luma_excursion);
        PRINT(chroma_offset);
        PRINT(chroma_excursion);
        PRINT(colour_primaries);
        PRINT(colour_matrix);
        PRINT(transfer_function);
        PRINT(interlaced_coding);

        return 0;
      }
    }
  }

  return 0;
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

