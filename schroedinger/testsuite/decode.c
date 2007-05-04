
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>

#include <liboil/liboilrandom.h>

static void decode (unsigned char *data, int length);

void *
getfile (const char *filename, int *length)
{
  FILE *file;
  void *data;
  int n_allocated = 0;
  int n = 0;
  int ret;

  file = fopen (filename, "r");
  if (!file) return NULL;

  n_allocated = 4096;
  n = 0;
  data = malloc(n_allocated);
  while(!feof(file)) {
    if (n == n_allocated) {
      n_allocated *= 2;
      data = realloc (data, n_allocated);
    }

    ret = fread (data + n, 1, n_allocated - n, file);
    n += ret;

    if (ferror(file)) {
      printf("error reading file\n");
      exit(0);
    }
  }

  fclose (file);
  *length = n;
  return data;
}

int
main (int argc, char *argv[])
{
  unsigned char *data;
  int length;

  schro_init();

  if (argc < 2) {
    printf("decode stream.drc\n");
    exit(1);
  }

  data = getfile (argv[1], &length);
  if (!data) {
    exit(1);
  }

  decode (data, length);

  free(data);

  return 0;
}


static void
decode (unsigned char *data, int length)
{
  SchroDecoder *decoder;
  SchroBuffer *buffer;
  SchroVideoFormat *format = NULL;
  SchroFrame *frame;
  int offset = 0;
  int packet_length;
  int go;
  int it;

  decoder = schro_decoder_new();

  while(length > 0) {
    if (length < 13) {
      printf("offset=%d: frame too short\n", offset);
      exit(0);
    }
    if (memcmp(data, "BBCD", 4) != 0) {
      printf("offset=%d: failed to find BBCD marker\n", offset);
      exit(0);
    }

    packet_length = (data[5]<<24) | (data[6]<<16) | (data[7]<<8) | (data[8]);
    if (packet_length > 0x1000000) {
      printf("offset=%d: large packet (%d > 0x1000000)\n", offset, packet_length);
      exit(0);
    }
    if (packet_length > length) {
      printf("offset=%d: packet past end (%d > %d)\n", offset, packet_length, length);
      exit(0);
    }

    buffer = schro_buffer_new_with_data (data, packet_length);
    schro_decoder_push (decoder, buffer);

    go = 1;
    while (go) {
      it = schro_decoder_iterate (decoder);

      switch (it) {
        case SCHRO_DECODER_FIRST_ACCESS_UNIT:
          format = schro_decoder_get_video_format (decoder);
          break;
        case SCHRO_DECODER_NEED_BITS:
          go = 0;
          break;
        case SCHRO_DECODER_NEED_FRAME:
          frame = schro_frame_new_and_alloc2 (SCHRO_FRAME_FORMAT_U8,
              format->width, format->height,
              format->width/2, format->height/2);
          schro_decoder_add_output_picture (decoder, frame);
          break;
        case SCHRO_DECODER_OK:
          frame = schro_decoder_pull (decoder);
          printf("got frame %p\n", frame);

          if (frame) {
            printf("picture number %d\n", frame->frame_number);

            schro_frame_unref (frame);
          }
        case SCHRO_DECODER_EOS:
          break;
        case SCHRO_DECODER_ERROR:
          printf("offset=%d: decoder error\n", offset);
          exit(0);
          break;
      }
    }

    offset += packet_length;
    data += packet_length;
    length -= packet_length;
  }

  schro_decoder_free (decoder);
  free(format);
}

