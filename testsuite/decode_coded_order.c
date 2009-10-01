
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <schroedinger/schro.h>
#include <schroedinger/schrodebug.h>


#include "common.h"

void decode (FILE *file);
void parse (FILE *file);

int
main (int argc, char *argv[])
{
  FILE *file;

  schro_init();

  if (argc < 2) {
    printf("decode stream.drc\n");
    exit(1);
  }

  file = fopen (argv[1], "r");
  if (file == NULL) {
    printf("cannot open %s\n", argv[1]);
    return 1;
  }

  decode (file);
  //parse (file);

  fclose (file);

  return 0;
}

static void
buffer_free (SchroBuffer *buf, void *priv)
{
  free (priv);
}

void
decode (FILE *file)
{
  SchroDecoder *decoder;
  SchroBuffer *buffer;
  SchroVideoFormat *format = NULL;
  SchroFrame *frame;
  int picture_number;
  int go;
  int it;
  int eos = FALSE;
  void *packet;
  int size;
  int ret;
  int parse_code;

  decoder = schro_decoder_new();
  schro_decoder_set_picture_order (decoder, SCHRO_DECODER_PICTURE_ORDER_CODED);

  while(!eos) {
    ret = parse_packet (file, &packet, &size);
    if (!ret) {
      exit(1);
    }

    if (size == 0) {
      schro_decoder_push_end_of_stream (decoder);
    } else {
      parse_code = ((uint8_t *)packet)[4];

      buffer = schro_buffer_new_with_data (packet, size);
      buffer->free = buffer_free;
      buffer->priv = packet;

      printf("pushing 0x%02x size %d\n", parse_code, size);

      if (SCHRO_PARSE_CODE_IS_PICTURE (parse_code)) {
        frame = schro_frame_new_and_alloc (NULL,
            schro_params_get_frame_format(8, format->chroma_format),
            format->width, format->height);
        schro_decoder_add_output_picture (decoder, frame);
      }

      it = schro_decoder_push (decoder, buffer);
      if (it == SCHRO_DECODER_FIRST_ACCESS_UNIT) {
        format = schro_decoder_get_video_format (decoder);
      }

      if (!SCHRO_PARSE_CODE_IS_PICTURE (parse_code)) {
        continue;
      }
    }

    go = 1;
    while (go) {
      it = schro_decoder_wait (decoder);

      switch (it) {
        case SCHRO_DECODER_NEED_BITS:
          go = 0;
          break;
        case SCHRO_DECODER_NEED_FRAME:
          frame = schro_frame_new_and_alloc (NULL,
              schro_params_get_frame_format(8, format->chroma_format),
              format->width, format->height);
          schro_decoder_add_output_picture (decoder, frame);
          break;
        case SCHRO_DECODER_OK:
          picture_number = schro_decoder_get_picture_number (decoder);
          frame = schro_decoder_pull (decoder);
          printf("got frame %p, picture number %d\n", frame,
              picture_number);

          if (frame) {
            schro_frame_unref (frame);
          }
          go = 0;
          break;
        case SCHRO_DECODER_EOS:
          printf("got eos\n");
          eos = TRUE;
          go = 0;
          break;
        case SCHRO_DECODER_STALLED:
          printf("stall\n");
          go = 0;
          break;
        case SCHRO_DECODER_ERROR:
          exit(0);
          break;
      }
    }
  }

  printf("freeing decoder\n");
  schro_decoder_free (decoder);
  free(format);
}

void
parse (FILE *file)
{
  void *packet;
  int size;
  int ret;

  while(1) {
    ret = parse_packet (file, &packet, &size);
    if (!ret) {
      exit(1);
    }

    if (packet) free (packet);
    if (size == 0) return;
  }
}

