#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <glib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <tsmux/tsmux.h>

struct WriteContext {
  gint fd;
};

#define BUF_SIZE 4096 

struct ReadContext {
  gint fd;
  gboolean eos;
  gint bufs_in_use;
  guint8 buf[2][BUF_SIZE];

  TsMuxStream *stream;
};

gboolean write_cb (guint8 *data, guint len, void *user_data)
{
  struct WriteContext *write_ctx = (struct WriteContext *) user_data;

  if (write (write_ctx->fd, data, len) < len) {
    g_print ("Failed to write buffer of len %u\n", len);
    return FALSE;
  }

  g_print ("Wrote %u bytes\n", len);

  return TRUE;
}

gboolean read_fill_buf (struct ReadContext *ctx, guint8 *buf, gint *bytes_read)
{
  int res = read (ctx->fd, buf, BUF_SIZE);

  *bytes_read = res;

  if (res < 0)
    return FALSE;

  return TRUE;
}

void buffer_release_cb (guint8 *data, void *user_data)
{
  struct ReadContext *ctx = (struct ReadContext *) user_data;
  gint bytes_read = BUF_SIZE;

  ctx->bufs_in_use--;

  /* Read in more data for this buffer */
  if (read_fill_buf (ctx, data, &bytes_read)) {
    if (bytes_read < BUF_SIZE)
      ctx->eos = TRUE;
    if (bytes_read > 0) {
      tsmux_stream_add_data (ctx->stream, data, bytes_read, ctx, -1, -1);
      ctx->bufs_in_use++;
    }
  }
  else
    ctx->eos = TRUE;
}


gboolean init_bufs (struct ReadContext *ctx)
{
  gint i;
  gint bytes_read = BUF_SIZE;

  for (i = 0; i < 2; i++) {
    if (!read_fill_buf (ctx, ctx->buf[i], &bytes_read))
      return FALSE;

    tsmux_stream_add_data (ctx->stream, ctx->buf[i], bytes_read, ctx, -1, -1);
    ctx->bufs_in_use++;

    /* Hit eos early */
    if (bytes_read < BUF_SIZE)
      return TRUE;
  }

  return TRUE;
} 

int main (int argc, char **argv)
{
  struct WriteContext write_ctx;
  struct ReadContext read_ctx;
  TsMux *mux;
  TsMuxProgram *program;
  gboolean first = TRUE;
  
  if (argc != 3) {
    g_print ("Usage: test_mux outfile.ts input-dirac-es\n");
    return 1;
  }

  write_ctx.fd = creat (argv[1], 
                S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
  if (write_ctx.fd < 0) {
    g_print ("Failed to open output file: %s\n", argv[1]);
    return 1;
  }
  g_print ("Writing to %s\n", argv[1]);

  read_ctx.fd = open (argv[2], O_RDONLY);
  read_ctx.bufs_in_use = 0;
  read_ctx.eos = FALSE;
  if (read_ctx.fd < 0) {
    g_print ("Failed to open input file %s\n", argv[2]);
    return 2;
  }
  g_print ("Reading from %s\n", argv[2]);

  mux = tsmux_new ();
  tsmux_set_write_func (mux, write_cb, &write_ctx);

  program = tsmux_program_new (mux);

  read_ctx.stream = tsmux_create_stream (mux, TSMUX_ST_VIDEO_DIRAC, 
        TSMUX_PID_AUTO);
  tsmux_stream_set_buffer_release_func (read_ctx.stream, buffer_release_cb);
  tsmux_program_add_stream (program, read_ctx.stream);
  tsmux_program_set_pcr_stream (program, read_ctx.stream);

  if (!init_bufs (&read_ctx)) {
    g_print ("Could not read initial data for input\n");
    return 3;
  }
  
  while (read_ctx.bufs_in_use) {
    TsMuxPacketInfo *pi = &read_ctx.stream->pi;

    /* Force an adaptation field on the first packet to write a null PCR */
    if (first) {
      pi->flags |= TSMUX_PACKET_FLAG_ADAPTATION | TSMUX_PACKET_FLAG_WRITE_PCR;
      pi->pcr = 0;
      first = FALSE;
    }

    g_print ("Writing packet with %d bytes avail, flags 0x%04x\n",
        tsmux_stream_bytes_avail (read_ctx.stream), pi->flags);

    if (!tsmux_write_stream_packet (mux, read_ctx.stream)) {
      g_print ("Failed writing buffer\n");
      return 5;
    }
  }

  close (write_ctx.fd);
  close (read_ctx.fd);

  tsmux_free (mux);
  return 0;
}

