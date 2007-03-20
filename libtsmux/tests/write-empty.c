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

gboolean write_cb (guint8 *data, guint len, void *user_data)
{
  guint i;

  g_print ("Output of %u bytes\nContents:\n", len);
 
  for (i = 0; i < len; i++) {
    g_print (" 0x%02x", data[i]);
    if ((i+1 % 16) == 0)
      g_print ("\n");
  }
  if (i % 16)
    g_print ("\n");

  return TRUE;
}

const guint8 dummy_data[4] = "BBCD";

int main (int argc, char **argv)
{
  TsMux *mux;
  TsMuxProgram *program;
  TsMuxStream *stream;
  int i;
  
  mux = tsmux_new ();
  tsmux_set_write_func (mux, write_cb, NULL);

  program = tsmux_program_new (mux);

  stream = tsmux_create_stream (mux, TSMUX_ST_VIDEO_DIRAC, TSMUX_PID_AUTO);
  tsmux_program_add_stream (program, stream);
  tsmux_program_set_pcr_stream (program, stream);

  /* Try to write a buffer without having added any data, should write the 
   * PAT, PMT, then padded buffers with no payload */

  for (i = 0; i < 3; i++) {
    if (!tsmux_write_stream_packet (mux, stream)) {
      g_print ("Failed writing buffer\n");
      return 5;
    }
  }
  
  tsmux_free (mux);
  return 0;
}

