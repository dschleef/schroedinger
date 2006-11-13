#ifndef __TSMUX_H__
#define __TSMUX_H__

#include <glib.h>

#include <tsmux/tsmuxcommon.h>
#include <tsmux/tsmuxstream.h>

G_BEGIN_DECLS

#define TSMUX_MAX_ES_INFO_LENGTH ((1 << 12) - 1)
#define TSMUX_MAX_SECTION_LENGTH (4096)

#define TSMUX_PID_AUTO ((guint16)-1)

typedef struct TsMuxSection TsMuxSection;
typedef struct TsMux TsMux;

typedef gboolean (*TsMuxWriteFunc) (guint8 *data, guint len, void *user_data);

struct TsMuxSection {
  TsMuxPacketInfo pi;

  /* Private sections can be up to 4096 bytes */
  guint8 data[TSMUX_MAX_SECTION_LENGTH];
};

/* Information for the streams associated with one program */
struct TsMuxProgram {
  TsMuxSection pmt;
  guint8   pmt_version;
  gboolean pmt_changed;

  guint    pmt_frequency;
  gint64   last_pmt_ts;

  guint16 pgm_number; /* program ID for the PAT */
  guint16 pmt_pid; /* PID to write the PMT */

  TsMuxStream *pcr_stream; /* Stream which carries the PCR */
  gint64 last_pcr;

  GArray *streams; /* Array of TsMuxStream pointers */
  guint nb_streams;
};

struct TsMux {
  guint nb_streams;
  GList *streams;    /* TsMuxStream* array of all streams */

  guint nb_programs;
  GList *programs;   /* TsMuxProgram* array of all programs */

  guint16 transport_id;

  guint16 next_pgm_no;
  guint16 next_pmt_pid;
  guint16 next_stream_pid;

  TsMuxSection pat;
  guint8   pat_version;
  gboolean pat_changed;

  guint    pat_frequency;
  gint64   last_pat_ts;

  guint8 packet_buf[TSMUX_PACKET_LENGTH];
  TsMuxWriteFunc write_func;
  void *write_func_data;

  /* Scratch space for writing ES_info descriptors */
  guint8 es_info_buf[TSMUX_MAX_ES_INFO_LENGTH];
};

/* create/free new muxer session */
TsMux *		tsmux_new 			(void);
void 		tsmux_free 			(TsMux *mux);

/* Setting muxing session properties */
void 		tsmux_set_write_func 		(TsMux *mux, TsMuxWriteFunc func, void *user_data);
void 		tsmux_set_pat_frequency 	(TsMux *mux, guint freq);
guint 		tsmux_get_pat_frequency 	(TsMux *mux);
guint16		tsmux_get_new_pid 		(TsMux *mux);

/* pid/program management */
TsMuxProgram *	tsmux_program_new 		(TsMux *mux);
void 		tsmux_program_free 		(TsMuxProgram *program);
void 		tsmux_set_pmt_frequency 	(TsMuxProgram *program, guint freq);
guint 		tsmux_get_pmt_frequency 	(TsMuxProgram *program);

/* stream management */
TsMuxStream *	tsmux_create_stream 		(TsMux *mux, TsMuxStreamType stream_type, guint16 pid);
TsMuxStream *	tsmux_find_stream 		(TsMux *mux, guint16 pid);

void 		tsmux_program_add_stream 	(TsMuxProgram *program, TsMuxStream *stream);
void 		tsmux_program_set_pcr_stream 	(TsMuxProgram *program, TsMuxStream *stream);

/* writing stuff */
gboolean 	tsmux_write_stream_packet 	(TsMux *mux, TsMuxStream *stream); 

G_END_DECLS

#endif
