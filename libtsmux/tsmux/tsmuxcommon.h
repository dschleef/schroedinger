#ifndef __TSMUX_COMMON_H__
#define __TSMUX_COMMON_H__

#include <glib.h>

#undef TS_DEBUG_ON

G_BEGIN_DECLS

#define TSMUX_SYNC_BYTE 0x47
#define TSMUX_PACKET_LENGTH 188
#define TSMUX_HEADER_LENGTH 4
#define TSMUX_PAYLOAD_LENGTH (TSMUX_PACKET_LENGTH - TSMUX_HEADER_LENGTH)

#define TSMUX_MIN_ES_DESC_LEN 8

/* Frequency for PCR representation */
#define TSMUX_SYS_CLOCK_FREQ (27000000L)
/* Frequency for PTS values */
#define TSMUX_CLOCK_FREQ (TSMUX_SYS_CLOCK_FREQ / 300)

#define TSMUX_PACKET_FLAG_NONE            (0)
#define TSMUX_PACKET_FLAG_ADAPTATION      (1 << 0)
#define TSMUX_PACKET_FLAG_DISCONT         (1 << 1)
#define TSMUX_PACKET_FLAG_RANDOM_ACCESS   (1 << 2)
#define TSMUX_PACKET_FLAG_PRIORITY        (1 << 3)
#define TSMUX_PACKET_FLAG_WRITE_PCR       (1 << 4)
#define TSMUX_PACKET_FLAG_WRITE_OPCR      (1 << 5)
#define TSMUX_PACKET_FLAG_WRITE_SPLICE    (1 << 6)
#define TSMUX_PACKET_FLAG_WRITE_ADAPT_EXT (1 << 7)

/* PES stream specific flags */
#define TSMUX_PACKET_FLAG_PES_FULL_HEADER   (1 << 8)
#define TSMUX_PACKET_FLAG_PES_WRITE_PTS     (1 << 9)
#define TSMUX_PACKET_FLAG_PES_WRITE_PTS_DTS (1 << 10)
#define TSMUX_PACKET_FLAG_PES_WRITE_ESCR    (1 << 11)
#define TSMUX_PACKET_FLAG_PES_EXT_STREAMID  (1 << 12)

typedef struct TsMuxPacketInfo TsMuxPacketInfo;
typedef struct TsMuxProgram TsMuxProgram;
typedef struct TsMuxStream TsMuxStream;

struct TsMuxPacketInfo {
  guint16 pid;
  guint32 flags;

  guint64 pcr;
  guint64 opcr;

  guint8 splice_countdown;

  guint8 private_data_len;
  guint8 private_data [256];

  guint8 packet_count; /* continuity counter */

  guint stream_avail; /* Number of payload bytes available */
  gboolean packet_start_unit_indicator;
};

static inline void
tsmux_put16 (guint8 **pos, guint16 val)
{
  *(*pos)++ = (val >> 8) & 0xff;
  *(*pos)++ = val & 0xff;
}

static inline void
tsmux_put32 (guint8 **pos, guint32 val)
{
  *(*pos)++ = (val >> 24) & 0xff;
  *(*pos)++ = (val >> 16) & 0xff;
  *(*pos)++ = (val >> 8) & 0xff;
  *(*pos)++ = val & 0xff;
}

static inline void
tsmux_put_ts (guint8 **pos, guint8 id, gint64 ts)
{
  /* 1: 4 bit id value | TS [32..30] | marker_bit */
  *(*pos)++ = ((id << 4) | ((ts >> 29) & 0x0E) | 0x01) & 0xff;
  /* 2, 3: TS[29..15] | marker_bit */
  tsmux_put16 (pos, ((ts >> 14) & 0xfffe) | 0x01);
  /* 4, 5: TS[14..0] | marker_bit */
  tsmux_put16 (pos, ((ts << 1) & 0xfffe) | 0x01);
}

#ifdef TS_DEBUG_ON
#define TS_DEBUG(...) g_print(__VA_ARGS__); g_print ("\n")
#else
#define TS_DEBUG(...)
#endif

G_END_DECLS

#endif
