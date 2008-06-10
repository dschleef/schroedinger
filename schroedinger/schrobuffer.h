
#ifndef __SCHRO_BUFFER_H__
#define __SCHRO_BUFFER_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroBuffer SchroBuffer;

struct _SchroBuffer
{
  /*< private >*/
  unsigned char *data;
  int length;

  int ref_count;

  SchroBuffer *parent;

  void (*free) (SchroBuffer *, void *);
  void *priv;
};

SchroBuffer *schro_buffer_new (void);
SchroBuffer *schro_buffer_new_and_alloc (int size);
SchroBuffer *schro_buffer_new_with_data (void *data, int size);
SchroBuffer *schro_buffer_new_subbuffer (SchroBuffer * buffer, int offset,
    int length);
SchroBuffer *schro_buffer_dup (SchroBuffer * buffer);
SchroBuffer * schro_buffer_ref (SchroBuffer * buffer);
void schro_buffer_unref (SchroBuffer * buffer);

SCHRO_END_DECLS

#endif
