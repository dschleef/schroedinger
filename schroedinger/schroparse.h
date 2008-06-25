
#ifndef __SCHRO_PARSE_H__
#define __SCHRO_PARSE_H__

#include <schroedinger/schrovideoformat.h>

SCHRO_BEGIN_DECLS

int schro_parse_decode_sequence_header (uint8_t *data, int length,
    SchroVideoFormat *video_format);

SCHRO_END_DECLS

#endif

