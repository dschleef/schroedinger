
#ifndef __SCHRO_PARSE_H__
#define __SCHRO_PARSE_H__

#include <schroedinger/schrovideoformat.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

int schro_parse_decode_sequence_header (uint8_t *data, int length,
    SchroVideoFormat *video_format);

#endif

SCHRO_END_DECLS

#endif

