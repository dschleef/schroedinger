
#ifndef __SCHRO_NOTCOG_H__
#define __SCHRO_NOTCOG_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void schro_cog_mas8_u8_edgeextend (uint8_t *d, const uint8_t *s,
    const int16_t *taps, int offset, int shift, int index_offset, int n);
void schro_cog_mas10_u8_edgeextend (uint8_t *d, const uint8_t *s,
    const int16_t *taps, int offset, int shift, int index_offset, int n);

#endif

SCHRO_END_DECLS

#endif

