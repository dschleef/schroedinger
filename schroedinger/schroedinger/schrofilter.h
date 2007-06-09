
#ifndef __SCHRO_FILTER_H__
#define __SCHRO_FILTER_H__

#include <schroedinger/schroframe.h>

SCHRO_BEGIN_DECLS

void schro_frame_filter_cwm7 (SchroFrame *frame);
void schro_frame_filter_cwmN (SchroFrame *frame, int weight);
void schro_frame_filter_cwmN_ref (SchroFrame *frame, int weight);

void schro_filter_cwmN (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n, int weight);
void schro_filter_cwmN_ref (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n, int weight);
void schro_filter_cwm7 (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n);

SCHRO_END_DECLS

#endif

