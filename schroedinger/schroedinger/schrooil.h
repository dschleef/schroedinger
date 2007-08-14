
#ifndef __SCHRO_NOTOIL_H__
#define __SCHRO_NOTOIL_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schro-stdint.h>

SCHRO_BEGIN_DECLS

void oil_splat_s16_ns (int16_t *dest, const int16_t *src, int n);
void oil_lift_haar_split (int16_t *i1, int16_t *i2, int n);
void oil_lift_haar_synth (int16_t *i1, int16_t *i2, int n);
void oil_synth_haar (int16_t *d, const int16_t *s, int n);
void oil_split_haar (int16_t *d, const int16_t *s, int n);
void oil_multsumshift8_str_s16 (int16_t *d, const int16_t *s, int sstr,
    const int16_t *s2_8, const int16_t *s3_1, const int16_t *s4_1, int n);
void oil_mas10_across_u8 (uint8_t *d, const uint8_t *s1, const int16_t *s2_10,
    const int16_t *s3_2, int n);
void oil_mas10_u8 (uint8_t *d, const uint8_t *s1, int sstr,
    const int16_t *s2_10, const int16_t *s3_2, int n);
void oil_add_const_rshift_u16 (uint16_t *d, const uint16_t *s1,
    const int16_t *s2_2, int n);

SCHRO_END_DECLS

#endif

