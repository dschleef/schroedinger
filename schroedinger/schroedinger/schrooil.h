
#ifndef __SCHRO_NOTOIL_H__
#define __SCHRO_NOTOIL_H__

#include <schroedinger/schro-stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void oil_add_s16_u8(int16_t *d_n, int16_t *s1_n, uint8_t *s2_n, int n);
void oil_subtract_s16(int16_t *d_n, int16_t *s1_n, int16_t *s2_n, int n);
void oil_subtract_s16_u8(int16_t *d_n, int16_t *s1_n, uint8_t *s2_n, int n);
void oil_splat_s16_ns (int16_t *dest, int16_t *src, int n);
void oil_lift_haar_split (int16_t *i1, int16_t *i2, int n);
void oil_lift_haar_synth (int16_t *i1, int16_t *i2, int n);
void oil_synth_haar (int16_t *d, int16_t *s, int n);
void oil_split_haar (int16_t *d, int16_t *s, int n);
void oil_multsumshift8_str_s16 (int16_t *d, int16_t *s, int sstr, int16_t *s2_8,
    int16_t *s3_1, int16_t *s4_1, int n);

#ifdef __cplusplus
}
#endif

#endif

