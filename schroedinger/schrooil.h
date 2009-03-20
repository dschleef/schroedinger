
#ifndef __SCHRO_NOTOIL_H__
#define __SCHRO_NOTOIL_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schroorc.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void oil_splat_s16_ns (int16_t *dest, const int16_t *src, int n);
void oil_sum_s32_u8 (int32_t *d_1, uint8_t *src, int n);
void oil_sum_s32_s16 (int32_t *d_1, int16_t *src, int n);
void oil_sum_square_diff_u8 (int32_t *d_1, uint8_t *s1, uint8_t *s2, int n);
void oil_mas10_s16 (int16_t *d, const int16_t *s1_np3, const int32_t *s2_4,
    const int32_t *s3_2, int n);
void oil_convert_f64_u8 (double *dest, uint8_t *src, int n);
void oil_iir3_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n);
void oil_iir3_rev_s16_f64 (int16_t *d, int16_t *s, double *i_3, double *s2_4, int n);
void oil_iir3_across_u8_f64 (uint8_t *d, uint8_t *s, double *i1, double *i2, double *i3, double *s2_4, int n);
void oil_iir3_across_s16_f64 (int16_t *d, int16_t *s, double *i1, double *i2, double *i3, double *s2_4, int n);
void oil_convert_f64_s16 (double *dest, int16_t *src, int n);
void oil_iir3_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n);
void oil_iir3_rev_u8_f64 (uint8_t *d, uint8_t *s, double *i_3, double *s2_4, int n);

#endif

SCHRO_END_DECLS

#endif

