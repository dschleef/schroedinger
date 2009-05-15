
#ifndef __SCHRO_ORC_H__
#define __SCHRO_ORC_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void orc_add2_rshift_add_s16_22 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int n);
void orc_add2_rshift_sub_s16_22 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int n);
void orc_add2_rshift_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int n);
void orc_add2_rshift_add_s16_11 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int n);
void orc_add2_rshift_sub_s16_11 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int n);
void orc_add2_rshift_sub_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int n);
void orc_add_const_rshift_s16_11 (int16_t *d1, int16_t *s1, int n);
void orc_add_const_rshift_s16_ip (int16_t *d1, int shift, int n);
void orc_add_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n);
void orc_addc_rshift_s16 (int16_t *d1, int16_t *s1, int16_t *s2_2, int n);
void orc_lshift1_s16 (int16_t *d1, int16_t *s1, int n);
void orc_lshift2_s16 (int16_t *d1, int16_t *s1, int n);
void orc_mas2_across_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int16_t *s5_2, int n);
void orc_mas2_add_s16_ip (int16_t *d1, int16_t *s2, int mult, int offset,
    int shift, int n);
void orc_mas2_sub_s16_ip (int16_t *d1, int16_t *s2, int mult, int offset,
    int shift, int n);
void orc_mas4_add_s16_1991 (int16_t *d1, int16_t *s1, int16_t *s2, int shift, int n);
void orc_mas4_sub_s16_1991 (int16_t *d1, int16_t *s1, int16_t *s2, int shift, int n);
void orc_mas4_across_add_s16_1991_ip (int16_t *d1, int16_t *s2, int stride, int shift, int n);
void orc_mas4_across_sub_s16_1991_ip (int16_t *d1, int16_t *s2, int stride, int shift, int n);
void orc_subtract_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n);
void orc_memcpy (void *dest, void *src, int n);
void orc_add_s16_u8 (int16_t *d, int16_t *src1, uint8_t *src2, int n);
void orc_convert_s16_u8 (int16_t *d, uint8_t *src1, int n);
void orc_convert_u8_s16 (uint8_t *d, int16_t *src1, int n);
void orc_subtract_s16_u8 (int16_t *d, int16_t *src1, uint8_t *src2, int n);
void orc_lshift_s16_ip (int16_t *d1, int shift, int n);
void orc_splat_s16_ns (int16_t *d1, int value, int n);
void orc_splat_u8_ns (uint8_t *d1, int value, int n);
void orc_average_u8 (uint8_t *d1, uint8_t *s1, uint8_t *s2, int n);
void orc_multiply_and_add_s16_u8 (int16_t *d, int16_t *src1, int16_t *src2,
    uint8_t *src3, int n);
void orc_rrshift6_s16_ip (int16_t *d1, int n);
void orc_unpack_yuyv_y (uint8_t *d1, uint16_t *s1, int n);
void orc_unpack_yuyv_u (uint8_t *d1, uint32_t *s1, int n);
void orc_unpack_yuyv_v (uint8_t *d1, uint32_t *s1, int n);


#endif

SCHRO_END_DECLS

#endif

