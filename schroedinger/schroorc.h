
#ifndef __SCHRO_ORC_H__
#define __SCHRO_ORC_H__

#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void orc_add2_rshift_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int n);
void orc_add2_rshift_sub_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int n);
void orc_add_const_rshift_s16 (int16_t *d1, int16_t *s1, int16_t *s3_2, int n);
void orc_add_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n);
void orc_addc_rshift_s16 (int16_t *d1, int16_t *s1, int16_t *s2_2, int n);
void orc_lshift_s16 (int16_t *d1, int16_t *s1, int16_t *s3_1, int n);
void orc_mas2_across_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3, int16_t *s4_2, int16_t *s5_2, int n);
void orc_mas2_add_s16 (int16_t *d1, int16_t *s1, int16_t *s2, int16_t *s3_2, int16_t *s4_2, int n);
void orc_mas4_add_s16 (int16_t *d1, int16_t *s1, int16_t *s2, int16_t *s3_4, int16_t *s4_2, int n);
void orc_subtract_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n);

#endif

SCHRO_END_DECLS

#endif

