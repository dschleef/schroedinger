
#ifndef SCHRO_METRIC_H
#define SCHRO_METRIC_H

#include <schroedinger/schroutils.h>
#include <schroedinger/schro-stdint.h>

SCHRO_BEGIN_DECLS

int schro_metric_absdiff_u8 (uint8_t *a, int a_stride, uint8_t *b,
    int b_stride, int width, int height);
int schro_metric_haar (uint8_t *src1, int stride1, uint8_t *src2, int stride2,
    int width, int height);
int schro_metric_haar_const (uint8_t *data, int stride, int dc_value,
    int width, int height);
int schro_metric_abssum_s16 (int16_t *data, int stride, int width, int height);
int schro_metric_sum_u8 (uint8_t *data, int stride, int width, int height);

SCHRO_END_DECLS

#endif

