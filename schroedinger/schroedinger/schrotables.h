
#ifndef __SCHRO_SCHRO_TABLES_H__
#define __SCHRO_SCHRO_TABLES_H__

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schroutils.h>

SCHRO_BEGIN_DECLS

extern uint32_t schro_table_offset_3_8[61];
extern uint32_t schro_table_offset_1_2[61];
extern uint32_t schro_table_quant[61];
extern uint32_t schro_table_inverse_quant[61];
extern uint16_t schro_table_division_factor[257];
extern double schro_table_error_hist_shift3_1_2[60][104];

extern const float schro_tables_wavelet_noise_curve[8][8][128];
extern const double schro_tables_wavelet_gain[8][2];

SCHRO_END_DECLS

#endif

