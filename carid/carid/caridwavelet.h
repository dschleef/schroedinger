
#ifndef _CARID_WAVELET_H_
#define _CARID_WAVELET_H_

#include <carid/carid-stdint.h>

enum {
  CARID_WAVELET_DAUB97,
  CARID_WAVELET_APPROX97,
  CARID_WAVELET_5_3,
  CARID_WAVELET_13_5
};

void carid_deinterleave (int16_t *d_n, int16_t *s_n, int n);
void carid_interleave (int16_t *d_n, int16_t *s_n, int n);
void carid_lift_synth_haar (int16_t *i_n, int n);
void carid_lift_split_haar (int16_t *i_n, int n);
void carid_lift_synth_daub97_ext (int16_t *i_n, int n);
void carid_lift_split_daub97_ext (int16_t *i_n, int n);
void carid_lift_split_53_ext (int16_t *i_n, int n);
void carid_lift_split_approx97_ext (int16_t *i_n, int n);
void carid_lift_split_135_ext (int16_t *i_n, int n);

void carid_lift_synth_daub97_ext (int16_t *i_n, int n);
void carid_lift_synth_53_ext (int16_t *i_n, int n);
void carid_lift_synth_approx97_ext (int16_t *i_n, int n);
void carid_lift_synth_135_ext (int16_t *i_n, int n);

void carid_iwt_daub97 (int16_t *i_n, int n);
void carid_iiwt_daub97 (int16_t *i_n, int n);
void carid_iwt_approx97 (int16_t *i_n, int n);
void carid_iiwt_approx97 (int16_t *i_n, int n);
void carid_iwt_5_3 (int16_t *i_n, int n);
void carid_iiwt_5_3 (int16_t *i_n, int n);
void carid_iwt_13_5 (int16_t *i_n, int n);
void carid_iiwt_13_5 (int16_t *i_n, int n);

void carid_iwt_2d (int type, int16_t *i_n, int n, int stride);
void carid_iiwt_2d (int type, int16_t *i_n, int n, int stride);

#endif

