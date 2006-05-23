
#ifndef _SCHRO_WAVELET_H_
#define _SCHRO_WAVELET_H_

#include <schroedinger/schro-stdint.h>

enum {
  SCHRO_WAVELET_APPROX97,
  SCHRO_WAVELET_5_3,
  SCHRO_WAVELET_13_5,
  SCHRO_WAVELET_HAAR,
  SCHRO_WAVELET_reserved1,
  SCHRO_WAVELET_reserved2,
  SCHRO_WAVELET_DAUB97
};

void schro_deinterleave (int16_t *d_n, int16_t *s_n, int n);
void schro_deinterleave_str (int16_t *d_n, int dstr, int16_t *s_n, int n);
void schro_interleave (int16_t *d_n, int16_t *s_n, int n);
void schro_interleave_str (int16_t *d_n, int16_t *s_n, int sstr, int n);

void schro_lift_split_daub97 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_split_daub97_str (int16_t *d_n, int16_t *s_n, int sstr, int n);
void schro_lift_split_53 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_split_53_str (int16_t *d_n, int16_t *s_n, int sstr, int n);
void schro_lift_split_approx97 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_split_135 (int16_t *d_n, int16_t *s_n, int n);

void schro_lift_split (int type, int16_t *dest, int16_t *src, int n);
void schro_lift_split_str (int type, int16_t *dest, int16_t *src, int sstr, int n);

void schro_lift_synth_daub97 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_synth_daub97_str (int16_t *d_n, int dstr, int16_t *s_n, int n);
void schro_lift_synth_53 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_synth_53_str (int16_t *d_n, int dstr, int16_t *s_n, int n);
void schro_lift_synth_approx97 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_synth_135 (int16_t *d_n, int16_t *s_n, int n);
void schro_lift_synth (int type, int16_t *dest, int16_t *src, int n);
void schro_lift_synth_str (int type, int16_t *dest, int dstr, int16_t *src, int n);

void schro_wt_daub97 (int16_t *i_n, int n);
void schro_iwt_daub97 (int16_t *i_n, int n);
void schro_wt_approx97 (int16_t *i_n, int n);
void schro_iwt_approx97 (int16_t *i_n, int n);
void schro_wt_5_3 (int16_t *i_n, int n);
void schro_iwt_5_3 (int16_t *i_n, int n);
void schro_wt_13_5 (int16_t *i_n, int n);
void schro_iwt_13_5 (int16_t *i_n, int n);

void schro_wt (int type, int16_t *d_n, int16_t *s_n, int n);
void schro_iwt (int type, int16_t *d_n, int16_t *s_n, int n);

void schro_wt_2d (int type, int16_t *i_n, int n, int stride);
void schro_iwt_2d (int type, int16_t *i_n, int n, int stride);

void schro_wavelet_transform_2d (int type, int16_t *i_n, int stride, int width, int height, int16_t *tmp);
void schro_wavelet_inverse_transform_2d (int type, int16_t *i_n, int stride, int width, int height, int16_t *tmp);

#endif

