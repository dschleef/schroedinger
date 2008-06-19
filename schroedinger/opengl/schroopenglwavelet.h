
#ifndef __SCHRO_OPENGL_WAVELET_H__
#define __SCHRO_OPENGL_WAVELET_H__

SCHRO_BEGIN_DECLS

void schro_opengl_wavelet_transform_2d (SchroFrameData *frame_data, int filter);
void schro_opengl_wavelet_inverse_transform_2d (SchroFrameData *frame_data,
    int filter);

SCHRO_END_DECLS

#endif

