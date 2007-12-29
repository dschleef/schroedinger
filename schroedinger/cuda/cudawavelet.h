#ifndef CUDAWAVELET
#define CUDAWAVELET

#ifdef __cplusplus
extern "C" {
#endif

extern void cuda_wavelet_inverse_transform_2d (int filter, int16_t *data, int stride, int width, int height);
extern void cuda_wavelet_transform_2d (int filter, int16_t *data, int stride, int width, int height);

#ifdef __cplusplus
};
#endif

#endif
