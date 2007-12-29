#ifndef H_CUDA_UPSAMPLE
#define H_CUDA_UPSAMPLE

#ifdef __cplusplus
extern "C" {
#endif

void cuda_upsample_horizontal(uint8_t *output, int ostride, uint8_t *input, int istride, int width, int height);
void cuda_upsample_vertical(uint8_t *output, int ostride, uint8_t *input, int istride, int width, int height);

#ifdef __cplusplus
}
#endif
#endif
