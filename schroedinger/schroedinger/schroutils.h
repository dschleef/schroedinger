
#ifndef __SCHRO_SCHROUTILS_H__
#define __SCHRO_SCHROUTILS_H__

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ARRAY_SIZE(x) (sizeof(x)/sizeof(x[0]))
#define DIVIDE_ROUND_UP(a,b) (((a) + (b) - 1)/(b))
#ifndef MIN
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif
#ifndef CLAMP
#define CLAMP(x,a,b) ((x)<(a) ? (a) : ((x)>(b) ? (b) : (x)))
#endif
#define ROUND_UP_SHIFT(x,y) (((x) + (1<<(y)) - 1)>>(y))
#define ROUND_UP_POW2(x,y) (((x) + (1<<(y)) - 1)&((~0)<<(y)))
#define ROUND_UP_2(x) ROUND_UP_POW2(x,1)
#define ROUND_UP_4(x) ROUND_UP_POW2(x,2)
#define ROUND_UP_8(x) ROUND_UP_POW2(x,3)
#define OFFSET(ptr,offset) ((void *)(((uint8_t *)(ptr)) + (offset)))
#define SCHRO_GET(ptr, offset, type) (*(type *)((uint8_t *)(ptr) + (offset)) )

#define schro_divide(a,b) (((a)<0)?(((a) - (b) + 1)/(b)):((a)/(b)))

#if defined(__GNUC__) && defined(__GNUC_MINOR__)
#define SCHRO_GNUC_PREREQ(maj, min) \
  ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
#define SCHRO_GNUC_PREREQ(maj, min) 0
#endif
  
#if SCHRO_GNUC_PREREQ(3,3) && defined(__ELF__)
#define SCHRO_INTERNAL __attribute__ ((visibility ("internal")))
#else
#define SCHRO_INTERNAL
#endif

#ifdef __cplusplus
#define SCHRO_BEGIN_DECLS extern "C" {
#define SCHRO_END_DECLS }
#else
#define SCHRO_BEGIN_DECLS
#define SCHRO_END_DECLS
#endif

int muldiv64 (int a, int b, int c);
int schro_utils_multiplier_to_quant_index (double x);
int schro_dequantise (int q, int quant_factor, int quant_offset);
int schro_quantise (int value, int quant_factor, int quant_offset);
double schro_utils_probability_to_entropy (double x);
double schro_utils_entropy (double a, double total);

#endif

