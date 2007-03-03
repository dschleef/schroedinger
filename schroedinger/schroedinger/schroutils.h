
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

#endif
