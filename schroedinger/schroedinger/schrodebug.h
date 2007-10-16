
#ifndef __SCHRO_DEBUG_H__
#define __SCHRO_DEBUG_H__

#include <schroedinger/schroutils.h>
#include <stdlib.h>

SCHRO_BEGIN_DECLS

enum
{
  SCHRO_LEVEL_NONE = 0,
  SCHRO_LEVEL_ERROR,
  SCHRO_LEVEL_WARNING,
  SCHRO_LEVEL_INFO,
  SCHRO_LEVEL_DEBUG,
  SCHRO_LEVEL_LOG
};

enum
{
  SCHRO_DUMP_SUBBAND_CURVE,
  SCHRO_DUMP_SUBBAND_EST,
  SCHRO_DUMP_PICTURE,
  SCHRO_DUMP_PSNR,
  SCHRO_DUMP_SSIM,
  SCHRO_DUMP_LAMBDA_CURVE,
  SCHRO_DUMP_HIST_TEST,
  SCHRO_DUMP_SCENE_CHANGE,
  SCHRO_DUMP_LAST
};

#define SCHRO_ERROR(...) \
  SCHRO_DEBUG_LEVEL(SCHRO_LEVEL_ERROR, __VA_ARGS__)
#define SCHRO_WARNING(...) \
  SCHRO_DEBUG_LEVEL(SCHRO_LEVEL_WARNING, __VA_ARGS__)
#define SCHRO_INFO(...) \
  SCHRO_DEBUG_LEVEL(SCHRO_LEVEL_INFO, __VA_ARGS__)
#define SCHRO_DEBUG(...) \
  SCHRO_DEBUG_LEVEL(SCHRO_LEVEL_DEBUG, __VA_ARGS__)
#define SCHRO_LOG(...) \
  SCHRO_DEBUG_LEVEL(SCHRO_LEVEL_LOG, __VA_ARGS__)

#define SCHRO_DEBUG_LEVEL(level,...) \
  schro_debug_log ((level), __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define SCHRO_ASSERT(test) do { \
  if (!(test)) { \
    SCHRO_ERROR("assertion failed: " #test ); \
    abort(); \
  } \
} while(0)

#define SCHRO_MILD_ASSERT(test) do { \
  if (!(test)) { \
    SCHRO_ERROR("mild assertion failed: " #test ); \
    abort(); \
  } \
} while(0)

void schro_debug_log (int level, const char *file, const char *function,
    int line, const char *format, ...);
void schro_debug_set_level (int level);
int schro_debug_get_level (void);

void schro_dump (int type, const char *format, ...);

SCHRO_END_DECLS

#endif
