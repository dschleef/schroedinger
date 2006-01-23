
#ifndef __CARID_DEBUG_H__
#define __CARID_DEBUG_H__

enum
{
  CARID_LEVEL_NONE = 0,
  CARID_LEVEL_ERROR,
  CARID_LEVEL_WARNING,
  CARID_LEVEL_INFO,
  CARID_LEVEL_DEBUG,
  CARID_LEVEL_LOG
};

#define CARID_ERROR(...) \
  CARID_DEBUG_LEVEL(CARID_LEVEL_ERROR, __VA_ARGS__)
#define CARID_WARNING(...) \
  CARID_DEBUG_LEVEL(CARID_LEVEL_WARNING, __VA_ARGS__)
#define CARID_INFO(...) \
  CARID_DEBUG_LEVEL(CARID_LEVEL_INFO, __VA_ARGS__)
#define CARID_DEBUG(...) \
  CARID_DEBUG_LEVEL(CARID_LEVEL_DEBUG, __VA_ARGS__)
#define CARID_LOG(...) \
  CARID_DEBUG_LEVEL(CARID_LEVEL_LOG, __VA_ARGS__)

#define CARID_DEBUG_LEVEL(level,...) \
  carid_debug_log ((level), __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define CARID_ASSERT(test) do { \
  if (!(test)) { \
    CARID_ERROR("assertion failed: " #test ); \
  } \
} while(0)

void carid_debug_log (int level, const char *file, const char *function,
    int line, const char *format, ...);
void carid_debug_set_level (int level);
int carid_debug_get_level (void);

#endif
