
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <stdarg.h>
#include <stdio.h>

static const char *carid_debug_level_names[] = {
  "NONE",
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG",
  "LOG"
};

static int carid_debug_level = CARID_LEVEL_ERROR;

void
carid_debug_log (int level, const char *file, const char *function,
    int line, const char *format, ...)
{
#ifdef HAVE_GLIB
  va_list varargs;
  char *s;

  if (level > carid_debug_level)
    return;

  va_start (varargs, format);
  s = g_strdup_vprintf (format, varargs);
  va_end (varargs);

  fprintf (stderr, "CARID: %s: %s(%d): %s: %s\n",
      carid_debug_level_names[level], file, line, function, s);
  g_free (s);
#else
  va_list varargs;
  char s[1000];

  if (level > carid_debug_level)
    return;

  va_start (varargs, format);
  vsnprintf (s, 999, format, varargs);
  va_end (varargs);

  fprintf (stderr, "CARID: %s: %s(%d): %s: %s\n",
      carid_debug_level_names[level], file, line, function, s);
#endif
}

void
carid_debug_set_level (int level)
{
  carid_debug_level = level;
}

int
carid_debug_get_level (void)
{
  return carid_debug_level;
}

