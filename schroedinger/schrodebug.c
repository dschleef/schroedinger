
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <stdarg.h>
#include <stdio.h>

static const char *schro_debug_level_names[] = {
  "NONE",
  "ERROR",
  "WARNING",
  "INFO",
  "DEBUG",
  "LOG"
};

static int schro_debug_level = SCHRO_LEVEL_ERROR;
int _schro_dump_enable;

void
schro_debug_log (int level, const char *file, const char *function,
    int line, const char *format, ...)
{
#ifdef HAVE_GLIB
  va_list varargs;
  char *s;

  if (level > schro_debug_level)
    return;

  va_start (varargs, format);
  s = g_strdup_vprintf (format, varargs);
  va_end (varargs);

  fprintf (stderr, "SCHRO: %s: %s(%d): %s: %s\n",
      schro_debug_level_names[level], file, line, function, s);
  g_free (s);
#else
  va_list varargs;
  char s[1000];

  if (level > schro_debug_level)
    return;

  va_start (varargs, format);
  vsnprintf (s, 999, format, varargs);
  va_end (varargs);

  fprintf (stderr, "SCHRO: %s: %s(%d): %s: %s\n",
      schro_debug_level_names[level], file, line, function, s);
#endif
}

void
schro_debug_set_level (int level)
{
  schro_debug_level = level;
}

int
schro_debug_get_level (void)
{
  return schro_debug_level;
}


static FILE *dump_files[SCHRO_DUMP_LAST];

static const char *dump_file_names[SCHRO_DUMP_LAST] = {
  "schro_dump.subband_curve",
  "schro_dump.subband_est",
  "schro_dump.picture",
  "schro_dump.psnr",
  "schro_dump.ssim",
  "schro_dump.lambda_curve",
  "schro_dump.hist_test",
  "schro_dump.scene_change",
  "schro_dump.phase_correlation",
  "schro_dump.motionest"
};

void
schro_dump (int type, const char *format, ...)
{
  va_list varargs;

  if (!_schro_dump_enable) return;

  if (dump_files[type] == NULL) {
    dump_files[type] = fopen (dump_file_names[type], "w");
  }

  va_start (varargs, format);
  vfprintf (dump_files[type], format, varargs);
  va_end (varargs);

  fflush (dump_files[type]);
}


