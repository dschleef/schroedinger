
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>

extern int _schro_decode_prediction_only;

/**
 * schro_init:
 *
 * Intializes the Schroedinger library.  This function must be called
 * before any other function in the library.
 */
void
schro_init(void)
{
  const char *s;

  oil_init();

  s = getenv ("SCHRO_DEBUG");
  if (s && s[0]) {
    char *end;
    int level;

    level = strtoul (s, &end, 0);
    if (end[0] == 0) {
      schro_debug_set_level (level);
    }
  }

  s = getenv ("SCHRO_DECODE_PREDICTION_ONLY");
  if (s && s[0]) {
    _schro_decode_prediction_only = TRUE;
  }
}

