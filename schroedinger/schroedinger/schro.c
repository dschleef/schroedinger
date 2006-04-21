
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schro/schro.h>
#include <liboil/liboil.h>
#include <stdlib.h>


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
}

