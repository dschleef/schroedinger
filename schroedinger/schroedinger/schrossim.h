
#ifndef __SCHRO_SSIM_H__
#define __SCHRO_SSIM_H__

#include <schroedinger/schroframe.h>

SCHRO_BEGIN_DECLS

#ifndef SCHRO_DISABLE_UNSTABLE_API

double schro_ssim (SchroFrame *a, SchroFrame *b);

#endif

SCHRO_END_DECLS

#endif

