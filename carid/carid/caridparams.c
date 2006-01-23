
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <carid/carid.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>


static int
round_up_pow2 (int x, int p)
{
  int y = (1<<p) - 1;
  x += y;
  x &= ~y;
  return x;
}

void
carid_params_calculate_iwt_sizes (CaridParams *params)
{

  CARID_DEBUG ("chroma size %d x %d", params->chroma_width,
      params->chroma_height);
  if (params->is_intra) {
    params->iwt_chroma_width =
      round_up_pow2(params->chroma_width,params->transform_depth);
    params->iwt_chroma_height =
      round_up_pow2(params->chroma_height, params->transform_depth);
  } else {
    params->iwt_chroma_width =
      round_up_pow2(params->mc_chroma_width, params->transform_depth);
    params->iwt_chroma_height =
      round_up_pow2(params->mc_chroma_height, params->transform_depth);
  }
  CARID_DEBUG ("iwt chroma size %d x %d", params->iwt_chroma_width,
      params->iwt_chroma_height);
  params->iwt_luma_width =
    params->iwt_chroma_width * params->chroma_h_scale;
  params->iwt_luma_height =
    params->iwt_chroma_height * params->chroma_v_scale;
  CARID_DEBUG ("iwt luma size %d x %d", params->iwt_luma_width,
      params->iwt_luma_height);
}

