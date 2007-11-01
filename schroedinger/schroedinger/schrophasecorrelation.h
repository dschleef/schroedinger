
#ifndef __SCHRO_PHASECORRELATION_H__
#define __SCHRO_PHASECORRELATION_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/schroparams.h>

SCHRO_BEGIN_DECLS

#ifdef SCHRO_ENABLE_UNSTABLE_API

void schro_encoder_phasecorr_prediction (SchroEncoderFrame *frame);

#endif

SCHRO_END_DECLS

#endif

