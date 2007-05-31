
#ifndef __SCHRO_ANALYSIS_H__
#define __SCHRO_ANALYSIS_H__


#include <schroedinger/schroencoder.h>

SCHRO_BEGIN_DECLS

void schro_encoder_frame_analyse (SchroEncoder *encoder, SchroEncoderFrame *frame);
void schro_encoder_reference_analyse (SchroEncoderFrame *frame);

SCHRO_END_DECLS

#endif
