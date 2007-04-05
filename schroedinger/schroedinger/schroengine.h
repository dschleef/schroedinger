
#ifndef _SCHRO_ENGINE_H_
#define _SCHRO_ENGINE_H_

SCHRO_BEGIN_DECLS

int schro_encoder_engine_intra_only (SchroEncoder *encoder);
int schro_encoder_engine_backref (SchroEncoder *encoder);
int schro_encoder_engine_backref2 (SchroEncoder *encoder);
int schro_encoder_engine_tworef (SchroEncoder *encoder);

SCHRO_END_DECLS

#endif

