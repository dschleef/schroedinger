
#ifndef _SCHRO_ENGINE_H_
#define _SCHRO_ENGINE_H_

int schro_encoder_engine_intra_only (SchroEncoder *encoder);
int schro_encoder_engine_backref (SchroEncoder *encoder);
int schro_encoder_engine_tworef (SchroEncoder *encoder);
int schro_encoder_engine_fourref (SchroEncoder *encoder);

#endif

