
#ifndef _SCHRO_ENGINE_H_
#define _SCHRO_ENGINE_H_

#include <schroedinger/schroencoder.h>

SCHRO_BEGIN_DECLS

typedef enum {
  SCHRO_QUANTISER_ENGINE_PERCEPTUAL,
  SCHRO_QUANTISER_ENGINE_LOSSLESS,
  SCHRO_QUANTISER_ENGINE_HARDCODED,
  SCHRO_QUANTISER_ENGINE_SIMPLE
} SchroQuantiserEngineEnum;

int schro_encoder_engine_intra_only (SchroEncoder *encoder);
int schro_encoder_engine_backref (SchroEncoder *encoder);
int schro_encoder_engine_backref2 (SchroEncoder *encoder);
int schro_encoder_engine_tworef (SchroEncoder *encoder);
int schro_encoder_engine_test_intra (SchroEncoder *encoder);
int schro_encoder_engine_lossless (SchroEncoder *encoder);
int schro_encoder_engine_backtest (SchroEncoder *encoder);

SCHRO_END_DECLS

#endif

