
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>
#include <string.h>


#if 0
/* This is 64*log2 of the gain of the DC part of the wavelet transform */
static const int wavelet_gain[] = { 64, 64, 64, 0, 64, 128, 128, 103 };
/* horizontal/vertical part */
static const int wavelet_gain_hv[] = { 64, 64, 64, 0, 64, 128, 0, 65 };
/* diagonal part */
static const int wavelet_gain_diag[] = { 128, 128, 128, 64, 128, 256, -64, 90 };
#endif

void schro_encoder_choose_quantisers_simple (SchroEncoderTask *task);
void schro_encoder_choose_quantisers_hardcoded (SchroEncoderTask *task);

void
schro_encoder_choose_quantisers (SchroEncoderTask *task)
{

  switch (task->encoder->quantiser_engine) {
    case 0:
      schro_encoder_choose_quantisers_hardcoded (task);
      break;
    case 1:
      schro_encoder_choose_quantisers_simple (task);
      break;
  }
}

void
schro_encoder_choose_quantisers_simple (SchroEncoderTask *task)
{
  SchroSubband *subbands = task->encoder_frame->subbands;
  int depth = task->params.transform_depth;
  int base;
  int i;

  base = task->encoder->prefs[SCHRO_PREF_QUANT_BASE];

  if (depth >= 1) {
    subbands[(depth-1)*3 + 1].quant_index = base;
    subbands[(depth-1)*3 + 2].quant_index = base;
    subbands[(depth-1)*3 + 3].quant_index = base + 4;
  }
  if (depth >= 2) {
    subbands[(depth-2)*3 + 1].quant_index = base - 5;
    subbands[(depth-2)*3 + 2].quant_index = base - 5;
    subbands[(depth-2)*3 + 3].quant_index = base - 1;
  }
  for(i=3;i<=depth;i++){
    subbands[(depth-i)*3 + 1].quant_index = base - 6;
    subbands[(depth-i)*3 + 2].quant_index = base - 6;
    subbands[(depth-i)*3 + 3].quant_index = base - 2;
  }
  subbands[0].quant_index = base - 10;

  if (!task->encoder_frame->is_ref) {
    for(i=0;i<depth*3+1;i++){
      subbands[i].quant_index += 4;
    }
  }

}

void
schro_encoder_choose_quantisers_hardcoded (SchroEncoderTask *task)
{
  SchroSubband *subbands = task->encoder_frame->subbands;
  int depth = task->params.transform_depth;
  int i;

  /* hard coded.  muhuhuhahaha */

  /* these really only work for DVD-ish quality with 5,3, 9,3 and 13,5 */

  if (depth >= 1) {
    subbands[(depth-1)*3 + 1].quant_index = 22;
    subbands[(depth-1)*3 + 2].quant_index = 22;
    subbands[(depth-1)*3 + 3].quant_index = 26;
  }
  if (depth >= 2) {
    subbands[(depth-2)*3 + 1].quant_index = 17;
    subbands[(depth-2)*3 + 2].quant_index = 17;
    subbands[(depth-2)*3 + 3].quant_index = 21;
  }
  for(i=3;i<=depth;i++){
    subbands[(depth-i)*3 + 1].quant_index = 16;
    subbands[(depth-i)*3 + 2].quant_index = 16;
    subbands[(depth-i)*3 + 3].quant_index = 20;
  }
  subbands[0].quant_index = 12;

  if (!task->encoder_frame->is_ref) {
    for(i=0;i<depth*3+1;i++){
      subbands[i].quant_index += 4;
    }
  }

}

