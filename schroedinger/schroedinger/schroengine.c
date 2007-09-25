
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schroencoder.h>
#include <liboil/liboil.h>


void
schro_engine_check_new_access_unit(SchroEncoder *encoder,
    SchroEncoderFrame *frame)
{
  if (encoder->au_frame == -1 ||
      frame->frame_number >= encoder->au_frame + encoder->au_distance) {
    frame->start_access_unit = TRUE;
    encoder->au_frame = frame->frame_number;
  }
}

int
schro_engine_pick_output_buffer_size (SchroEncoder *encoder,
    SchroEncoderFrame *frame)
{
  int size;

  size = encoder->video_format.width * encoder->video_format.height;
  switch (encoder->video_format.chroma_format) {
    case SCHRO_CHROMA_444:
      size *= 3;
      break;
    case SCHRO_CHROMA_422:
      size *= 2;
      break;
    case SCHRO_CHROMA_420:
      size += size/2;
      break;
  }

  /* random scale factor of 2 in order to be safe */
  size *= 2;

  return size;
}

static void
init_params (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroEncoder *encoder = frame->encoder;

  params->video_format = &encoder->video_format;

  schro_params_init (params, params->video_format->index);

  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTER_WAVELET];
  } else {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTRA_WAVELET];
  }
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];

  params->mv_precision = 0;
  //params->have_global_motion = TRUE;
  
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);
}

static void
init_small_codeblocks (SchroParams *params)
{
  int i;
  int shift;

  params->nondefault_partition_flag = TRUE;
  params->horiz_codeblocks[0] = 1;
  params->vert_codeblocks[0] = 1;
  for(i=1;i<params->transform_depth+1;i++){
    shift = params->transform_depth + 1 - i;
    /* Size of codeblock is 32.  This value was pulled out of my anus of
     * holding. */
    params->horiz_codeblocks[i] = params->iwt_luma_width >> (shift + 3);
    params->vert_codeblocks[i] = params->iwt_luma_height >> (shift + 2);
    SCHRO_DEBUG("codeblocks %d %d %d", i, params->horiz_codeblocks[i],
        params->vert_codeblocks[i]);
  }
}

int
schro_encoder_engine_intra_only (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        frame->presentation_frame = frame->frame_number;

        frame->slot = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        init_params (frame);
        if (params->is_noarith) {
          init_small_codeblocks (params);
        }

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}


int
schro_encoder_engine_backref (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_ref;

    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        is_ref = FALSE;
        if (encoder->last_ref == -1 || 
            frame->frame_number >= encoder->last_ref + encoder->ref_distance) {
          is_ref = TRUE;
        }
        if (frame->start_access_unit) {
          is_ref = TRUE;
        }

        if (!is_ref) {
          if (!schro_encoder_reference_get (encoder, encoder->last_ref)) {
            continue;
          }
        }

        frame->presentation_frame = frame->frame_number;
        frame->slot = encoder->next_slot;
        encoder->next_slot++;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        frame->is_ref = is_ref;
        if (frame->is_ref) {
          params->num_refs = 0;
          if (frame->frame_number > 0) {
            frame->retire[0] = encoder->last_ref;
            frame->n_retire = 1;
          } else {
            frame->n_retire = 0;
          }
          encoder->last_ref = frame->frame_number;
        } else {
          params->num_refs = 1;
          frame->reference_frame_number[0] = encoder->last_ref;
          frame->n_retire = 0;
        }

        init_params (frame);

        if (params->num_refs > 0) {
          frame->ref_frame0 = schro_encoder_reference_get (encoder,
              frame->reference_frame_number[0]);
          schro_encoder_frame_ref (frame->ref_frame0);
        } else {
          frame->ref_frame0 = NULL;
        }

        SCHRO_DEBUG("queueing %d", frame->frame_number);

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

int
schro_encoder_engine_backref2 (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_intra;

    frame = encoder->frame_queue->elements[i].data;
    //SCHRO_ERROR("backref i=%d picture=%d", i, frame->frame_number);

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) {
      continue;
    }

    schro_engine_check_new_access_unit (encoder, frame);

    is_intra = FALSE;
    if (frame->start_access_unit) {
      is_intra = TRUE;
    }

    if (!is_intra) {
      if (!schro_encoder_reference_get (encoder, encoder->last_ref)) {
        return FALSE;
      }
    }

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;

    frame->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    encoder->next_slot++;

    frame->output_buffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);

    /* set up params */
    params = &frame->params;
    if (is_intra) {
      frame->is_ref = TRUE;
      params->num_refs = 0;
      if (frame->frame_number > 0) {
        frame->retire[0] = encoder->last_ref;
        frame->n_retire = 1;
      } else {
        frame->n_retire = 0;
      }
      encoder->last_ref = frame->frame_number;
      encoder->mid1_ref = frame->frame_number;
    } else if (frame->frame_number >= encoder->last_ref + 4) {
      frame->is_ref = TRUE;
      params->num_refs = 1;
      frame->reference_frame_number[0] = encoder->last_ref;
      frame->retire[0] = encoder->last_ref;
      frame->n_retire = 1;
      encoder->last_ref = frame->frame_number;
    } else {
      frame->is_ref = FALSE;
      params->num_refs = 1;
      frame->reference_frame_number[0] = encoder->last_ref;
      frame->n_retire = 0;
    }

    init_params (frame);

    if (params->num_refs > 0) {
      frame->ref_frame0 = schro_encoder_reference_get (encoder,
          frame->reference_frame_number[0]);
      schro_encoder_frame_ref (frame->ref_frame0);
    } else {
      frame->ref_frame0 = NULL;
    }

    SCHRO_DEBUG("queueing %d", frame->frame_number);

    schro_async_run_locked (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture_all, frame);

    return TRUE;
  }

  return FALSE;
}

static void
handle_gop (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *f;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);
  if (frame->start_access_unit) {
    frame->is_ref = TRUE;
    frame->num_refs = 0;
    SCHRO_DEBUG("preparing %d as AU", frame->frame_number);
    frame->state = SCHRO_ENCODER_FRAME_STATE_ENGINE_1;
    frame->slot = encoder->next_slot++;
    frame->presentation_frame = frame->frame_number;
    if (encoder->last_ref >= 0) {
      frame->n_retire = 1;
      SCHRO_DEBUG("marking %d for retire", encoder->last_ref);
      frame->retire[0] = encoder->last_ref;
    }
    encoder->last_ref = frame->frame_number;
    encoder->gop_picture = frame->frame_number + 1;
  } else {
    int ref_slot;
    int j;
    int gop_length;

    gop_length = 4;
    SCHRO_DEBUG("handling gop from %d to %d", encoder->gop_picture,
        encoder->gop_picture + gop_length - 1);

    if (i + gop_length >= encoder->frame_queue->n) {
      if (encoder->end_of_stream) {
        gop_length = encoder->frame_queue->n - i;
      } else {
        SCHRO_DEBUG("not enough pictures in queue");
        return;
      }
    }
    for (j = 0; j < gop_length; j++) {
      f = encoder->frame_queue->elements[i+j].data;

      if (f->busy || f->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) {
        SCHRO_DEBUG("picture %d not ready", i + j);
        return;
      }
    }

    ref_slot = encoder->next_slot++;
    for (j = 0; j < gop_length - 1; j++) {
      f = encoder->frame_queue->elements[i+j].data;
      f->is_ref = FALSE;
      f->num_refs = 2;
      f->picture_number_ref0 = frame->frame_number - 1;
      f->picture_number_ref1 = frame->frame_number + gop_length - 1;
      f->state = SCHRO_ENCODER_FRAME_STATE_ENGINE_1;
      SCHRO_DEBUG("preparing %d as inter (%d,%d)", f->frame_number,
          f->picture_number_ref0, f->picture_number_ref1);
      f->slot = encoder->next_slot++;
      f->presentation_frame = f->frame_number;
      if (j == gop_length - 2) {
        f->n_retire = 1;
        f->retire[0] = frame->frame_number - 1;
      }
    }

    f = encoder->frame_queue->elements[i+j].data;
    f->is_ref = TRUE;
    f->num_refs = 1;
    f->picture_number_ref0 = frame->frame_number - 1;
    f->state = SCHRO_ENCODER_FRAME_STATE_ENGINE_1;
    SCHRO_DEBUG("preparing %d as inter ref (%d)", f->frame_number,
        f->picture_number_ref0);
    f->presentation_frame = f->frame_number - 1;
    f->slot = ref_slot;
    encoder->last_ref = f->frame_number;

    i += gop_length - 1;

    encoder->gop_picture += gop_length;
  }
}

int
schro_encoder_engine_tworef (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;
  int ref;

  SCHRO_DEBUG("engine iteration");

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("analyse i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) continue;

    frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
    frame->busy = TRUE;
    schro_async_run_locked (encoder->async,
        (void (*)(void *))schro_encoder_analyse_picture, frame);
    return TRUE;
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      handle_gop (encoder, i);
      break;
    }
  }

  for(ref = 1; ref >= 0; ref--){

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    if (frame->is_ref != ref) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_ENGINE_1:
        if (frame->num_refs > 0 &&
            !schro_encoder_reference_get (encoder, frame->picture_number_ref0)) {
          SCHRO_DEBUG("ref0 (%d) not ready", frame->picture_number_ref0);
          continue;
        }

        if (frame->num_refs > 1 &&
            !schro_encoder_reference_get (encoder, frame->picture_number_ref1)) {
          SCHRO_DEBUG("ref1 (%d) not ready", frame->picture_number_ref1);
          continue;
        }

        frame->presentation_frame = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        params->num_refs = frame->num_refs;
        params->xbsep_luma = 8;
        params->xblen_luma = 12;
        params->ybsep_luma = 8;
        params->yblen_luma = 12;

        if (frame->num_refs > 0) {
          frame->reference_frame_number[0] = frame->picture_number_ref0;
        }
        if (frame->num_refs > 1) {
          frame->reference_frame_number[1] = frame->picture_number_ref1;
        }

        init_params (frame);

        if (params->num_refs > 0) {
          frame->ref_frame0 = schro_encoder_reference_get (encoder,
              frame->reference_frame_number[0]);
          schro_encoder_frame_ref (frame->ref_frame0);
        } else {
          frame->ref_frame0 = NULL;
        }
        if (params->num_refs > 1) {
          frame->ref_frame1 = schro_encoder_reference_get (encoder,
              frame->reference_frame_number[1]);
          schro_encoder_frame_ref (frame->ref_frame1);
        } else {
          frame->ref_frame1 = NULL;
        }

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }
  }

  return FALSE;
}


static struct {
  int type;
  int depth;
} test_wavelet_types[] = {
  /* These are the main wavelet/levels that get used */
  { SCHRO_WAVELET_DESL_9_3, 2 },
  { SCHRO_WAVELET_DESL_9_3, 3 },
  { SCHRO_WAVELET_DESL_9_3, 4 },
  { SCHRO_WAVELET_5_3, 2 },
  { SCHRO_WAVELET_5_3, 3 },
  { SCHRO_WAVELET_5_3, 4 },
  { SCHRO_WAVELET_13_5, 2 },
  { SCHRO_WAVELET_13_5, 3 },
  { SCHRO_WAVELET_13_5, 4 },
  { SCHRO_WAVELET_HAAR_0, 2 },
  { SCHRO_WAVELET_HAAR_0, 3 },
  { SCHRO_WAVELET_HAAR_0, 4 },
  { SCHRO_WAVELET_HAAR_1, 2 },
  { SCHRO_WAVELET_HAAR_1, 3 },
  { SCHRO_WAVELET_HAAR_1, 4 },
  { SCHRO_WAVELET_FIDELITY, 2 },
  { SCHRO_WAVELET_FIDELITY, 3 },
  { SCHRO_WAVELET_DAUB_9_7, 2 },
  { SCHRO_WAVELET_DAUB_9_7, 3 },

  /* 1-level transforms look crappy */
  { SCHRO_WAVELET_DESL_9_3, 1 },
  { SCHRO_WAVELET_5_3, 1 },
  { SCHRO_WAVELET_13_5, 1 },
  { SCHRO_WAVELET_HAAR_0, 1 },
  { SCHRO_WAVELET_HAAR_1, 1 },
  { SCHRO_WAVELET_FIDELITY, 1 },
  { SCHRO_WAVELET_DAUB_9_7, 1 },

#ifdef SCHRO_HAVE_DEEP_WAVELETS
  { SCHRO_WAVELET_FIDELITY, 4 },
  { SCHRO_WAVELET_DAUB_9_7, 4 }
#endif

#ifdef USE_TRANSFORM_LEVEL_5
  /* 5-level transforms don't decrease bitrate */
  { SCHRO_WAVELET_DESL_9_3, 5 },
  { SCHRO_WAVELET_5_3, 5 },
  { SCHRO_WAVELET_13_5, 5 },
  { SCHRO_WAVELET_HAAR_0, 5 },
  { SCHRO_WAVELET_HAAR_1, 5 },
#endif

#ifdef USE_TRANSFORM_LEVEL_6
  /* 6-level transforms don't decrease bitrate */
  { SCHRO_WAVELET_DESL_9_3, 6 },
  { SCHRO_WAVELET_5_3, 6 },
  { SCHRO_WAVELET_13_5, 6 },
  { SCHRO_WAVELET_HAAR_0, 6 },
  { SCHRO_WAVELET_HAAR_1, 6 },
#endif
};

int
schro_encoder_engine_test_intra (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;
  int j;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_SIMPLE;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        frame->presentation_frame = frame->frame_number;

        frame->slot = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        j = frame->frame_number % ARRAY_SIZE(test_wavelet_types);
        encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = test_wavelet_types[j].type;
        encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = test_wavelet_types[j].depth;
        init_params (frame);

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;

        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

int
schro_encoder_engine_lossless (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOSSLESS;
  encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_HAAR_0;
  encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_HAAR_0;
  encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 3;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_ref;

    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        is_ref = FALSE;
        if (encoder->last_ref == -1 || 
            frame->frame_number >= encoder->last_ref + encoder->ref_distance) {
          is_ref = TRUE;
        }
        if (frame->start_access_unit) {
          is_ref = TRUE;
        }

        if (!is_ref) {
          if (!schro_encoder_reference_get (encoder, encoder->last_ref)) {
            continue;
          }
        }

        frame->presentation_frame = frame->frame_number;

        frame->slot = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        frame->is_ref = is_ref;
        if (frame->is_ref) {
          params->num_refs = 0;
          if (frame->frame_number > 0) {
            frame->retire[0] = encoder->last_ref;
            frame->n_retire = 1;
          } else {
            frame->n_retire = 0;
          }
          encoder->last_ref = frame->frame_number;
        } else {
          params->num_refs = 1;
          frame->reference_frame_number[0] = encoder->last_ref;
          frame->n_retire = 0;
        }

        init_params (frame);
        if (params->is_noarith) {
          init_small_codeblocks (params);
        }

        params->xbsep_luma = 8;
        params->xblen_luma = 8;
        params->ybsep_luma = 8;
        params->yblen_luma = 8;

        if (params->num_refs > 0) {
          frame->ref_frame0 = schro_encoder_reference_get (encoder,
              frame->reference_frame_number[0]);
          schro_encoder_frame_ref (frame->ref_frame0);
        } else {
          frame->ref_frame0 = NULL;
        }

        SCHRO_DEBUG("queueing %d", frame->frame_number);

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

int
schro_encoder_engine_backtest (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;
  int j;
  int comp;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_SIMPLE;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_ref;

    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        is_ref = FALSE;
        if (encoder->last_ref == -1) {
          is_ref = TRUE;
        }
        if (frame->start_access_unit) {
          is_ref = TRUE;
        }

        if (!is_ref) {
          if (!schro_encoder_reference_get (encoder, encoder->last_ref)) {
            continue;
          }
        }

        frame->presentation_frame = frame->frame_number;
        frame->slot = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        frame->is_ref = is_ref;
        if (frame->is_ref) {
          params->num_refs = 0;
          if (frame->frame_number > 0) {
            frame->retire[0] = encoder->last_ref;
            frame->n_retire = 1;
          } else {
            frame->n_retire = 0;
          }
          encoder->last_ref = frame->frame_number;
        } else {
          params->num_refs = 1;
          frame->reference_frame_number[0] = encoder->last_ref;
          frame->n_retire = 0;
        }

        init_params (frame);

        params->xbsep_luma = 8;
        params->xblen_luma = 8;
        params->ybsep_luma = 8;
        params->yblen_luma = 8;

        for(comp=0;comp<3;comp++){
          for(j=0;j<1+3*SCHRO_MAX_TRANSFORM_DEPTH;j++){
            frame->quant_index[comp][j] = 0;
          }
        }

        if (params->num_refs > 0) {
          frame->ref_frame0 = schro_encoder_reference_get (encoder,
              frame->reference_frame_number[0]);
          schro_encoder_frame_ref (frame->ref_frame0);
        } else {
          frame->ref_frame0 = NULL;
        }

        SCHRO_DEBUG("queueing %d", frame->frame_number);

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

void
reduce_fraction (int *n, int *d)
{
  static const int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
    91 };
  int i;
  int p;

  SCHRO_DEBUG("reduce %d/%d", *n, *d);
  for(i=0;i<sizeof(primes)/sizeof(primes[0]);i++){
    p = primes[i];
    while (*n % p == 0 && *d % p == 0) {
      *n /= p;
      *d /= p;
    }
  }
  SCHRO_DEBUG("to %d/%d", *n, *d);
}

int
multdiv64 (int a, int b, int c)
{
  long long x;

  x = a;
  x *= b;
  x /= c;

  return (int)x;
}

int
schro_encoder_engine_lowdelay (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;
  int n_slices;
  int num;
  int denom;
  int bytes_per_picture;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_analyse_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        schro_engine_check_new_access_unit (encoder, frame);

        frame->presentation_frame = frame->frame_number;

        frame->slot = frame->frame_number;

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        params->is_lowdelay = TRUE;

        /* FIXME should be parameters */
        params->slice_width_exp = 4;
        params->slice_height_exp = 4;
        init_params (frame);
        //schro_params_init_lowdelay_quantisers(params);

        bytes_per_picture = multdiv64(encoder->prefs[SCHRO_PREF_BITRATE],
            encoder->video_format.frame_rate_denominator,
            encoder->video_format.frame_rate_numerator * 8);
        n_slices = (params->iwt_luma_width>>params->slice_width_exp) *
          (params->iwt_luma_height>>params->slice_height_exp);
        num = bytes_per_picture;
        denom = n_slices;
        SCHRO_ASSERT(denom != 0);
        reduce_fraction (&num, &denom);
        params->slice_bytes_num = num;
        params->slice_bytes_denom = denom;

        frame->state = SCHRO_ENCODER_FRAME_STATE_PREDICT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_predict_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_encode_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        frame->state = SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_reconstruct_picture, frame);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        frame->state = SCHRO_ENCODER_FRAME_STATE_POSTANALYSE;
        frame->busy = TRUE;
        schro_async_run_locked (encoder->async,
            (void (*)(void *))schro_encoder_postanalyse_picture, frame);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

