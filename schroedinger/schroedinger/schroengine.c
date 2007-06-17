
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schroencoder.h>
#include <liboil/liboil.h>
//#include <stdlib.h>
//#include <string.h>
//#include <stdio.h>


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
init_params (SchroEncoderTask *task)
{
  SchroParams *params = &task->params;
  SchroEncoder *encoder = task->encoder;

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
  schro_params_init_subbands (params, task->subbands,
      task->iwt_frame->components[0].stride,
      task->iwt_frame->components[1].stride);
  schro_encoder_choose_quantisers (task);
}

int
schro_encoder_engine_intra_only (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) {
      continue;
    }

    schro_engine_check_new_access_unit (encoder, frame);

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    task->slot = encoder->next_slot;
    encoder->next_slot++;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    init_params (task);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}


int
schro_encoder_engine_backref (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_ref;

    frame = encoder->frame_queue->elements[i].data;
    //SCHRO_ERROR("backref i=%d picture=%d", i, frame->frame_number);

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) {
      continue;
    }

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

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    task->slot = encoder->next_slot;
    encoder->next_slot++;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    task->is_ref = is_ref;
    if (task->is_ref) {
      params->num_refs = 0;
      if (task->frame_number > 0) {
        frame->retire = encoder->last_ref;
        frame->n_retire = 1;
      } else {
        frame->n_retire = 0;
      }
      encoder->last_ref = task->frame_number;
    } else {
      params->num_refs = 1;
      task->reference_frame_number[0] = encoder->last_ref;
      frame->n_retire = 0;
    }

    task->n_retire = frame->n_retire;
    task->retire[0] = frame->retire;

    init_params (task);

    if (params->num_refs > 0) {
      task->ref_frame0 = schro_encoder_reference_get (encoder,
          task->reference_frame_number[0]);
      schro_encoder_frame_ref (task->ref_frame0);
    } else {
      task->ref_frame0 = NULL;
    }

    SCHRO_DEBUG("queueing %d", task->frame_number);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}

int
schro_encoder_engine_backref2 (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
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

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    task->slot = encoder->next_slot;
    encoder->next_slot++;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    if (is_intra) {
      task->is_ref = TRUE;
      params->num_refs = 0;
      if (task->frame_number > 0) {
        frame->retire = encoder->last_ref;
        frame->n_retire = 1;
      } else {
        frame->n_retire = 0;
      }
      encoder->last_ref = task->frame_number;
      encoder->mid1_ref = task->frame_number;
    } else if (frame->frame_number >= encoder->last_ref + 4) {
      task->is_ref = TRUE;
      params->num_refs = 1;
      task->reference_frame_number[0] = encoder->last_ref;
      frame->retire = encoder->last_ref;
      frame->n_retire = 1;
      encoder->last_ref = task->frame_number;
    } else {
      task->is_ref = FALSE;
      params->num_refs = 1;
      task->reference_frame_number[0] = encoder->last_ref;
      frame->n_retire = 0;
    }

    task->n_retire = frame->n_retire;
    task->retire[0] = frame->retire;

    init_params (task);

    if (params->num_refs > 0) {
      task->ref_frame0 = schro_encoder_reference_get (encoder,
          task->reference_frame_number[0]);
      schro_encoder_frame_ref (task->ref_frame0);
    } else {
      task->ref_frame0 = NULL;
    }

    SCHRO_DEBUG("queueing %d", task->frame_number);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}

int
schro_encoder_engine_tworef (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroEncoderFrame *frame;
  SchroEncoderFrame *f;
  int i;

  SCHRO_DEBUG("engine iteration");

  encoder->quantiser_engine = 1;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("%d: picture %d is in state %d", i, frame->frame_number,
        frame->state);
    if (frame->state == SCHRO_ENCODER_FRAME_STATE_NEW) {
      break;
    }
  }
  SCHRO_DEBUG("i is %d", i);
  for(;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
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
        frame->retire = encoder->last_ref;
      }
      encoder->last_ref = frame->frame_number;
    } else {
      int ref_slot;
      int j;
      int gop_length;

      gop_length = 4;

      if (i + gop_length >= encoder->frame_queue->n) {
        if (encoder->end_of_stream) {
          gop_length = encoder->frame_queue->n - i;
        } else {
          break;
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
          f->retire = frame->frame_number - 1;
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
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++){
    frame = encoder->frame_queue->elements[i].data;

    SCHRO_DEBUG("checking %d...", frame->frame_number);

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_ENGINE_1) {
      SCHRO_DEBUG("wrong state");
      continue;
    }

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

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    task->slot = frame->slot;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    params->num_refs = frame->num_refs;
    params->xbsep_luma = 8;
    params->xblen_luma = 12;
    params->ybsep_luma = 8;
    params->yblen_luma = 12;

    task->is_ref = frame->is_ref;
    if (frame->num_refs > 0) {
      task->reference_frame_number[0] = frame->picture_number_ref0;
    }
    if (frame->num_refs > 1) {
      task->reference_frame_number[1] = frame->picture_number_ref1;
    }

    task->n_retire = frame->n_retire;
    if (frame->n_retire) {
      task->retire[0] = frame->retire;
    }

    init_params (task);

    if (params->num_refs > 0) {
      task->ref_frame0 = schro_encoder_reference_get (encoder,
          task->reference_frame_number[0]);
      schro_encoder_frame_ref (task->ref_frame0);
    } else {
      task->ref_frame0 = NULL;
    }
    if (params->num_refs > 1) {
      task->ref_frame1 = schro_encoder_reference_get (encoder,
          task->reference_frame_number[1]);
      schro_encoder_frame_ref (task->ref_frame1);
    } else {
      task->ref_frame1 = NULL;
    }

    SCHRO_DEBUG("queueing %d", task->frame_number);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}


static struct {
  int type;
  int depth;
} test_wavelet_types[] = {
  { SCHRO_WAVELET_DESL_9_3, 1 },
  { SCHRO_WAVELET_DESL_9_3, 2 },
  { SCHRO_WAVELET_DESL_9_3, 3 },
  { SCHRO_WAVELET_DESL_9_3, 4 },
  { SCHRO_WAVELET_5_3, 1 },
  { SCHRO_WAVELET_5_3, 2 },
  { SCHRO_WAVELET_5_3, 3 },
  { SCHRO_WAVELET_5_3, 4 },
  { SCHRO_WAVELET_13_5, 1 },
  { SCHRO_WAVELET_13_5, 2 },
  { SCHRO_WAVELET_13_5, 3 },
  { SCHRO_WAVELET_13_5, 4 },
  { SCHRO_WAVELET_HAAR_0, 1 },
  { SCHRO_WAVELET_HAAR_0, 2 },
  { SCHRO_WAVELET_HAAR_0, 3 },
  { SCHRO_WAVELET_HAAR_0, 4 },
  { SCHRO_WAVELET_HAAR_1, 1 },
  { SCHRO_WAVELET_HAAR_1, 2 },
  { SCHRO_WAVELET_HAAR_1, 3 },
  { SCHRO_WAVELET_HAAR_1, 4 },
  { SCHRO_WAVELET_HAAR_2, 1 },
  { SCHRO_WAVELET_HAAR_2, 2 },
  //{ SCHRO_WAVELET_HAAR_2, 3 },
  //{ SCHRO_WAVELET_HAAR_2, 4 },
  { SCHRO_WAVELET_FIDELITY, 1 },
  { SCHRO_WAVELET_FIDELITY, 2 },
  { SCHRO_WAVELET_FIDELITY, 3 },
  //{ SCHRO_WAVELET_FIDELITY, 4 },
  { SCHRO_WAVELET_DAUB_9_7, 1 },
  { SCHRO_WAVELET_DAUB_9_7, 2 },
  { SCHRO_WAVELET_DAUB_9_7, 3 },
  //{ SCHRO_WAVELET_DAUB_9_7, 4 }
};

int
schro_encoder_engine_test_intra (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroEncoderFrame *frame;
  int i;
  int j;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) {
      continue;
    }

    schro_engine_check_new_access_unit (encoder, frame);

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    task->slot = encoder->next_slot;
    encoder->next_slot++;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    j = frame->frame_number % ARRAY_SIZE(test_wavelet_types);
    encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = test_wavelet_types[j].type;
    encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = test_wavelet_types[j].depth;
    init_params (task);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}

int
schro_encoder_engine_lossless (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroEncoderFrame *frame;
  int i;
  int j;

  for(i=0;i<encoder->frame_queue->n;i++) {
    int is_ref;

    frame = encoder->frame_queue->elements[i].data;
    //SCHRO_ERROR("backref i=%d picture=%d", i, frame->frame_number);

    if (frame->state != SCHRO_ENCODER_FRAME_STATE_NEW) {
      continue;
    }

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
        return FALSE;
      }
    }

    task = schro_encoder_task_new (encoder);

    frame->state = SCHRO_ENCODER_FRAME_STATE_ENCODING;
    task->encoder_frame = frame;

    task->encode_frame = frame->original_frame;
    task->frame_number = frame->frame_number;
    task->presentation_frame = frame->frame_number;

    frame->slot = encoder->next_slot;
    task->slot = encoder->next_slot;
    encoder->next_slot++;

    task->outbuffer_size =
      schro_engine_pick_output_buffer_size (encoder, frame);
    task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

    /* set up params */
    params = &task->params;
    task->is_ref = is_ref;
    if (task->is_ref) {
      params->num_refs = 0;
      if (task->frame_number > 0) {
        frame->retire = encoder->last_ref;
        frame->n_retire = 1;
      } else {
        frame->n_retire = 0;
      }
      encoder->last_ref = task->frame_number;
    } else {
      params->num_refs = 1;
      task->reference_frame_number[0] = encoder->last_ref;
      frame->n_retire = 0;
    }

    task->n_retire = frame->n_retire;
    task->retire[0] = frame->retire;

    encoder->prefs[SCHRO_PREF_INTRA_WAVELET] = SCHRO_WAVELET_HAAR_0;
    encoder->prefs[SCHRO_PREF_INTER_WAVELET] = SCHRO_WAVELET_HAAR_0;
    encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH] = 3;
    init_params (task);

    params->xbsep_luma = 8;
    params->xblen_luma = 8;
    params->ybsep_luma = 8;
    params->yblen_luma = 8;

    for(j=0;j<19;j++){
      task->subbands[j].quant_index = 0;
    }

    if (params->num_refs > 0) {
      task->ref_frame0 = schro_encoder_reference_get (encoder,
          task->reference_frame_number[0]);
      schro_encoder_frame_ref (task->ref_frame0);
    } else {
      task->ref_frame0 = NULL;
    }

    SCHRO_DEBUG("queueing %d", task->frame_number);

    schro_async_run (encoder->async,
        (void (*)(void *))schro_encoder_encode_picture, task);

    return TRUE;
  }

  return FALSE;
}

