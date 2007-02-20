
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schrointernal.h>
#include <schroedinger/schroencoder.h>
#include <liboil/liboil.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


int
schro_encoder_engine_intra_only (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroFrame *encode_frame;

  if (encoder->au_frame == -1 ||
      encoder->next_frame - encoder->au_frame >= encoder->au_distance) {
    SchroBuffer *buffer;

    encoder->au_frame = encoder->next_frame;
    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot,
        encoder->next_frame - 1);
    encoder->next_slot++;
  }

  encode_frame = schro_encoder_frame_queue_get (encoder, encoder->next_frame);
  if (encode_frame == NULL) {
    if (encoder->end_of_stream && !encoder->end_of_stream_handled) {
      SchroBuffer *buffer;

      buffer = schro_encoder_encode_end_of_stream (encoder);

      schro_encoder_output_push (encoder, buffer, encoder->next_slot,
          encoder->next_frame - 1);
      encoder->next_slot++;
      encoder->end_of_stream_handled = TRUE;
      return TRUE;
    } else {
      return FALSE;
    }
  }

  task = schro_encoder_task_new (encoder);

  task->encode_frame = encode_frame;
  task->frame_number = encoder->next_frame;
  task->presentation_frame = encoder->next_frame;

  schro_encoder_frame_queue_remove (encoder, encoder->next_frame);
  encoder->next_frame++;

  task->slot = encoder->next_slot;
  encoder->next_slot++;

  /* FIXME 4:2:0 assumption */
  task->outbuffer_size = encoder->video_format.width *
    encoder->video_format.height * 3 / 2;
  task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

  task->bits = schro_bits_new ();
  schro_bits_encode_init (task->bits, task->outbuffer);

  /* set up params */
  params = &task->params;
  params->video_format = &encoder->video_format;
  params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTRA_WAVELET];
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];
  schro_params_set_default_codeblock (params);

  params->num_refs = 0;
  params->have_global_motion = FALSE;
  params->xblen_luma = 12;
  params->yblen_luma = 12;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->mv_precision = 0;
  params->picture_pred_mode = 0;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;

  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  schro_async_run (encoder->async,
      (void (*)(void *))schro_encoder_encode_picture, task);

  return TRUE;
}

int
schro_encoder_engine_backref (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  SchroFrame *encode_frame;

  if (encoder->au_frame == -1) {
    SchroBuffer *buffer;

    encoder->au_frame = 0;

    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot, -1);
    encoder->next_slot++;
  }

  encode_frame = schro_encoder_frame_queue_get (encoder, encoder->next_frame);
  if (encode_frame == NULL) {
    if (encoder->end_of_stream) {
      SchroBuffer *buffer;

      buffer = schro_encoder_encode_end_of_stream (encoder);

      schro_encoder_output_push (encoder, buffer, encoder->next_slot,
          encoder->next_frame - 1);
      encoder->next_slot++;
      return TRUE;
    } else {
      return FALSE;
    }
  }

  if (encoder->last_ref >= 0 &&
      encoder->next_frame - encoder->last_ref < encoder->ref_distance) {
    if (!schro_encoder_reference_get (encoder, encoder->last_ref)) {
      return FALSE;
    }
  }

  task = schro_encoder_task_new (encoder);
  task->encode_frame = encode_frame;
  task->is_ref = FALSE;
  if (encoder->next_frame - encoder->au_frame >= encoder->au_distance) {
    SchroBuffer *buffer;

    encoder->au_frame = encoder->next_frame;
    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot,
        task->frame_number - 1);
    encoder->next_slot++;
    
    task->is_ref = TRUE;
  }

  if (encoder->last_ref == -1 || 
      encoder->next_frame - encoder->last_ref >= encoder->ref_distance) {
    task->is_ref = TRUE;
  }

  schro_encoder_frame_queue_remove (encoder, encoder->next_frame);
  task->frame_number = encoder->next_frame;
  encoder->next_frame++;
  task->presentation_frame = task->frame_number;

  task->slot = encoder->next_slot;
  encoder->next_slot++;

  /* FIXME 4:2:0 assumption */
  task->outbuffer_size = encoder->video_format.width *
    encoder->video_format.height * 3 / 2;
  task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

  task->bits = schro_bits_new ();
  schro_bits_encode_init (task->bits, task->outbuffer);

  /* set up params */
  params = &task->params;
  if (task->is_ref) {
    params->num_refs = 0;
    if (task->frame_number > 0) {
      task->retire[0] = encoder->last_ref;
      task->n_retire = 1;
    } else {
      task->n_retire = 0;
    }
    encoder->last_ref = task->frame_number;
  } else {
    params->num_refs = 1;
    task->reference_frame_number[0] = encoder->last_ref;
    task->n_retire = 0;
  }
  params->video_format = &encoder->video_format;
  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTER_WAVELET];
  } else {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTRA_WAVELET];
  }
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];
  schro_params_set_default_codeblock (params);

  params->have_global_motion = FALSE;
  params->xblen_luma = 12;
  params->yblen_luma = 12;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->mv_precision = 0;
  params->picture_pred_mode = 0;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;

  /* calculations */
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  if (params->num_refs > 0) {
    task->ref_frame0 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[0]);
  } else {
    task->ref_frame0 = NULL;
  }
  if (params->num_refs > 1) {
    task->ref_frame1 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[1]);
  } else {
    task->ref_frame1 = NULL;
  }
  if (task->is_ref) {
    task->dest_ref = schro_encoder_reference_add (encoder);
  } else {
    task->dest_ref = NULL;
  }

  schro_async_run (encoder->async,
      (void (*)(void *))schro_encoder_encode_picture, task);

  return TRUE;
}

int
schro_encoder_engine_tworef (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  int type;

  if (encoder->au_frame == -1) {
    SchroBuffer *buffer;

    encoder->au_frame = 0;

    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot, -1);
    encoder->next_slot++;
  }

  SCHRO_DEBUG("iterate: %d %d %d", encoder->next_frame, encoder->last_ref,
    encoder->next_ref);
  task = schro_encoder_task_new (encoder);
  params = &task->params;
  if (encoder->last_ref == -1) {
    type = 0;
    task->frame_number = 0;
    task->is_ref = TRUE;
    task->n_retire = 0;
    params->num_refs = 0;
    task->presentation_frame = 0;
  } else if (encoder->next_ref == -1) {
    type = 1;
    task->frame_number = encoder->last_ref + encoder->ref_distance;
    task->is_ref = TRUE;
    params->num_refs = 0;
    task->presentation_frame = encoder->next_frame - 1;
  } else {
    type = 2;
    task->frame_number = encoder->next_frame;
    task->is_ref = FALSE;
    params->num_refs = 2;
    task->reference_frame_number[0] = encoder->last_ref;
    task->reference_frame_number[1] = encoder->next_ref;
    task->presentation_frame = encoder->next_frame;
  }

  task->encode_frame = schro_encoder_frame_queue_get (encoder,
      task->frame_number);
  if (task->encode_frame == NULL) {
    schro_encoder_task_free (task);
    return FALSE;
  }

  if (type == 1 &&
      task->frame_number - encoder->au_frame >= encoder->au_distance) {
    SchroBuffer *buffer;

    encoder->au_frame = task->frame_number;
    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot,
        task->frame_number - 1);
    encoder->next_slot++;
  }

  schro_encoder_frame_queue_remove (encoder, task->frame_number);

  switch(type) {
    case 0:
      params->num_refs = 0;
      task->n_retire = 0;
      encoder->last_ref = 0;
      encoder->next_frame = 1;
      break;
    case 1:
      params->num_refs = 0;
      encoder->next_ref = task->frame_number;
      task->n_retire = 0;
      break;
    case 2:
      params->num_refs = 2;
      task->reference_frame_number[0] = encoder->last_ref;
      task->reference_frame_number[1] = encoder->next_ref;
      encoder->next_frame++;
      if (encoder->next_frame == encoder->next_ref) {
        task->n_retire = 1;
        task->retire[0] = encoder->last_ref;
        encoder->last_ref = encoder->next_ref;
        encoder->next_ref = -1;
        encoder->next_frame++;
      }
      task->presentation_frame = task->frame_number;
      break;
  }

  task->slot = encoder->next_slot;
  encoder->next_slot++;

  /* FIXME 4:2:0 assumption */
  task->outbuffer_size = encoder->video_format.width *
    encoder->video_format.height * 3 / 2;
  task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

  task->bits = schro_bits_new ();
  schro_bits_encode_init (task->bits, task->outbuffer);

  /* set up params */
  params->video_format = &encoder->video_format;
  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTER_WAVELET];
  } else {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTRA_WAVELET];
  }
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];
  schro_params_set_default_codeblock (params);

  params->have_global_motion = TRUE;
  params->xblen_luma = 12;
  params->yblen_luma = 12;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->mv_precision = 0;
  params->picture_pred_mode = 0;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;

  /* calculations */
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  if (params->num_refs > 0) {
    task->ref_frame0 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[0]);
  } else {
    task->ref_frame0 = NULL;
  }
  if (params->num_refs > 1) {
    task->ref_frame1 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[1]);
  } else {
    task->ref_frame1 = NULL;
  }
  if (task->is_ref) {
    task->dest_ref = schro_encoder_reference_add (encoder);
  } else {
    task->dest_ref = NULL;
  }

  schro_async_run (encoder->async,
      (void (*)(void *))schro_encoder_encode_picture, task);

  return TRUE;
}

int
schro_encoder_engine_fourref (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderTask *task;
  int type;

  if (encoder->au_frame == -1) {
    SchroBuffer *buffer;

    encoder->au_frame = 0;

    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot, -1);
    encoder->next_slot++;
  }

  SCHRO_DEBUG("iterate: %d %d %d %d", encoder->next_frame, encoder->last_ref,
    encoder->next_ref, encoder->mid1_ref);
  task = schro_encoder_task_new (encoder);
  params = &task->params;
  if (encoder->last_ref == -1) {
    type = 0;
    task->frame_number = 0;
    task->is_ref = TRUE;
    task->n_retire = 0;
    params->num_refs = 0;
  } else if (encoder->next_ref == -1) {
    type = 1;
    task->frame_number = encoder->last_ref + encoder->ref_distance;
    task->is_ref = TRUE;
    params->num_refs = 0;
  } else if (encoder->mid1_ref == -1) {
    type = 2;
    task->frame_number = encoder->last_ref + encoder->ref_distance/2;
    task->is_ref = TRUE;
    params->num_refs = 2;
    task->reference_frame_number[0] = encoder->last_ref;
    task->reference_frame_number[1] = encoder->next_ref;
  } else {
    type = 3;
    task->frame_number = encoder->next_frame;
    task->is_ref = FALSE;
    params->num_refs = 2;
    if (task->frame_number < encoder->mid1_ref) {
      task->reference_frame_number[0] = encoder->last_ref;
      task->reference_frame_number[1] = encoder->mid1_ref;
    } else {
      task->reference_frame_number[0] = encoder->mid1_ref;
      task->reference_frame_number[1] = encoder->next_ref;
    }
  }

  task->encode_frame = schro_encoder_frame_queue_get (encoder,
      task->frame_number);
  if (task->encode_frame == NULL) {
    schro_encoder_task_free (task);
    return FALSE;
  }

#if 0
  if (encoder->next_frame - encoder->au_frame >= encoder->au_distance) {
    SchroBuffer *buffer;

    encoder->au_frame = encoder->next_frame;

    buffer = schro_encoder_encode_access_unit (encoder);

    schro_encoder_output_push (encoder, buffer, encoder->next_slot,
        task->frame_number - 1);
    encoder->next_slot++;
    
    task->is_ref = TRUE;
  }
#endif

  schro_encoder_frame_queue_remove (encoder, task->frame_number);

  switch(type) {
    case 0:
      task->n_retire = 0;
      encoder->last_ref = 0;
      encoder->next_frame = 1;
      break;
    case 1:
      encoder->next_ref = task->frame_number;
      task->n_retire = 0;
      break;
    case 2:
      encoder->mid1_ref = task->frame_number;
      task->n_retire = 0;
      break;
    case 3:
      encoder->next_frame++;
      if (encoder->next_frame == encoder->mid1_ref) {
        encoder->next_frame++;
      }
      if (encoder->next_frame == encoder->next_ref) {
        task->n_retire = 2;
        task->retire[0] = encoder->last_ref;
        task->retire[1] = encoder->mid1_ref;
        encoder->last_ref = encoder->next_ref;
        encoder->next_ref = -1;
        encoder->mid1_ref = -1;
        encoder->next_frame++;
      }
      break;
  }

  task->presentation_frame = task->frame_number;

  task->slot = encoder->next_slot;
  encoder->next_slot++;

  /* FIXME 4:2:0 assumption */
  task->outbuffer_size = encoder->video_format.width *
    encoder->video_format.height * 3 / 2;
  task->outbuffer = schro_buffer_new_and_alloc (task->outbuffer_size);

  task->bits = schro_bits_new ();
  schro_bits_encode_init (task->bits, task->outbuffer);

  /* set up params */
  params->video_format = &encoder->video_format;
  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTER_WAVELET];
  } else {
    params->wavelet_filter_index = encoder->prefs[SCHRO_PREF_INTRA_WAVELET];
  }
  params->transform_depth = encoder->prefs[SCHRO_PREF_TRANSFORM_DEPTH];
  schro_params_set_default_codeblock (params);

  params->have_global_motion = FALSE;
  params->xblen_luma = 12;
  params->yblen_luma = 12;
  params->xbsep_luma = 8;
  params->ybsep_luma = 8;
  params->mv_precision = 0;
  params->picture_pred_mode = 0;
  params->picture_weight_1 = 1;
  params->picture_weight_2 = 1;

  /* calculations */
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  if (params->num_refs > 0) {
    task->ref_frame0 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[0]);
  } else {
    task->ref_frame0 = NULL;
  }
  if (params->num_refs > 1) {
    task->ref_frame1 = schro_encoder_reference_get (encoder,
        task->reference_frame_number[1]);
  } else {
    task->ref_frame1 = NULL;
  }
  if (task->is_ref) {
    task->dest_ref = schro_encoder_reference_add (encoder);
  } else {
    task->dest_ref = NULL;
  }

  schro_async_run (encoder->async,
      (void (*)(void *))schro_encoder_encode_picture, task);

  return TRUE;
}

