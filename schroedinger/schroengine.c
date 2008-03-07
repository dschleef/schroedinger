
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>

#include <math.h>

int schro_engine_get_scene_change_score (SchroEncoder *encoder, int i);
void schro_encoder_calculate_allocation (SchroEncoderFrame *frame);
static void choose_quantisers (SchroEncoderFrame *frame);

static void
schro_engine_check_new_access_unit(SchroEncoder *encoder,
    SchroEncoderFrame *frame)
{
  if (encoder->au_frame == -1 ||
      frame->frame_number >= encoder->au_frame + encoder->au_distance) {
    frame->start_access_unit = TRUE;
    encoder->au_frame = frame->frame_number;
  }
}

static void
handle_gop (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *ref2;
  SchroEncoderFrame *f;
  int j;
  int gop_length;
  schro_bool intra_start;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);

  gop_length = 4;
  SCHRO_DEBUG("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture + gop_length - 1, i);

  if (i + gop_length >= encoder->frame_queue->n) {
    if (encoder->end_of_stream) {
      gop_length = encoder->frame_queue->n - i;
    } else {
      SCHRO_DEBUG("not enough pictures in queue");
      return;
    }
  }

  intra_start = frame->start_access_unit;
  for (j = 0; j < gop_length; j++) {
    /* FIXME set the gop length correctly for IBBBP */
    f = encoder->frame_queue->elements[i+j].data;

    if (f->busy || f->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) {
      SCHRO_DEBUG("picture %d not ready", i + j);
      return;
    }

    schro_engine_get_scene_change_score (encoder, i+j);

    schro_dump (SCHRO_DUMP_SCENE_CHANGE, "%d %g %g\n",
        f->frame_number, f->scene_change_score,
        f->average_luma);
    SCHRO_DEBUG("scene change score %g", f->scene_change_score);

    if (j==0 && f->scene_change_score > encoder->magic_scene_change_threshold) {
      intra_start = TRUE;
    }
    if (j>=1 && f->scene_change_score > encoder->magic_scene_change_threshold) {
      /* probably a scene change.  terminate gop */
      gop_length = j;
    }
  }

  SCHRO_DEBUG("gop length %d", gop_length);

  if (gop_length == 1) {
    frame->is_ref = FALSE;
    frame->num_refs = 0;
    SCHRO_DEBUG("preparing %d as intra non-ref", frame->frame_number);
    frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
    frame->slot = encoder->next_slot++;
    frame->presentation_frame = frame->frame_number;
    frame->picture_weight = encoder->magic_bailout_weight;
  } else {
    if (intra_start) {
      /* IBBBP */
      frame->is_ref = TRUE;
      frame->num_refs = 0;
      SCHRO_DEBUG("preparing %d as intra ref", frame->frame_number);
      frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
      frame->slot = encoder->next_slot++;
      frame->presentation_frame = frame->frame_number;
      //frame->picture_weight = 1 + (gop_length - 1) * 0.6;
      frame->picture_weight = encoder->magic_keyframe_weight;
      if (encoder->last_ref != -1) {
        frame->retired_picture_number = encoder->last_ref;
      }
      encoder->last_ref = encoder->last_ref2;
      encoder->last_ref2 = frame->frame_number;

      f = encoder->frame_queue->elements[i+gop_length-1].data;
      f->is_ref = TRUE;
      f->num_refs = 1;
      f->picture_number_ref0 = frame->frame_number;
      f->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
      SCHRO_DEBUG("preparing %d as inter ref (%d)", f->frame_number,
          f->picture_number_ref0);
      f->slot = encoder->next_slot++;
      f->presentation_frame = f->frame_number;
      f->picture_weight = encoder->magic_inter_p_weight;
      //f->picture_weight += (gop_length - 2) * (1 - encoder->magic_inter_b_weight);
      if (encoder->last_ref != -1) {
        f->retired_picture_number = encoder->last_ref;
      }
      encoder->last_ref = encoder->last_ref2;
      encoder->last_ref2 = f->frame_number;
      ref2 = f;

      for (j = 1; j < gop_length - 1; j++) {
        f = encoder->frame_queue->elements[i+j].data;
        f->is_ref = FALSE;
        f->num_refs = 2;
        f->picture_number_ref0 = frame->frame_number;
        f->picture_number_ref1 = ref2->frame_number;
        f->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
        SCHRO_DEBUG("preparing %d as inter (%d)", f->frame_number,
            f->picture_number_ref0);
        f->slot = encoder->next_slot++;
        f->presentation_frame = f->frame_number;
        f->picture_weight = encoder->magic_inter_b_weight;
      }
    } else {
      /* BBBP */
      f = encoder->frame_queue->elements[i+gop_length-1].data;
      f->is_ref = TRUE;
      f->num_refs = 1;
      f->picture_number_ref0 = encoder->last_ref2;
      f->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
      SCHRO_DEBUG("preparing %d as inter ref (%d)", f->frame_number,
          f->picture_number_ref0);
      f->slot = encoder->next_slot++;
      f->presentation_frame = f->frame_number;
      f->picture_weight = encoder->magic_inter_p_weight;
      //f->picture_weight += (gop_length - 1) * (1 - encoder->magic_inter_b_weight);
      if (encoder->last_ref != -1) {
        f->retired_picture_number = encoder->last_ref;
      }
      encoder->last_ref = encoder->last_ref2;
      encoder->last_ref2 = f->frame_number;

      for (j = 0; j < gop_length - 1; j++) {
        f = encoder->frame_queue->elements[i+j].data;
        f->is_ref = FALSE;
        f->num_refs = 2;
        f->picture_number_ref0 = encoder->last_ref;
        f->picture_number_ref1 = encoder->last_ref2;
        f->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
        SCHRO_DEBUG("preparing %d as inter (%d)", f->frame_number,
            f->picture_number_ref0);
        f->slot = encoder->next_slot++;
        f->presentation_frame = f->frame_number;
        f->picture_weight = encoder->magic_inter_b_weight;
      }
    }
  }

  encoder->gop_picture += gop_length;
}

static void
handle_gop_backref (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *f;
  int j;
  int gop_length;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);

  gop_length = 4;
  SCHRO_DEBUG("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture + gop_length - 1, i);

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

    schro_engine_get_scene_change_score (encoder, i+j);

    schro_dump (SCHRO_DUMP_SCENE_CHANGE, "%d %g %g\n",
        f->frame_number, f->scene_change_score,
        f->average_luma);

    if (j>=1 && f->scene_change_score > encoder->magic_scene_change_threshold) {
      /* probably a scene change.  terminate gop */
      gop_length = j;
    }
  }

  SCHRO_DEBUG("gop length %d", gop_length);

  frame->is_ref = TRUE;
  frame->num_refs = 0;
  SCHRO_DEBUG("preparing %d as intra ref", frame->frame_number);
  frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
  frame->slot = encoder->next_slot++;
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1 + (gop_length - 1) * (1 - encoder->magic_inter_b_weight);
  if (encoder->last_ref != -1) {
    frame->retired_picture_number = encoder->last_ref;
  }
  encoder->last_ref = frame->frame_number;

  for (j = 1; j < gop_length; j++) {
    f = encoder->frame_queue->elements[i+j].data;
    f->is_ref = FALSE;
    f->num_refs = 1;
    f->picture_number_ref0 = frame->frame_number;
    f->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
    SCHRO_DEBUG("preparing %d as inter (%d)", f->frame_number,
        f->picture_number_ref0);
    f->slot = encoder->next_slot++;
    f->presentation_frame = f->frame_number;
    f->picture_weight = encoder->magic_inter_b_weight;
  }

  encoder->gop_picture += gop_length;
}

static int
check_refs (SchroEncoderFrame *frame)
{
  if (frame->num_refs == 0) return TRUE;

  frame->ref_frame0 = schro_encoder_reference_get (frame->encoder,
      frame->picture_number_ref0);
  if (frame->ref_frame0 == NULL) return FALSE;

  if (frame->num_refs == 1) {
    schro_encoder_frame_ref (frame->ref_frame0);
    return TRUE;
  }

  frame->ref_frame1 = schro_encoder_reference_get (frame->encoder,
      frame->picture_number_ref1);
  if (frame->ref_frame1 == NULL) return FALSE;

  schro_encoder_frame_ref (frame->ref_frame0);
  schro_encoder_frame_ref (frame->ref_frame1);
  return TRUE;
}

int
schro_engine_get_scene_change_score (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame1;
  SchroEncoderFrame *frame2;
  double luma;

  frame1 = encoder->frame_queue->elements[i].data;
  if (frame1->have_scene_change_score) return TRUE;

  /* FIXME Just because it's the first picture in the queue doesn't
   * mean it's a scene change.  (But likely is.) */
  if (i == 0) {
    frame1->scene_change_score = 1.0;
    frame1->have_scene_change_score = TRUE;
    return TRUE;
  }

  frame2 = encoder->frame_queue->elements[i-1].data;
  if (frame2->state == SCHRO_ENCODER_FRAME_STATE_ANALYSE && frame2->busy) {
    return FALSE;
  }

  SCHRO_DEBUG("%g %g", frame1->average_luma, frame2->average_luma);

  luma = frame1->average_luma - 16.0;
  if (luma > 0.01) {
    double mse[3];
    schro_frame_mean_squared_error (frame1->downsampled_frames[3],
        frame2->downsampled_frames[3], mse);
    frame1->scene_change_score = mse[0] / (luma * luma);
  } else {
    frame1->scene_change_score = 1.0;
  }

  SCHRO_DEBUG("scene change score %g", frame1->scene_change_score);

  frame1->have_scene_change_score = TRUE;
  return TRUE;
}


static int
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
  SchroVideoFormat *video_format = params->video_format;

  params->video_format = &encoder->video_format;

  schro_params_init (params, params->video_format->index);

  if (encoder->enable_noarith && frame->num_refs == 0) {
    params->is_noarith = TRUE;
  }

  if (params->num_refs > 0) {
    params->wavelet_filter_index = encoder->inter_wavelet;
  } else {
    params->wavelet_filter_index = encoder->intra_wavelet;
  }
  params->transform_depth = encoder->transform_depth;

  switch (encoder->motion_block_size) {
    case 0:
      if (video_format->width * video_format->height >= 1920*1080) {
        params->xbsep_luma = 16;
        params->ybsep_luma = 16;
      } else if (video_format->width * video_format->height >= 960 * 540) {
        params->xbsep_luma = 12;
        params->ybsep_luma = 12;
      } else {
        params->xbsep_luma = 8;
        params->ybsep_luma = 8;
      }
      break;
    case 1:
      params->xbsep_luma = 8;
      params->ybsep_luma = 8;
      break;
    case 2:
      params->xbsep_luma = 12;
      params->ybsep_luma = 12;
      break;
    case 3:
      params->xbsep_luma = 16;
      params->ybsep_luma = 16;
      break;
  }
  switch (encoder->motion_block_overlap) {
    case 1:
      params->xblen_luma = params->xbsep_luma;
      params->yblen_luma = params->ybsep_luma;
      break;
    case 0:
    case 2:
      params->xblen_luma = (params->xbsep_luma * 3 / 2) & (~3);
      params->yblen_luma = (params->ybsep_luma * 3 / 2) & (~3);
      break;
    case 3:
      params->xblen_luma = 2 * params->xbsep_luma;
      params->yblen_luma = 2 * params->ybsep_luma;
      break;
  }

  params->mv_precision = encoder->mv_precision;
  //params->have_global_motion = TRUE;
  params->codeblock_mode_index = 0;
  
  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);
}

static void
init_small_codeblocks (SchroParams *params)
{
  int i;
  int shift;

  params->horiz_codeblocks[0] = 1;
  params->vert_codeblocks[0] = 1;
  for(i=1;i<params->transform_depth+1;i++){
    shift = params->transform_depth + 1 - i;
    /* These values are empirically derived from fewer than 2 test results */
    params->horiz_codeblocks[i] = (params->iwt_luma_width >> shift) / 5;
    params->vert_codeblocks[i] = (params->iwt_luma_height >> shift) / 5;
    SCHRO_DEBUG("codeblocks %d %d %d", i, params->horiz_codeblocks[i],
        params->vert_codeblocks[i]);
  }
}

static int
get_residual_alloc (SchroEncoder *encoder, int buffer_level, double picture_weight)
{
  double x;
  int bits;

  x = (double)buffer_level/encoder->buffer_size;

  if (picture_weight == 0) {
    picture_weight = 1.0;
  }
  bits = rint (x * encoder->bits_per_picture * picture_weight *
      encoder->magic_allocation_scale);

  if (bits > buffer_level) bits = buffer_level;

  return bits;
}

static int
get_mc_alloc (SchroEncoderFrame *frame)
{
  return 10 * frame->params.x_num_blocks * frame->params.y_num_blocks / 16;
}

void
schro_encoder_calculate_allocation (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->allocated_mc_bits = get_mc_alloc (frame);
  frame->allocated_residual_bits = get_residual_alloc (encoder,
      encoder->buffer_level, frame->picture_weight);
  frame->allocated_residual_bits -= frame->allocated_mc_bits;
  if (frame->allocated_residual_bits < 0) {
    frame->allocated_residual_bits = 0;
  }

}

void
schro_encoder_recalculate_allocations (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;
  int buffer_level;

  buffer_level = encoder->buffer_level;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->actual_residual_bits) {
      buffer_level += frame->actual_residual_bits + frame->actual_mc_bits;
    } else if (frame->state == SCHRO_ENCODER_FRAME_STATE_NEW ||
        frame->state == SCHRO_ENCODER_FRAME_STATE_ANALYSE ||
        frame->state == SCHRO_ENCODER_FRAME_STATE_PREDICT) {
      frame->allocated_mc_bits = get_mc_alloc (frame);
      frame->allocated_residual_bits =
        get_residual_alloc (encoder, buffer_level, frame->picture_weight);
      frame->allocated_residual_bits -= frame->allocated_mc_bits;
      if (frame->allocated_residual_bits < 0) {
        frame->allocated_residual_bits = 0;
      }
      buffer_level -= frame->allocated_residual_bits + frame->allocated_mc_bits;
    } else {
      buffer_level -= frame->allocated_residual_bits + frame->allocated_mc_bits;
    }
    SCHRO_DEBUG("%d: %d %d %d", i, frame->state,
        frame->actual_residual_bits, frame->allocated_residual_bits);
    buffer_level += encoder->bits_per_picture;
    if (buffer_level > encoder->buffer_size) {
      buffer_level = encoder->buffer_size;
    }
    if (buffer_level < 0) {
      buffer_level = 0;
    }
  }
}

static void
run_stage (SchroEncoderFrame *frame, SchroEncoderFrameStateEnum state)
{
  void *func;

  frame->state = state;
  frame->busy = TRUE;
  switch (state) {
    case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
      func = schro_encoder_analyse_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_PREDICT:
      func = schro_encoder_predict_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_ENCODING:
      func = schro_encoder_encode_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
      func = schro_encoder_reconstruct_picture;
      break;
    case SCHRO_ENCODER_FRAME_STATE_POSTANALYSE:
      func = schro_encoder_postanalyse_picture;
      break;
    default:
      SCHRO_ASSERT(0);
  }
  schro_async_run_locked (frame->encoder->async, func, frame);
}

static int
init_frame (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->params.video_format = &encoder->video_format;

  frame->need_filtering = (encoder->filtering != 0);
  switch (encoder->gop_structure) {
    case SCHRO_ENCODER_GOP_INTRA_ONLY:
      frame->need_downsampling = FALSE;
      frame->need_average_luma = FALSE;
      break;
    case SCHRO_ENCODER_GOP_ADAPTIVE:
    case SCHRO_ENCODER_GOP_BACKREF:
    case SCHRO_ENCODER_GOP_CHAINED_BACKREF:
      frame->need_downsampling = TRUE;
      frame->need_average_luma = TRUE;
      break;
    case SCHRO_ENCODER_GOP_BIREF:
    case SCHRO_ENCODER_GOP_CHAINED_BIREF:
      frame->need_downsampling = TRUE;
      frame->need_average_luma = TRUE;
      break;
  }
  return TRUE;
}

static void
setup_params_intra_only (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  schro_engine_check_new_access_unit (encoder, frame);

  frame->presentation_frame = frame->frame_number;

  frame->slot = frame->frame_number;

  frame->output_buffer_size =
    schro_engine_pick_output_buffer_size (encoder, frame);
  frame->picture_weight = 1.0;

  /* set up params */
  init_params (frame);
  if (frame->params.is_noarith) {
    init_small_codeblocks (&frame->params);
  }
}

int
schro_encoder_engine_intra_only (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;

  /* FIXME there must be a better place to put this */
  //schro_encoder_recalculate_allocations (encoder);

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        init_frame (frame);

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ANALYSE:
        setup_params_intra_only (frame);
        frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        choose_quantisers (frame);
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

static int
setup_frame_backref (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  if (!check_refs (frame)) {
    return FALSE;
  }

  frame->output_buffer_size =
    schro_engine_pick_output_buffer_size (encoder, frame);

  /* set up params */
  frame->params.num_refs = frame->num_refs;
  init_params (frame);

  return TRUE;
}

int
schro_encoder_engine_backref (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;

  /* FIXME there must be a better place to put this */
  //schro_encoder_recalculate_allocations (encoder);

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        init_frame (frame);

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      handle_gop_backref (encoder, i);
      break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_HAVE_GOP:
        if (setup_frame_backref (frame)) {
          frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        }
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
#if 0
      default:
        break;
    }
  }
  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
#endif
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

static int
setup_frame_tworef (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  if (!check_refs (frame)) {
    return FALSE;
  }

  frame->output_buffer_size =
    schro_engine_pick_output_buffer_size (encoder, frame);
  SCHRO_ASSERT(frame->output_buffer_size != 0);

  /* set up params */
  frame->params.num_refs = frame->num_refs;

  init_params (frame);

  return TRUE;
}

static void
choose_quantisers (SchroEncoderFrame *frame)
{
  schro_encoder_calculate_allocation (frame);
  schro_encoder_choose_quantisers (frame);
  schro_encoder_estimate_entropy (frame);
}

int
schro_encoder_engine_tworef (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;
  int ref;

  SCHRO_DEBUG("engine iteration");

  /* FIXME there must be a better place to put this */
  //schro_encoder_recalculate_allocations (encoder);

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("analyse i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        init_frame (frame);

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      handle_gop (encoder, i);
      break;
    }
  }

  /* Reference pictures are higher priority, so we pass over the list
   * first for reference pictures, then for non-ref. */
  for(ref = 1; ref >= 0; ref--){

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    if (frame->is_ref != ref) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_HAVE_GOP:
        if (setup_frame_tworef (frame)) {
          frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        }
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        choose_quantisers (frame);
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
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
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 2 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 3 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 4 },
  { SCHRO_WAVELET_LE_GALL_5_3, 2 },
  { SCHRO_WAVELET_LE_GALL_5_3, 3 },
  { SCHRO_WAVELET_LE_GALL_5_3, 4 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 2 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 3 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 4 },
  { SCHRO_WAVELET_HAAR_0, 2 },
  { SCHRO_WAVELET_HAAR_0, 3 },
  { SCHRO_WAVELET_HAAR_0, 4 },
  { SCHRO_WAVELET_HAAR_1, 2 },
  { SCHRO_WAVELET_HAAR_1, 3 },
  { SCHRO_WAVELET_HAAR_1, 4 },
  { SCHRO_WAVELET_FIDELITY, 2 },
  { SCHRO_WAVELET_FIDELITY, 3 },
  { SCHRO_WAVELET_DAUBECHIES_9_7, 2 },
  { SCHRO_WAVELET_DAUBECHIES_9_7, 3 },

  /* 1-level transforms look crappy */
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 1 },
  { SCHRO_WAVELET_LE_GALL_5_3, 1 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 1 },
  { SCHRO_WAVELET_HAAR_0, 1 },
  { SCHRO_WAVELET_HAAR_1, 1 },
  { SCHRO_WAVELET_FIDELITY, 1 },
  { SCHRO_WAVELET_DAUBECHIES_9_7, 1 },

#ifdef SCHRO_HAVE_DEEP_WAVELETS
  { SCHRO_WAVELET_FIDELITY, 4 },
  { SCHRO_WAVELET_DAUBECHIES_9_7, 4 }
#endif

#ifdef USE_TRANSFORM_LEVEL_5
  /* 5-level transforms don't decrease bitrate */
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 5 },
  { SCHRO_WAVELET_LE_GALL_5_3, 5 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 5 },
  { SCHRO_WAVELET_HAAR_0, 5 },
  { SCHRO_WAVELET_HAAR_1, 5 },
#endif

#ifdef USE_TRANSFORM_LEVEL_6
  /* 6-level transforms don't decrease bitrate */
  { SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 6 },
  { SCHRO_WAVELET_LE_GALL_5_3, 6 },
  { SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 6 },
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
        frame->need_downsampling = FALSE;
        frame->need_filtering = (encoder->filtering != 0);
        frame->need_average_luma = FALSE;

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
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
        /* FIXME don't change config on user */
        encoder->intra_wavelet = test_wavelet_types[j].type;
        encoder->transform_depth = test_wavelet_types[j].depth;
        init_params (frame);

        frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
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

  /* FIXME don't change config on user */
  encoder->intra_wavelet = SCHRO_WAVELET_HAAR_0;
  encoder->inter_wavelet = SCHRO_WAVELET_HAAR_0;
  encoder->transform_depth = 3;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->need_downsampling = TRUE;
        frame->need_filtering = FALSE;
        frame->need_average_luma = FALSE;

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    if (frame->frame_number == encoder->gop_picture) {
      handle_gop_backref (encoder, i);
      break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;
    SCHRO_DEBUG("backref i=%d picture=%d state=%d busy=%d", i, frame->frame_number, frame->state, frame->busy);

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_HAVE_GOP:
        if (!check_refs (frame)) {
          continue;
        }

        frame->output_buffer_size =
          schro_engine_pick_output_buffer_size (encoder, frame);

        /* set up params */
        params = &frame->params;
        params->num_refs = frame->num_refs;
        params->video_format = &encoder->video_format;
        init_params (frame);
        if (params->is_noarith) {
          init_small_codeblocks (params);
        }

        params->xbsep_luma = 8;
        params->xblen_luma = 8;
        params->ybsep_luma = 8;
        params->yblen_luma = 8;

        SCHRO_DEBUG("queueing %d", frame->frame_number);

        frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
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
        frame->need_downsampling = TRUE;
        frame->need_filtering = (encoder->filtering != 0);
        frame->need_average_luma = FALSE;

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
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
            frame->retired_picture_number = encoder->last_ref;
          }
          encoder->last_ref = frame->frame_number;
        } else {
          params->num_refs = 1;
          frame->picture_number_ref0 = encoder->last_ref;
        }

        init_params (frame);

        params->xbsep_luma = 8;
        params->xblen_luma = 8;
        params->ybsep_luma = 8;
        params->yblen_luma = 8;

        for(comp=0;comp<3;comp++){
          for(j=0;j<SCHRO_LIMIT_SUBBANDS;j++){
            frame->quant_index[comp][j] = 0;
          }
        }

        if (params->num_refs > 0) {
          frame->ref_frame0 = schro_encoder_reference_get (encoder,
              frame->picture_number_ref0);
          schro_encoder_frame_ref (frame->ref_frame0);
        } else {
          frame->ref_frame0 = NULL;
        }

        SCHRO_DEBUG("queueing %d", frame->frame_number);

        frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

int
schro_encoder_engine_lowdelay (SchroEncoder *encoder)
{
  SchroParams *params;
  SchroEncoderFrame *frame;
  int i;
  int num;
  int denom;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_NEW:
        frame->need_downsampling = FALSE;
        frame->need_filtering = (encoder->filtering != 0);
        frame->need_average_luma = FALSE;

        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ANALYSE);
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
        params->video_format = &encoder->video_format;

        params->n_horiz_slices = encoder->horiz_slices;
        params->n_vert_slices = encoder->vert_slices;
        init_params (frame);
        //schro_params_init_lowdelay_quantisers(params);

        num = muldiv64(encoder->bitrate,
            encoder->video_format.frame_rate_denominator,
            encoder->video_format.frame_rate_numerator * 8);
        denom = params->n_horiz_slices * params->n_vert_slices;
        SCHRO_ASSERT(denom != 0);
        schro_utils_reduce_fraction (&num, &denom);
        params->slice_bytes_num = num;
        params->slice_bytes_denom = denom;

        frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS;
        break;
      case SCHRO_ENCODER_FRAME_STATE_HAVE_PARAMS:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_PREDICT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_PREDICT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_ENCODING);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_ENCODING:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT);
        return TRUE;
      case SCHRO_ENCODER_FRAME_STATE_RECONSTRUCT:
        run_stage (frame, SCHRO_ENCODER_FRAME_STATE_POSTANALYSE);
        return TRUE;
      default:
        break;
    }
  }

  return FALSE;
}

