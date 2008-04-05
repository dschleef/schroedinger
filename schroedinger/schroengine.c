
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>

#include <math.h>

int schro_engine_get_scene_change_score (SchroEncoder *encoder, int i);
void schro_encoder_calculate_allocation (SchroEncoderFrame *frame);
static void choose_quantisers (SchroEncoderFrame *frame);

/**
 * schro_engine_check_new_access_unit:
 * @encoder: encoder
 * @frame: encoder frame
 *
 * Checks if the current picture should be the start of a new access
 * unit.
 */
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

/**
 * schro_engine_code_picture:
 * @frame: encoder frame
 * @is_ref:
 * @retire:
 * @num_refs:
 * @ref0:
 * @ref1:
 *
 * Used to set coding order and coding parameters for a picture.
 */
static void
schro_engine_code_picture (SchroEncoderFrame *frame,
    int is_ref, int retire, int num_refs, int ref0, int ref1)
{
  SchroEncoder *encoder = frame->encoder;

  SCHRO_DEBUG("preparing %d as is_ref=%d retire=%d num_refs=%d ref0=%d ref1=%d",
      frame->frame_number, is_ref, retire, num_refs, ref0, ref1);

  frame->is_ref = is_ref;
  frame->retired_picture_number = retire;
  frame->num_refs = num_refs;
  frame->picture_number_ref[0] = ref0;
  frame->picture_number_ref[1] = ref1;

  frame->state = SCHRO_ENCODER_FRAME_STATE_HAVE_GOP;
  frame->slot = encoder->next_slot++;

  if (num_refs > 0) {
    frame->ref_frame[0] = schro_encoder_reference_get (encoder, ref0);
    SCHRO_ASSERT(frame->ref_frame[0]);
    schro_encoder_frame_ref (frame->ref_frame[0]);
  }
  if (num_refs > 1) {
    frame->ref_frame[1] = schro_encoder_reference_get (encoder, ref1);
    SCHRO_ASSERT(frame->ref_frame[1]);
    schro_encoder_frame_ref (frame->ref_frame[1]);
  }
  if (is_ref) {
    int i;
    for(i=0;i<SCHRO_LIMIT_REFERENCE_FRAMES;i++){
      if (encoder->reference_pictures[i] == NULL) break;
      if (encoder->reference_pictures[i]->frame_number == retire) {
        schro_encoder_frame_unref(encoder->reference_pictures[i]);
        break;
      }
    }
    SCHRO_ASSERT(i<SCHRO_LIMIT_REFERENCE_FRAMES);
    encoder->reference_pictures[i] = frame;
    schro_encoder_frame_ref (frame);
  }
}

/**
 * check_refs:
 * @frame: encoder frame
 *
 * Checks whether reference pictures are available to be used for motion
 * rendering.
 */
static int
check_refs (SchroEncoderFrame *frame)
{
  if (frame->num_refs == 0) return TRUE;

  if (frame->num_refs > 0 &&
      !(frame->ref_frame[0]->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
    return FALSE;
  }
  if (frame->num_refs > 1 &&
      !(frame->ref_frame[1]->state & SCHRO_ENCODER_FRAME_STATE_DONE)) {
    return FALSE;
  }

  return TRUE;
}

/**
 * schro_engine_get_scene_change_score:
 * @frame: encoder frame
 * @i: index
 *
 * Calculates scene change score for two pictures.
 */
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


/**
 * schro_engine_pick_output_buffer_size:
 * @encoder: encoder
 * @frame: encoder frame
 *
 * Calculates allocated size of output buffer for a picture.  Horribly
 * inefficient and outdated.
 */
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

/**
 * init_params:
 * @frame: encoder frame
 *
 * Initializes params structure for picture based on encoder parameters
 * and some heuristics.
 */
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

  schro_params_calculate_mc_sizes (params);
  schro_params_calculate_iwt_sizes (params);

  if (frame->params.is_noarith) {
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

  params->mv_precision = encoder->mv_precision;
  //params->have_global_motion = TRUE;
  params->codeblock_mode_index = 0;
}

/**
 * get_residual_alloc:
 * @encoder:
 * @buffer_level:
 * @picture_weight:
 *
 * Calculates the number of bits allocated for coding residual for a
 * picture.
 */
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

/**
 * get_mc_alloc:
 * @frame: encoder frame
 *
 * Calculates the number of bits allocated for coding MC for a picture.
 */
static int
get_mc_alloc (SchroEncoderFrame *frame)
{
  return 10 * frame->params.x_num_blocks * frame->params.y_num_blocks / 16;
}

/**
 * schro_encoder_calculate_allocation:
 * @frame:
 *
 * Calculates the number of bits to allocate to a picture.
 */
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

/**
 * run_stage:
 * @frame:
 * @state:
 *
 * Runs a stage in the encoding process.
 */
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

/**
 * init_frame:
 * @frame:
 *
 * Initializes a frame prior to any analysis.
 */
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
choose_quantisers (SchroEncoderFrame *frame)
{
  schro_encoder_calculate_allocation (frame);
  schro_encoder_choose_quantisers (frame);
  schro_encoder_estimate_entropy (frame);
}



/***** tworef *****/

/**
 * handle_gop_tworef:
 * @encoder:
 * @i:
 *
 * Sets up a minor group of pictures for the tworef engine.
 */
static void
handle_gop_tworef (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;
  SchroEncoderFrame *ref2;
  SchroEncoderFrame *f;
  int j;
  int gop_length;
  schro_bool intra_start;
  double scs_sum;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);

  gop_length = encoder->magic_subgroup_length;
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
  scs_sum = 0;
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

    if (j==0) {
      if (f->scene_change_score > encoder->magic_scene_change_threshold) {
        intra_start = TRUE;
      }
    } else {
      scs_sum += f->scene_change_score;
      if (scs_sum > encoder->magic_scene_change_threshold) {
        /* matching is getting bad.  terminate gop */
        gop_length = j;
      }
    }
  }

  SCHRO_DEBUG("gop length %d", gop_length);

  if (gop_length == 1) {
    schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
    frame->presentation_frame = frame->frame_number;
    frame->picture_weight = encoder->magic_bailout_weight;
  } else {
    if (intra_start) {
      /* IBBBP */
      schro_engine_code_picture (frame, TRUE, encoder->intra_ref, 0, -1, -1);
      encoder->intra_ref = frame->frame_number;

      frame->presentation_frame = frame->frame_number;
      //frame->picture_weight = 1 + (gop_length - 1) * 0.6;
      frame->picture_weight = encoder->magic_keyframe_weight;

      f = encoder->frame_queue->elements[i+gop_length-1].data;
      schro_engine_code_picture (f, TRUE, encoder->last_ref,
          1, frame->frame_number, -1);

      f->presentation_frame = frame->frame_number;
      f->picture_weight = encoder->magic_inter_p_weight;
      //f->picture_weight += (gop_length - 2) * (1 - encoder->magic_inter_b_weight);
      encoder->last_ref = encoder->last_ref2;
      encoder->last_ref2 = f->frame_number;
      ref2 = f;

      for (j = 1; j < gop_length - 1; j++) {
        f = encoder->frame_queue->elements[i+j].data;
        schro_engine_code_picture (f, FALSE, -1,
            2, frame->frame_number, ref2->frame_number);
        f->presentation_frame = f->frame_number;
        if (j == gop_length-2) {
          f->presentation_frame++;
        }
        f->picture_weight = encoder->magic_inter_b_weight;
      }
    } else {
      /* BBBP */
      f = encoder->frame_queue->elements[i+gop_length-1].data;
      schro_engine_code_picture (f, TRUE, encoder->last_ref,
          2, encoder->last_ref2, encoder->intra_ref);
      f->presentation_frame = encoder->last_ref2;
      f->picture_weight = encoder->magic_inter_p_weight;
      //f->picture_weight += (gop_length - 1) * (1 - encoder->magic_inter_b_weight);
      encoder->last_ref = encoder->last_ref2;
      encoder->last_ref2 = f->frame_number;

      for (j = 0; j < gop_length - 1; j++) {
        f = encoder->frame_queue->elements[i+j].data;
        schro_engine_code_picture (f, FALSE, -1,
            2, encoder->last_ref, encoder->last_ref2);
        f->presentation_frame = f->frame_number;
        if (j == gop_length-2) {
          f->presentation_frame++;
        }
        f->picture_weight = encoder->magic_inter_b_weight;
      }
    }
  }

  encoder->gop_picture += gop_length;
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

/**** backref ****/

/**
 * handle_gop_backref:
 * @encoder:
 * @i:
 *
 * Sets up a minor group of pictures for the backref engine.
 */
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

  gop_length = encoder->magic_subgroup_length;
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

  schro_engine_code_picture (frame, TRUE, encoder->last_ref, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1 + (gop_length - 1) * (1 - encoder->magic_inter_b_weight);
  encoder->last_ref = frame->frame_number;

  for (j = 1; j < gop_length; j++) {
    f = encoder->frame_queue->elements[i+j].data;
    schro_engine_code_picture (f, FALSE, -1, 1, frame->frame_number, -1);
    f->presentation_frame = f->frame_number;
    f->picture_weight = encoder->magic_inter_b_weight;
  }

  encoder->gop_picture += gop_length;
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
/*** intra-only ***/

/**
 * handle_gop_intra_only:
 * @encoder:
 * @i:
 *
 * Sets up GOP structure for an intra picture.
 */
static void
handle_gop_intra_only (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);

  SCHRO_DEBUG("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture, i);

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) {
    SCHRO_DEBUG("picture %d not ready", i);
    return;
  }

  schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1.0;

  encoder->gop_picture++;
}

/**
 * setup_params_intra_only:
 * @frame:
 *
 * sets up parameters for a picture for intra-only encoding.
 */
static void
setup_params_intra_only (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;

  frame->output_buffer_size =
    schro_engine_pick_output_buffer_size (encoder, frame);

  frame->params.num_refs = frame->num_refs;

  /* set up params */
  init_params (frame);
}

/**
 * schro_encoder_engine_intra_only:
 * @encoder:
 *
 * engine for intra-only encoding.
 */
int
schro_encoder_engine_intra_only (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

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
      handle_gop_intra_only (encoder, i);
      break;
    }
  }

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

    if (frame->busy) continue;

    switch (frame->state) {
      case SCHRO_ENCODER_FRAME_STATE_HAVE_GOP:
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




int
schro_encoder_engine_tworef (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;
  int ref;

  SCHRO_DEBUG("engine iteration");

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
      handle_gop_tworef (encoder, i);
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

/*** lossless ***/

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

/*** low delay ***/

static void
handle_gop_lowdelay (SchroEncoder *encoder, int i)
{
  SchroEncoderFrame *frame;

  frame = encoder->frame_queue->elements[i].data;

  if (frame->busy || frame->state != SCHRO_ENCODER_FRAME_STATE_ANALYSE) return;

  schro_engine_check_new_access_unit (encoder, frame);

  SCHRO_DEBUG("handling gop from %d to %d (index %d)", encoder->gop_picture,
      encoder->gop_picture, i);

  schro_engine_code_picture (frame, FALSE, -1, 0, -1, -1);
  frame->presentation_frame = frame->frame_number;
  frame->picture_weight = 1.0;

  encoder->gop_picture++;
}

static void
setup_params_lowdelay (SchroEncoderFrame *frame)
{
  SchroEncoder *encoder = frame->encoder;
  SchroParams *params = &frame->params;
  int num;
  int denom;

  frame->output_buffer_size =
    schro_engine_pick_output_buffer_size (encoder, frame);

  /* set up params */
  params->num_refs = frame->num_refs;
  params->is_lowdelay = TRUE;

  params->n_horiz_slices = encoder->horiz_slices;
  params->n_vert_slices = encoder->vert_slices;
  init_params (frame);
  //schro_params_init_lowdelay_quantisers(params);

  num = muldiv64(encoder->bitrate,
      encoder->video_format.frame_rate_denominator,
      encoder->video_format.frame_rate_numerator * 8);
  denom = params->n_horiz_slices * params->n_vert_slices;
  if (encoder->video_format.interlaced_coding) {
    denom *= 2;
  }
  SCHRO_ASSERT(denom != 0);
  schro_utils_reduce_fraction (&num, &denom);
  params->slice_bytes_num = num;
  params->slice_bytes_denom = denom;
}

int
schro_encoder_engine_lowdelay (SchroEncoder *encoder)
{
  SchroEncoderFrame *frame;
  int i;
  int ref;

  encoder->quantiser_engine = SCHRO_QUANTISER_ENGINE_LOWDELAY;

  for(i=0;i<encoder->frame_queue->n;i++) {
    frame = encoder->frame_queue->elements[i].data;

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
      handle_gop_lowdelay (encoder, i);
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
        setup_params_lowdelay (frame);
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
  }

  return FALSE;
}

