
#ifndef __SCHRO_MOTIONEST_H__
#define __SCHRO_MOTIONEST_H__

#include <schroedinger/schroencoder.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroMotionEst SchroMotionEst;
typedef struct _SchroRoughME SchroRoughME;
typedef struct _SchroBlock SchroBlock;

#ifdef SCHRO_ENABLE_UNSTABLE_API

#define SCHRO_MAX_HIER_LEVELS 8

struct _SchroMotionEst {
  SchroEncoderFrame *encoder_frame;
  SchroParams *params;

  double lambda;

  SchroFrame *downsampled_src0[SCHRO_MAX_HIER_LEVELS];
  SchroFrame *downsampled_src1[SCHRO_MAX_HIER_LEVELS];

  SchroMotion *motion;

  SchroBlock *sblocks;

  //SchroMotionField *downsampled_mf[2][8];
  int scan_distance;

  int badblocks;
  double hier_score;
};

struct _SchroRoughME {
  SchroEncoderFrame *encoder_frame;
  SchroEncoderFrame *ref_frame;

  SchroMotionField *motion_fields[SCHRO_MAX_HIER_LEVELS];
};

struct _SchroBlock {
  int valid;
  int error;
  int entropy;

  double score;

  SchroMotionVector mv[4][4];
};

SchroMotionEst *schro_motionest_new (SchroEncoderFrame *frame);
void schro_motionest_free (SchroMotionEst *me);

SchroRoughME * schro_rough_me_new (SchroEncoderFrame *frame, SchroEncoderFrame *ref);
void schro_rough_me_free (SchroRoughME *rme);
void schro_rough_me_heirarchical_scan (SchroRoughME *rme);
void schro_rough_me_heirarchical_scan_nohint (SchroRoughME *rme, int shift,
    int distance);
void schro_rough_me_heirarchical_scan_hint (SchroRoughME *rme, int shift,
    int distance);


void schro_encoder_motion_predict_rough (SchroEncoderFrame *frame);
void schro_encoder_motion_predict_pel (SchroEncoderFrame *frame);
void schro_encoder_motion_predict_subpel (SchroEncoderFrame *frame);

void schro_encoder_global_estimation (SchroEncoderFrame *frame);

SchroMotionField * schro_motion_field_new (int x_num_blocks, int y_num_blocks);
void schro_motion_field_free (SchroMotionField *field);
void schro_motion_field_scan (SchroMotionField *field, SchroParams *params, SchroFrame *frame, SchroFrame *ref, int dist);
void schro_motion_field_inherit (SchroMotionField *field, SchroMotionField *parent);
void schro_motion_field_copy (SchroMotionField *field, SchroMotionField *parent);

int schro_frame_get_metric (SchroFrame *frame1, int x1, int y1,
    SchroFrame *frame2, int x2, int y2);
void schro_motion_field_lshift (SchroMotionField *mf, int n);

int schro_motion_estimate_entropy (SchroMotion *motion);
int schro_motion_block_estimate_entropy (SchroMotion *motion, int i, int j);
int schro_motion_superblock_estimate_entropy (SchroMotion *motion, int i, int j);
int schro_motion_superblock_try_estimate_entropy (SchroMotion *motion, int i,
    int j, SchroBlock *block);
int schro_motionest_superblock_get_metric (SchroMotionEst *me,
    SchroBlock *block, int i, int j);
void schro_motion_copy_from (SchroMotion *motion, int i, int j, SchroBlock *block);
void schro_motion_copy_to (SchroMotion *motion, int i, int j, SchroBlock *block);

void schro_block_fixup (SchroBlock *block);
int schro_block_check (SchroBlock *block);

#endif

SCHRO_END_DECLS

#endif

