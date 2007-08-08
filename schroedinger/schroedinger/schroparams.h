
#ifndef __SCHRO_PARAMS_H__
#define __SCHRO_PARAMS_H__

#include <schroedinger/schrobitstream.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schro-stdint.h>

SCHRO_BEGIN_DECLS

typedef uint32_t SchroPictureNumber;

typedef struct _SchroVideoFormat SchroVideoFormat;
typedef struct _SchroParams SchroParams;
typedef struct _SchroSubband SchroSubband;
typedef struct _SchroMotionVector SchroMotionVector;
typedef struct _SchroMotionVectorDC SchroMotionVectorDC;
typedef struct _SchroMotionField SchroMotionField;
typedef struct _SchroGlobalMotion SchroGlobalMotion;

struct _SchroVideoFormat {
  int index;
  int width;
  int height;
  int chroma_format;
  int video_depth;
    
  int interlaced;
  int top_field_first;
  int sequential_fields;
  
  int frame_rate_numerator;
  int frame_rate_denominator;
  int aspect_ratio_numerator;
  int aspect_ratio_denominator;
    
  int clean_width;
  int clean_height;
  int left_offset;
  int top_offset;
    
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
    
  int colour_primaries;
  int colour_matrix;
  int transfer_function;

  /* calculated values */

  int chroma_h_shift;
  int chroma_v_shift;
  int chroma_width;
  int chroma_height;
};  

struct _SchroProfile {
  int max_pixels_per_second;
  int max_blocks_per_second;
  int max_arith_ops_per_second;

  int max_transform_depth;

  int allow_global_motion;
  int allow_spatial_partition;
  int allow_inter;
};

struct _SchroGlobalMotion {
  int shift;
  int b0;
  int b1;
  int a_exp;
  int a00;
  int a01;
  int a10;
  int a11;
  int c_exp;
  int c0;
  int c1;
};

struct _SchroParams {
  SchroVideoFormat *video_format;

  /* transform parameters */
  int wavelet_filter_index;
  int transform_depth;
  int spatial_partition_flag;
  int nondefault_partition_flag;
  int horiz_codeblocks[SCHRO_MAX_TRANSFORM_DEPTH + 1];
  int vert_codeblocks[SCHRO_MAX_TRANSFORM_DEPTH + 1];
  int codeblock_mode_index;

  /* motion prediction parameters */
  int num_refs;
  int have_global_motion;
  int xblen_luma;
  int yblen_luma;
  int xbsep_luma;
  int ybsep_luma;
  int mv_precision;
  SchroGlobalMotion global_motion[2];
  int picture_pred_mode;
  int picture_weight_bits;
  int picture_weight_1;
  int picture_weight_2;

  /* DiracPro parameters */
  int slice_width_exp;
  int slice_height_exp;
  int slice_bytes_num;
  int slice_bytes_denom;
  int quant_matrix[3*SCHRO_MAX_TRANSFORM_DEPTH+1];
  int luma_quant_offset;
  int chroma1_quant_offset;
  int chroma2_quant_offset;

  /* calculated sizes */
  int iwt_chroma_width;
  int iwt_chroma_height;
  int iwt_luma_width;
  int iwt_luma_height;
  int mc_chroma_width;
  int mc_chroma_height;
  int mc_luma_width;
  int mc_luma_height;
  int x_num_blocks;
  int y_num_blocks;
};

struct _SchroSubband {
  int has_parent;
  int quant_index;
  int position;
};

#define SCHRO_SUBBAND_IS_HORIZONTALLY_ORIENTED(position) (((position)&3) == 2)
#define SCHRO_SUBBAND_IS_VERTICALLY_ORIENTED(position) (((position)&3) == 1)
#define SCHRO_SUBBAND_SHIFT(position) ((position)>>2)

struct _SchroMotionVector {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
  int16_t x1;
  int16_t y1;
  int16_t x2;
  int16_t y2;
};

struct _SchroMotionVectorDC {
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
  uint8_t dc[3];
  uint8_t _padding1;
  uint32_t _padding2;
};

struct _SchroMotionField {
  int x_num_blocks;
  int y_num_blocks;
  SchroMotionVector *motion_vectors;
};

void schro_params_init (SchroParams *params, int video_format);

void schro_params_calculate_iwt_sizes (SchroParams *params);
void schro_params_calculate_mc_sizes (SchroParams *params);

int schro_params_validate (SchroVideoFormat *format);

void schro_params_set_video_format (SchroVideoFormat *format, SchroVideoFormatEnum index);
SchroVideoFormatEnum schro_params_get_video_format (SchroVideoFormat *format);
void schro_params_set_frame_rate (SchroVideoFormat *format, int index);
int schro_params_get_frame_rate (SchroVideoFormat *format);
void schro_params_set_aspect_ratio (SchroVideoFormat *format, int index);
int schro_params_get_aspect_ratio (SchroVideoFormat *format);
void schro_params_set_signal_range (SchroVideoFormat *format, int index);
int schro_params_get_signal_range (SchroVideoFormat *format);
void schro_params_set_colour_spec (SchroVideoFormat *format, int index);
int schro_params_get_colour_spec (SchroVideoFormat *format);
void schro_params_set_block_params (SchroParams *params, int index);
int schro_params_get_block_params (SchroParams *params);

void schro_params_set_default_codeblock (SchroParams *params);

void schro_params_init_subbands (SchroParams *params, SchroSubband *subbands,
    int luma_frame_stride, int chroma_frame_stride);
void schro_subband_get_frame_component (SchroFrameComponent *dest,
    SchroFrameComponent *full_frame, int position);
void schro_subband_get (SchroFrame *frame, int component, int position,
    SchroParams *params, int16_t **data, int *stride, int *width, int *height);

/* FIXME should be SchroFrameFormat */
int schro_params_get_frame_format (int depth,
    SchroChromaFormat chroma_format);

/* FIXME should be moved */
void schro_frame_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp);
void schro_frame_inverse_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp);

SCHRO_END_DECLS

#endif

