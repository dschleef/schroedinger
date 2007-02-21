
#ifndef __SCHRO_PARAMS_H__
#define __SCHRO_PARAMS_H__

#include <schroedinger/schrobitstream.h>
#include <schroedinger/schro-stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _SchroVideoFormat SchroVideoFormat;
typedef struct _SchroParams SchroParams;
typedef struct _SchroSubband SchroSubband;
typedef struct _SchroMotionVector SchroMotionVector;
typedef struct _SchroMotionField SchroMotionField;
typedef struct _SchroPicture SchroPicture;
typedef struct _SchroGlobalMotion SchroGlobalMotion;

struct _SchroVideoFormat {
  int width;
  int height;
  int chroma_format;
  int video_depth;
    
  int interlaced_source;
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

  int chroma_h_scale;
  int chroma_v_scale;
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
  int x;
  int y;
  int w;
  int h;
  int offset;
  int stride;
  int chroma_w;
  int chroma_h;
  int chroma_offset;
  int chroma_stride;
  int has_parent;
  int scale_factor_shift;
  int horizontally_oriented;
  int vertically_oriented;
  int quant_index;
};

struct _SchroMotionVector {
  union {
    struct {
      int16_t x;
      int16_t y;
    } xy;
    uint8_t dc[3];
  } u;
  unsigned int pred_mode : 2;
  unsigned int using_global : 1;
  unsigned int split : 2;
  unsigned int unused : 3;
  unsigned int scan : 8;
  unsigned int metric : 16;
};

struct _SchroMotionField {
  int x_num_blocks;
  int y_num_blocks;
  SchroMotionVector *motion_vectors;
};

struct _SchroPicture {
  int is_ref;
  int n_refs;

  int frame_number;
  int reference_frame_number[2];

  int n_retire;
  int retire[SCHRO_MAX_REFERENCE_FRAMES];
};

void schro_params_calculate_iwt_sizes (SchroParams *params);
void schro_params_calculate_mc_sizes (SchroParams *params);

int schro_params_validate (SchroVideoFormat *format);

void schro_params_set_video_format (SchroVideoFormat *format, int index);
int schro_params_get_video_format (SchroVideoFormat *format);
void schro_params_set_frame_rate (SchroVideoFormat *format, int index);
int schro_params_get_frame_rate (SchroVideoFormat *format);
void schro_params_set_aspect_ratio (SchroVideoFormat *format, int index);
int schro_params_get_aspect_ratio (SchroVideoFormat *format);
void schro_params_set_signal_range (SchroVideoFormat *format, int index);
int schro_params_get_signal_range (SchroVideoFormat *format);
void schro_params_set_colour_spec (SchroVideoFormat *format, int index);
int schro_params_get_colour_spec (SchroVideoFormat *format);
void schro_params_set_block_params (SchroParams *params, int index);

void schro_params_set_default_codeblock (SchroParams *params);

void schro_params_init_subbands (SchroParams *params, SchroSubband *subbands);

#ifdef __cplusplus
}
#endif


#endif

