
#ifndef __SCHRO_PARAMS_H__
#define __SCHRO_PARAMS_H__

typedef struct _SchroParams SchroParams;
typedef struct _SchroSubband SchroSubband;
typedef struct _SchroMotionVector SchroMotionVector;
typedef struct _SchroPicture SchroPicture;

struct _SchroParams {

  int major_version;
  int minor_version;
  int profile;
  int level;

  int height;
  int width;

  /* */

  int chroma_format_index;
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
  int interlace;
  int top_field_first;
  int frame_rate_numerator;
  int frame_rate_denominator;
  int pixel_aspect_ratio_numerator;
  int pixel_aspect_ratio_denominator;
  int clean_tl_x;
  int clean_tl_y;
  int clean_width;
  int clean_height;
  int colour_matrix_index;
  int signal_range_index;
  int colour_primaries_index;
  int transfer_char_index;

  int non_spec_input;

  int have_chroma;
  int chroma_h_scale;
  int chroma_v_scale;
  int chroma_width;
  int chroma_height;
  int mc_chroma_width;
  int mc_chroma_height;
  int mc_luma_width;
  int mc_luma_height;
  int is_intra;

  /* transform parameters */
  int wavelet_filter_index;
  int transform_depth;
  int spatial_partition_flag;
  int nondefault_partition_flag;
  int codeblock_width[8];
  int codeblock_height[8];
  int codeblock_mode_index;

  /* motion prediction parameters */
  int num_refs;
  int global_motion;
  int xblen_luma;
  int yblen_luma;
  int xbsep_luma;
  int ybsep_luma;
  int x_num_mb;
  int y_num_mb;
  int x_num_blocks;
  int y_num_blocks;
  int mv_precision;
  int global_only_flag;
  int global_prec_bits;
  int b_1[2];
  int b_2[2];
  int a_11[2];
  int a_12[2];
  int a_21[2];
  int a_22[2];
  int c_1[2];
  int c_2[2];

  /* frame padding */
  int iwt_chroma_width;
  int iwt_chroma_height;
  int iwt_luma_width;
  int iwt_luma_height;


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
  int pred_mode;
  int mb_using_global;
  int mb_split;
  int mb_common;
  int x;
  int y;
  int dc[3];
  int metric;
};

struct _SchroPicture {
  int is_ref;
  int n_refs;

  int frame_number;
  int reference_frame_number[2];

  int n_retire;
  int retire[10];
};

void schro_params_calculate_mc_sizes (SchroParams *params);
void schro_params_calculate_iwt_sizes (SchroParams *params);

void schro_params_set_video_format (SchroParams *params, int index);
int schro_params_get_video_format (SchroParams *params);
void schro_params_set_chroma_format (SchroParams *params, int index);
int schro_params_get_chroma_format (SchroParams *params);
void schro_params_set_frame_rate (SchroParams *params, int index);
int schro_params_get_frame_rate (SchroParams *params);
void schro_params_set_pixel_aspect_ratio (SchroParams *params, int index);
int schro_params_get_pixel_aspect_ratio (SchroParams *params);
void schro_params_set_block_params (SchroParams *params, int index);
void schro_params_set_default_codeblock (SchroParams *params);


#endif

