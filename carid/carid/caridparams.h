
#ifndef __CARID_PARAMS_H__
#define __CARID_PARAMS_H__

typedef struct _CaridParams CaridParams;
typedef struct _CaridSubband CaridSubband;

struct _CaridParams {

  int major_version;
  int minor_version;
  int profile;
  int level;

  int height;
  int width;

  /* */

  int video_format_index;
  int chroma_format_index;
  int signal_range_index;
  int luma_offset;
  int luma_excursion;
  int chroma_offset;
  int chroma_excursion;
  int interlace;
  int top_field_first;
  int frame_rate_index;
  int frame_rate_numerator;
  int frame_rate_denominator;
  int aspect_ratio_index;
  int aspect_ratio_numerator;
  int aspect_ratio_denominator;
  int clean_tl_x;
  int clean_tl_y;
  int clean_width;
  int clean_height;

  int non_spec_input;

  int chroma_h_scale;
  int chroma_v_scale;
  int chroma_width;
  int chroma_height;
  int mc_chroma_width;
  int mc_chroma_height;
  int is_intra;

  /* transform parameters */
  int wavelet_filter_index;
  int transform_depth;
  int spatial_partition;
  int partition_index;
  int max_xblocks;
  int max_yblocks;
  int multi_quant;

  /* frame padding */
  int iwt_chroma_width;
  int iwt_chroma_height;
  int iwt_luma_width;
  int iwt_luma_height;

};

struct _CaridSubband {
  int x;
  int y;
  int w;
  int h;
  int offset;
  int stride;
  int has_parent;
  int scale_factor_shift;
  int horizontally_oriented;
  int vertically_oriented;
  int quant_index;
};

void carid_params_calculate_iwt_sizes (CaridParams *params);

#endif

