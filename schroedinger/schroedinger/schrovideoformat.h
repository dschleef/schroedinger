
#ifndef __SCHRO_VIDEO_FORMAT_H__
#define __SCHRO_VIDEO_FORMAT_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schrobitstream.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroVideoFormat SchroVideoFormat;

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

int schro_video_format_validate (SchroVideoFormat *format);

void schro_video_format_set_std_video_format (SchroVideoFormat *format,
    SchroVideoFormatEnum index);
SchroVideoFormatEnum schro_video_format_get_std_video_format (SchroVideoFormat *format);
void schro_video_format_set_std_frame_rate (SchroVideoFormat *format, int index);
int schro_video_format_get_std_frame_rate (SchroVideoFormat *format);
void schro_video_format_set_std_aspect_ratio (SchroVideoFormat *format, int index);
int schro_video_format_get_std_aspect_ratio (SchroVideoFormat *format);
void schro_video_format_set_std_signal_range (SchroVideoFormat *format, int index);
int schro_video_format_get_std_signal_range (SchroVideoFormat *format);
void schro_video_format_set_std_colour_spec (SchroVideoFormat *format, int index);
int schro_video_format_get_std_colour_spec (SchroVideoFormat *format);

SCHRO_END_DECLS

#endif

