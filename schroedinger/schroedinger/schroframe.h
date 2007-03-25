
#ifndef __SCHRO_FRAME_H__
#define __SCHRO_FRAME_H__

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schrobuffer.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _SchroFrame SchroFrame;
typedef struct _SchroFrameComponent SchroFrameComponent;

typedef void (*SchroFrameFreeFunc)(SchroFrame *frame, void *priv);

typedef enum _SchroFrameFormat {
  SCHRO_FRAME_FORMAT_U8,
  SCHRO_FRAME_FORMAT_S16
} SchroFrameFormat;

struct _SchroFrameComponent {
  void *data;
  int stride;
  int width;
  int height;
  int length;
  int h_shift;
  int v_shift;
};

struct _SchroFrame {
  //SchroBuffer *buffer;

  SchroFrameFreeFunc free;
  void *regions[3];
  void *priv;

  SchroFrameFormat format;

  SchroFrameComponent components[3];

  uint32_t frame_number;
};


SchroFrame * schro_frame_new (void);
SchroFrame * schro_frame_new_and_alloc2 (SchroFrameFormat format, int width,
    int height, int width2, int height2);
SchroFrame * schro_frame_new_I420 (void *data, int width, int height);
void schro_frame_set_free_callback (SchroFrame *frame,
    SchroFrameFreeFunc free_func, void *priv);
void schro_frame_free (SchroFrame *frame);
void schro_frame_convert (SchroFrame *dest, SchroFrame *src);
void schro_frame_add (SchroFrame *dest, SchroFrame *src);
void schro_frame_subtract (SchroFrame *dest, SchroFrame *src);
void schro_frame_shift_left (SchroFrame *frame, int shift);
void schro_frame_shift_right (SchroFrame *frame, int shift);
void schro_frame_edge_extend (SchroFrame *frame, int width, int height);
void schro_frame_zero_extend (SchroFrame *frame, int width, int height);

void schro_frame_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp);
void schro_frame_inverse_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp);

void schro_frame_downsample (SchroFrame *dest, SchroFrame *src, int shift);
void schro_frame_h_upsample (SchroFrame *dest, SchroFrame *src);
void schro_frame_v_upsample (SchroFrame *dest, SchroFrame *src);
int schro_frame_calculate_average_luma (SchroFrame *frame);


#ifdef __cplusplus
}
#endif

#endif

