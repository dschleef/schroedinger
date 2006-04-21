
#ifndef __CARID_FRAME_H__
#define __CARID_FRAME_H__

#include <carid/caridbuffer.h>

typedef struct _CaridFrame CaridFrame;
typedef struct _CaridFrameComponent CaridFrameComponent;

typedef void (*CaridFrameFreeFunc)(CaridFrame *frame, void *priv);

enum _CaridFrameFormat {
  CARID_FRAME_FORMAT_U8,
  CARID_FRAME_FORMAT_S16
};
typedef enum _CaridFrameFormat CaridFrameFormat;

struct _CaridFrameComponent {
  void *data;
  int stride;
  int width;
  int height;
  int length;
};

struct _CaridFrame {
  CaridBuffer *buffer;

  CaridFrameFreeFunc free;
  void *regions[3];
  void *priv;

  CaridFrameFormat format;

  CaridFrameComponent components[3];

  uint32_t frame_number;
};


CaridFrame * carid_frame_new (void);
CaridFrame * carid_frame_new_and_alloc (CaridFrameFormat format, int width,
    int height, int sub_x, int sub_y);
CaridFrame * carid_frame_new_I420 (void *data, int width, int height);
void carid_frame_set_free_callback (CaridFrame *frame,
    CaridFrameFreeFunc free_func, void *priv);
void carid_frame_free (CaridFrame *frame);
void carid_frame_convert (CaridFrame *dest, CaridFrame *src);


#endif

