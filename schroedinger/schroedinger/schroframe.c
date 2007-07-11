

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

#include <stdlib.h>
#include <string.h>

SchroFrame *
schro_frame_new (void)
{
  SchroFrame *frame;

  frame = malloc (sizeof(*frame));
  memset (frame, 0, sizeof(*frame));
  frame->refcount = 1;

  return frame;
}

SchroFrame *
schro_frame_new_and_alloc (SchroFrameFormat format, int width, int height)
{
  SchroFrame *frame = schro_frame_new();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  
  SCHRO_ASSERT(width > 0);
  SCHRO_ASSERT(height > 0);

  frame->format = format;
  frame->width = width;
  frame->height = height;

  switch (SCHRO_FRAME_FORMAT_DEPTH(format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      bytes_pp = 1;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      bytes_pp = 2;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      bytes_pp = 4;
      break;
    default:
      SCHRO_ASSERT(0);
      bytes_pp = 0;
      break;
  }

  h_shift = SCHRO_FRAME_FORMAT_H_SHIFT(format);
  v_shift = SCHRO_FRAME_FORMAT_V_SHIFT(format);
  chroma_width = ROUND_UP_SHIFT(width, h_shift);
  chroma_height = ROUND_UP_SHIFT(height, v_shift);

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_4(width * bytes_pp);
  frame->components[0].length = 
    frame->components[0].stride * frame->components[0].height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride = ROUND_UP_4(chroma_width * bytes_pp);
  frame->components[1].length = 
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride = ROUND_UP_4(chroma_width * bytes_pp);
  frame->components[2].length = 
    frame->components[2].stride * frame->components[2].height;
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  frame->regions[0] = malloc (frame->components[0].length +
      frame->components[1].length + frame->components[2].length);

  frame->components[0].data = frame->regions[0];
  frame->components[1].data = frame->components[0].data +
    frame->components[0].length;
  frame->components[2].data = frame->components[1].data +
    frame->components[1].length;

  return frame;
}

SchroFrame *
schro_frame_new_from_data_YUY2 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new();

  frame->format = SCHRO_FRAME_FORMAT_YUYV;

  frame->width = width;
  frame->height = height;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2(width,1) * 2;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

SchroFrame *
schro_frame_new_from_data_AYUV (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new();

  frame->format = SCHRO_FRAME_FORMAT_AYUV;

  frame->width = width;
  frame->height = height;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = width * 4;
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride * height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  return frame;
}

SchroFrame *
schro_frame_new_from_data_I420 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new();

  frame->format = SCHRO_FRAME_FORMAT_U8_420;

  frame->width = width;
  frame->height = height;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2(width,2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
    ROUND_UP_POW2(frame->components[0].height,1);
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].width = ROUND_UP_SHIFT(width,1);
  frame->components[1].height = ROUND_UP_SHIFT(height,1);
  frame->components[1].stride = ROUND_UP_POW2(frame->components[1].width,2);
  frame->components[1].length =
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].data =
    frame->components[0].data + frame->components[0].length; 
  frame->components[1].v_shift = 1;
  frame->components[1].h_shift = 1;

  frame->components[2].width = ROUND_UP_SHIFT(width,1);
  frame->components[2].height = ROUND_UP_SHIFT(height,1);
  frame->components[2].stride = ROUND_UP_POW2(frame->components[2].width,2);
  frame->components[2].length =
    frame->components[2].stride * frame->components[2].height;
  frame->components[2].data =
    frame->components[1].data + frame->components[1].length; 
  frame->components[2].v_shift = 1;
  frame->components[2].h_shift = 1;

  return frame;
}

SchroFrame *
schro_frame_ref (SchroFrame *frame)
{
  frame->refcount++;
  return frame;
}

void
schro_frame_unref (SchroFrame *frame)
{
  frame->refcount--;
  if (frame->refcount == 0) {
    if (frame->free) {
      frame->free (frame, frame->priv);
    }
    if (frame->regions[0]) {
      free(frame->regions[0]);
    }

    free(frame);
  }
}

void schro_frame_set_free_callback (SchroFrame *frame,
    SchroFrameFreeFunc free_func, void *priv)
{
  frame->free = free_func;
  frame->priv = priv;
}

static void schro_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_u8_422_yuyv (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_u8_422_uyvy (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_u8_444_ayuv (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_yuyv_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_uyvy_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_ayuv_u8_444 (SchroFrame *dest, SchroFrame *src);

typedef void (*SchroFrameBinaryFunc) (SchroFrame *dest, SchroFrame *src);

struct binary_struct {
  SchroFrameFormat from;
  SchroFrameFormat to;
  SchroFrameBinaryFunc func;
};
static struct binary_struct schro_frame_convert_func_list[] = {
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, schro_frame_convert_u8_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, schro_frame_convert_u8_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, schro_frame_convert_u8_s16 },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, schro_frame_convert_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, schro_frame_convert_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, schro_frame_convert_s16_u8 },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_U8_444, schro_frame_convert_u8_u8 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_U8_422, schro_frame_convert_u8_u8 },
  { SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_U8_420, schro_frame_convert_u8_u8 },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, schro_frame_convert_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, schro_frame_convert_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, schro_frame_convert_s16_s16 },

  { SCHRO_FRAME_FORMAT_YUYV, SCHRO_FRAME_FORMAT_U8_422, schro_frame_convert_u8_422_yuyv },
  { SCHRO_FRAME_FORMAT_UYVY, SCHRO_FRAME_FORMAT_U8_422, schro_frame_convert_u8_422_uyvy },
  { SCHRO_FRAME_FORMAT_AYUV, SCHRO_FRAME_FORMAT_U8_444, schro_frame_convert_u8_444_ayuv },

  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV, schro_frame_convert_yuyv_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_UYVY, schro_frame_convert_uyvy_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_AYUV, schro_frame_convert_ayuv_u8_444 },
  { 0 }
};

void
schro_frame_convert (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  dest->frame_number = src->frame_number;

  for(i=0;schro_frame_convert_func_list[i].func;i++){
    if (schro_frame_convert_func_list[i].from == src->format &&
        schro_frame_convert_func_list[i].to == dest->format) {
      schro_frame_convert_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR("conversion unimplemented");
  SCHRO_ASSERT(0);
}

static void schro_frame_add_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_add_s16_u8 (SchroFrame *dest, SchroFrame *src);

static struct binary_struct schro_frame_add_func_list[] = {
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, schro_frame_add_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, schro_frame_add_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, schro_frame_add_s16_s16 },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, schro_frame_add_s16_u8 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, schro_frame_add_s16_u8 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, schro_frame_add_s16_u8 },

  { 0 }
};

void
schro_frame_add (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  for(i=0;schro_frame_add_func_list[i].func;i++){
    if (schro_frame_add_func_list[i].from == src->format &&
        schro_frame_add_func_list[i].to == dest->format) {
      schro_frame_add_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR("add function unimplemented");
  SCHRO_ASSERT(0);
}

static void schro_frame_subtract_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_subtract_s16_u8 (SchroFrame *dest, SchroFrame *src);

static struct binary_struct schro_frame_subtract_func_list[] = {
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, schro_frame_subtract_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, schro_frame_subtract_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, schro_frame_subtract_s16_s16 },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, schro_frame_subtract_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, schro_frame_subtract_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, schro_frame_subtract_s16_u8 },

  { 0 }
};

void
schro_frame_subtract (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  for(i=0;schro_frame_subtract_func_list[i].func;i++){
    if (schro_frame_subtract_func_list[i].from == src->format &&
        schro_frame_subtract_func_list[i].to == dest->format) {
      schro_frame_subtract_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR(0);
  SCHRO_ASSERT("subtract function unimplemented");
}

static void
schro_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  uint8_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;

  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(dest->format) == SCHRO_FRAME_FORMAT_DEPTH_U8);
  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(src->format) == SCHRO_FRAME_FORMAT_DEPTH_S16);

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      oil_convert_u8_s16(ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  uint8_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      memcpy (ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
schro_frame_convert_s16_s16 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      memcpy (ddata, sdata, width * sizeof(int16_t));
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
schro_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      oil_convert_s16_u8(ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
unmix_yuyv (uint8_t *y, uint8_t *u, uint8_t *v, uint32_t *src, int n)
{
  int i;
  uint8_t *s = (uint8_t *)src;

  for(i=0;i<n;i++){
    y[i*2+0] = s[i*4 + 0];
    y[i*2+1] = s[i*4 + 2];
    u[i] = s[i*4 + 1];
    v[i] = s[i*4 + 3];
  }
}

static void
schro_frame_convert_u8_422_yuyv (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *sdata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int width, height;
  int n;
  int y;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  n = ROUND_UP_SHIFT(width,1);
  for(y=0;y<height;y++){
    sdata = OFFSET(src->components[0].data, src->components[0].stride * y);
    ydata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    udata = OFFSET(dest->components[1].data, dest->components[1].stride * y);
    vdata = OFFSET(dest->components[2].data, dest->components[2].stride * y);

    unmix_yuyv (ydata, udata, vdata, sdata, n);
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
unmix_uyvy (uint8_t *y, uint8_t *u, uint8_t *v, uint32_t *src, int n)
{
  int i;
  uint8_t *s = (uint8_t *)src;

  for(i=0;i<n;i++){
    y[i*2+0] = s[i*4 + 1];
    y[i*2+1] = s[i*4 + 3];
    u[i] = s[i*4 + 0];
    v[i] = s[i*4 + 2];
  }
}

static void
schro_frame_convert_u8_422_uyvy (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *sdata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int y;
  int width, height;
  int n;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  n = ROUND_UP_SHIFT(width,1);
  for(y=0;y<height;y++){
    sdata = OFFSET(src->components[0].data, src->components[0].stride * y);
    ydata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    udata = OFFSET(dest->components[1].data, dest->components[1].stride * y);
    vdata = OFFSET(dest->components[2].data, dest->components[2].stride * y);

    unmix_uyvy (ydata, udata, vdata, sdata, n);
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
unmix_ayuv (uint8_t *y, uint8_t *u, uint8_t *v, uint32_t *src, int n)
{
  int i;
  uint8_t *s = (uint8_t *)src;

  for(i=0;i<n;i++){
    y[i] = s[i*4 + 1];
    u[i] = s[i*4 + 2];
    v[i] = s[i*4 + 3];
  }
}

static void
schro_frame_convert_u8_444_ayuv (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *sdata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int y;
  int width, height;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  for(y=0;y<height;y++){
    sdata = OFFSET(src->components[0].data, src->components[0].stride * y);
    ydata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    udata = OFFSET(dest->components[1].data, dest->components[1].stride * y);
    vdata = OFFSET(dest->components[2].data, dest->components[2].stride * y);

    unmix_ayuv (ydata, udata, vdata, sdata, width);
  }

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
mix_yuyv (uint32_t *dest, uint8_t *y, uint8_t *u, uint8_t *v, int n)
{
  int i;
  uint8_t *d = (uint8_t *)dest;

  for(i=0;i<n;i++){
    d[i*4 + 0] = y[i*2+0];
    d[i*4 + 2] = y[i*2+1];
    d[i*4 + 1] = u[i];
    d[i*4 + 3] = v[i];
  }
}

static void
schro_frame_convert_yuyv_u8_422 (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *ddata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int width, height;
  int n;
  int y;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  n = ROUND_UP_SHIFT(width,1);
  for(y=0;y<height;y++){
    ddata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    ydata = OFFSET(src->components[0].data, src->components[0].stride * y);
    udata = OFFSET(src->components[1].data, src->components[1].stride * y);
    vdata = OFFSET(src->components[2].data, src->components[2].stride * y);

    mix_yuyv (ddata, ydata, udata, vdata, n);
  }

  /* FIXME edge extend */
}

static void
mix_uyvy (uint32_t *dest, uint8_t *y, uint8_t *u, uint8_t *v, int n)
{
  int i;
  uint8_t *d = (uint8_t *)dest;

  for(i=0;i<n;i++){
    d[i*4 + 1] = y[i*2+0];
    d[i*4 + 3] = y[i*2+1];
    d[i*4 + 0] = u[i];
    d[i*4 + 2] = v[i];
  }
}

static void
schro_frame_convert_uyvy_u8_422 (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *ddata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int width, height;
  int n;
  int y;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  n = ROUND_UP_SHIFT(width,1);
  for(y=0;y<height;y++){
    ddata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    ydata = OFFSET(src->components[0].data, src->components[0].stride * y);
    udata = OFFSET(src->components[1].data, src->components[1].stride * y);
    vdata = OFFSET(src->components[2].data, src->components[2].stride * y);

    mix_uyvy (ddata, ydata, udata, vdata, n);
  }

  /* FIXME edge extend */
  //schro_frame_edge_extend (dest, src->width, src->height);
}

static void
mix_ayuv (uint32_t *dest, uint8_t *y, uint8_t *u, uint8_t *v, int n)
{
  int i;
  uint8_t *d = (uint8_t *)dest;

  for(i=0;i<n;i++){
    d[i*4 + 0] = 0xff;
    d[i*4 + 1] = y[i];
    d[i*4 + 2] = u[i];
    d[i*4 + 3] = v[i];
  }
}

static void
schro_frame_convert_ayuv_u8_444 (SchroFrame *dest, SchroFrame *src)
{
  uint32_t *ddata;
  uint8_t *ydata;
  uint8_t *udata;
  uint8_t *vdata;
  int width, height;
  int y;

  width = MIN(src->width, dest->width);
  height = MIN(src->height, dest->height);
  for(y=0;y<height;y++){
    ddata = OFFSET(dest->components[0].data, dest->components[0].stride * y);
    ydata = OFFSET(src->components[0].data, src->components[0].stride * y);
    udata = OFFSET(src->components[1].data, src->components[1].stride * y);
    vdata = OFFSET(src->components[2].data, src->components[2].stride * y);

    mix_ayuv (ddata, ydata, udata, vdata, width);
  }

  /* FIXME edge extend */
  //schro_frame_edge_extend (dest, src->width, src->height);
}


static void
schro_frame_add_s16_s16 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = (dcomp->width < scomp->width) ? dcomp->width : scomp->width;
    height = (dcomp->height < scomp->height) ? dcomp->height : scomp->height;

    for(y=0;y<height;y++){
      oil_add_s16 (ddata, ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }
}

static void
schro_frame_add_s16_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = (dcomp->width < scomp->width) ? dcomp->width : scomp->width;
    height = (dcomp->height < scomp->height) ? dcomp->height : scomp->height;

    for(y=0;y<height;y++){
      oil_add_s16_u8 (ddata, ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }
}

static void
schro_frame_subtract_s16_s16 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = (dcomp->width < scomp->width) ? dcomp->width : scomp->width;
    height = (dcomp->height < scomp->height) ? dcomp->height : scomp->height;

    for(y=0;y<height;y++){
      oil_subtract_s16 (ddata, ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }
}

static void
schro_frame_subtract_s16_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  int16_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = (dcomp->width < scomp->width) ? dcomp->width : scomp->width;
    height = (dcomp->height < scomp->height) ? dcomp->height : scomp->height;

    for(y=0;y<height;y++){
      oil_subtract_s16_u8 (ddata, ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }
}

void
schro_frame_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp)
{
  int16_t *frame_data;
  int component;
  int width;
  int height;
  int level;

  //SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16_420);

  for(component=0;component<3;component++){
    SchroFrameComponent *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)comp->data;
    for(level=0;level<params->transform_depth;level++) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;

      schro_wavelet_transform_2d (params->wavelet_filter_index,
          frame_data, stride, w, h, tmp);
    }
  }
}

void
schro_frame_inverse_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp)
{
  int16_t *frame_data;
  int width;
  int height;
  int level;
  int component;

  //SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16_420);

  for(component=0;component<3;component++){
    SchroFrameComponent *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)comp->data;
    for(level=params->transform_depth-1; level >=0;level--) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;

      schro_wavelet_inverse_transform_2d (params->wavelet_filter_index,
          frame_data, stride, w, h, tmp);
    }
  }
}


void schro_frame_shift_left (SchroFrame *frame, int shift)
{
  SchroFrameComponent *comp;
  int16_t *data;
  int i;
  int y;
  int16_t x = shift;

  for(i=0;i<3;i++){
    comp = &frame->components[i];
    data = comp->data;

    for(y=0;y<comp->height;y++){
      oil_lshift_s16 (data, data, &x, comp->width);
      data = OFFSET(data, comp->stride);
    }
  }
}

void schro_frame_shift_right (SchroFrame *frame, int shift)
{
  SchroFrameComponent *comp;
  int16_t *data;
  int i;
  int y;
  int16_t s[2] = { (1<<shift)>>1, shift };

  for(i=0;i<3;i++){
    comp = &frame->components[i];
    data = comp->data;

    for(y=0;y<comp->height;y++){
      oil_add_const_rshift_s16 (data, data, s, comp->width);
      data = OFFSET(data, comp->stride);
    }
  }
}


void
schro_frame_edge_extend (SchroFrame *frame, int width, int height)
{
  SchroFrameComponent *comp;
  int i;
  int y;
  int chroma_width;
  int chroma_height;

  SCHRO_DEBUG("extending %d %d -> %d %d", width, height,
      frame->width, frame->height);

  chroma_width = ROUND_UP_SHIFT(width,
      SCHRO_FRAME_FORMAT_H_SHIFT(frame->format));
  chroma_height = ROUND_UP_SHIFT(height,
      SCHRO_FRAME_FORMAT_V_SHIFT(frame->format));

  SCHRO_DEBUG("chroma %d %d -> %d %d", chroma_width, chroma_height,
      frame->components[1].width, frame->components[1].height);
  
  switch(SCHRO_FRAME_FORMAT_DEPTH(frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for(i=0;i<3;i++){
        uint8_t *data;
        int w,h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i>0) ? chroma_width : width;
        h = (i>0) ? chroma_height : height;

        if (w < comp->width) {
          for(y = 0; y<MIN(h,comp->height); y++) {
            data = OFFSET(comp->data, comp->stride * y);
            oil_splat_u8_ns (data + w, data + w - 1, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_memcpy (OFFSET(comp->data, comp->stride * y),
              OFFSET(comp->data, comp->stride * (h-1)), comp->width);
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for(i=0;i<3;i++){
        int16_t *data;
        int w,h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i>0) ? chroma_width : width;
        h = (i>0) ? chroma_height : height;

        if (w < comp->width) {
          for(y = 0; y<MIN(h,comp->height); y++) {
            data = OFFSET(comp->data, comp->stride * y);
            oil_splat_s16_ns (data + w, data + w - 1, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_memcpy (OFFSET(comp->data, comp->stride * y),
              OFFSET(comp->data, comp->stride * (h-1)), comp->width * 2);
        }
      }
      break;
    default:
      SCHRO_ERROR("unimplemented case");
      SCHRO_ASSERT(0);
      break;
  }
}

void
schro_frame_zero_extend (SchroFrame *frame, int width, int height)
{
  SchroFrameComponent *comp;
  int i;
  int y;
  int chroma_width;
  int chroma_height;

  SCHRO_DEBUG("extending %d %d -> %d %d", width, height,
      frame->width, frame->height);

  chroma_width = ROUND_UP_SHIFT(width,
      SCHRO_FRAME_FORMAT_H_SHIFT(frame->format));
  chroma_height = ROUND_UP_SHIFT(height,
      SCHRO_FRAME_FORMAT_V_SHIFT(frame->format));

  switch(SCHRO_FRAME_FORMAT_DEPTH(frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for(i=0;i<3;i++){
        uint8_t zero = 0;
        uint8_t *data;
        int w,h;

        comp = &frame->components[i];
        data = comp->data;

        w = (i>0) ? chroma_width : width;
        h = (i>0) ? chroma_height : height;
        
        if (w < comp->width) {
          for(y = 0; y<h; y++) {
            data = OFFSET(comp->data, comp->stride * y);
            oil_splat_u8_ns (data + w, &zero, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_splat_u8_ns (OFFSET(comp->data, comp->stride * y), &zero,
              comp->width);
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for(i=0;i<3;i++){
        int16_t *data;
        int w,h;
        int16_t zero = 0;

        comp = &frame->components[i];
        data = comp->data;

        w = (i>0) ? chroma_width : width;
        h = (i>0) ? chroma_height : height;
        
        if (w < comp->width) {
          for(y = 0; y<h; y++) {
            data = OFFSET(comp->data, comp->stride * y);
            oil_splat_s16_ns (data + w, &zero, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_splat_s16_ns (OFFSET(comp->data, comp->stride * y), &zero,
              comp->width);
        }
      }
      break;
    default:
      SCHRO_ERROR("unimplemented case");
      break;
  }
}

void
notoil_downsample_horiz_u8 (uint8_t *dest, uint8_t *src, int n)
{
  static const int taps[12] = { 4, -4, -8, 4, 46, 86, 86, 46, 4, -8, -4, 4 };
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<12;j++){
      x += taps[j]*src[CLAMP(i*2 + j - 5, 0, n*2-1)];
    }
    dest[i] = CLAMP((x + 128) >> 8,0,255);
  }
}

void
notoil_downsample_vert_u8 (uint8_t *dest, uint8_t *src[], int n)
{
  static const int taps[12] = { 4, -4, -8, 4, 46, 86, 86, 46, 4, -8, -4, 4 };
  int i;
  int j;
  int x;

  for(i=0;i<n;i++){
    x = 0;
    for(j=0;j<12;j++){
      x += taps[j]*src[j][i];
    }
    dest[i] = CLAMP((x + 128) >> 8,0,255);
  }
}

void
schro_frame_component_downsample (SchroFrameComponent *dest,
    SchroFrameComponent *src)
{
  int i,j;
  uint8_t *tmp, *tmp0, *tmp1;
  uint8_t *tmplist[12];

  tmp = malloc(dest->width * 12);
  for(i=0;i<12;i++){
    tmplist[i] = tmp + dest->width * i;
  }

  for(i=0;i<7;i++){
    notoil_downsample_horiz_u8 (tmplist[i+5], src->data + src->stride * i,
        dest->width);
  }
  for(i=0;i<5;i++){
    memcpy (tmplist[i], tmplist[5], dest->width);
  }
  notoil_downsample_vert_u8 (dest->data + dest->stride * 0, tmplist,
      dest->width);

  for (j=1;j<dest->height;j++){
    tmp0 = tmplist[0];
    tmp1 = tmplist[1];
    for(i=0;i<10;i++){
      tmplist[i] = tmplist[i+2];
    }
    tmplist[10] = tmp0;
    tmplist[11] = tmp1;

    notoil_downsample_horiz_u8 (tmplist[10],
        src->data + src->stride * CLAMP(j*2+5,0,src->height-1), dest->width);
    notoil_downsample_horiz_u8 (tmplist[11],
        src->data + src->stride * CLAMP(j*2+6,0,src->height-1), dest->width);
    
    notoil_downsample_vert_u8 (dest->data + dest->stride * j, tmplist,
        dest->width);
  }

  free (tmp);
}

void
schro_frame_downsample (SchroFrame *dest, SchroFrame *src)
{
  schro_frame_component_downsample (&dest->components[0],
      &src->components[0]);
  schro_frame_component_downsample (&dest->components[1],
      &src->components[1]);
  schro_frame_component_downsample (&dest->components[2],
      &src->components[2]);
}

#if 0
void
notoil_downsample2x2_u8 (uint8_t *dest, uint8_t *src1, uint8_t *src2,
    int n)
{
  int i;
  for(i=0;i<n;i++){
    dest[i] = (src1[i*2] + src1[i*2+1] + src2[i*2] + src2[i*2 + 1] + 2)>>2;
  }
}

void
schro_frame_downsample (SchroFrame *dest, SchroFrame *src, int shift)
{
  int j, k;
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;

  SCHRO_ASSERT(shift == 1);

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH(src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR("unimplemented");
    SCHRO_ASSERT(0);
    return;
  }

  SCHRO_ASSERT(ROUND_UP_SHIFT(src->width,1) == dest->width);
  SCHRO_ASSERT(ROUND_UP_SHIFT(src->height,1) == dest->height);

  for(k=0;k<3;k++){
    uint8_t *sdata;
    uint8_t *ddata;

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    sdata = scomp->data;
    ddata = dcomp->data;
    for(j=0;j<scomp->height/2;j++){
      notoil_downsample2x2_u8 (ddata + dcomp->stride * j,
          sdata + scomp->stride * j * 2,
          sdata + scomp->stride * (j * 2 + 1),
          scomp->width/2);
    }
    if (dcomp->height > scomp->height/2) {
      notoil_downsample2x2_u8 (ddata + dcomp->stride * (dcomp->height - 1),
          sdata + scomp->stride * (scomp->height - 1),
          sdata + scomp->stride * (scomp->height - 1),
          dcomp->width/2);
    }
    if (dcomp->width/2 < scomp->width) {
      for(j = 0; j< scomp->height/2; j++){
        ddata[dcomp->stride * j + dcomp->width - 1] =
          (sdata[scomp->stride * (j*2) + scomp->width - 1] + 
          sdata[scomp->stride * (j*2 + 1) + scomp->width - 1] + 1)/2;
      }
      if (dcomp->height > scomp->height/2) {
        ddata[dcomp->stride * (dcomp->height - 1) + dcomp->width - 1] =
          sdata[scomp->stride * (scomp->height - 1) + scomp->width - 1];
      }
    }
  }
}
#endif

void
schro_frame_upsample_horiz (SchroFrame *dest, SchroFrame *src)
{
  int i, j, k, l;
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH(src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR("unimplemented");
    return;
  }

  for(k=0;k<3;k++){
    static const int taps[10] = { 3, -11, 25, -56, 167, 167, -56, 25, -11, 3 };
    uint8_t *sdata;
    uint8_t *ddata;
    int x;

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    sdata = scomp->data;
    ddata = dcomp->data;

    for(j=0;j<dcomp->height;j++){
      for(i=0;i<4;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * j +
            CLAMP(i - 4 + l,0,scomp->width-1)];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
      for(i=4;i<dcomp->width-5;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * j + i - 4 + l];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
      for(;i<dcomp->width;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * j +
            CLAMP(i - 4 + l,0,scomp->width-1)];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
    }
  }
}

void
schro_frame_upsample_vert (SchroFrame *dest, SchroFrame *src)
{
  int i, j, k, l;
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH(src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR("unimplemented");
    return;
  }

  for(k=0;k<3;k++){
    static const int taps[10] = { 3, -11, 25, -56, 167, 167, -56, 25, -11, 3 };
    uint8_t *sdata;
    uint8_t *ddata;
    int x;

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    sdata = scomp->data;
    ddata = dcomp->data;

    for(j=0;j<4;j++){
      for(i=0;i<dcomp->width;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * CLAMP(j - 4 + l,
              0, scomp->height-1) + i];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
    }
    for(j=4;j<dcomp->height-5;j++){
      for(i=0;i<dcomp->width;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * (j - 4 + l) + i];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
    }
    for(j=dcomp->height-5;j<dcomp->height;j++){
      for(i=0;i<dcomp->width;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * CLAMP(j - 4 + l,
              0, dcomp->height-1) + i];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
    }
  }
}

int
schro_frame_calculate_average_luma (SchroFrame *frame)
{
  SchroFrameComponent *comp;
  int i,j;
  int sum = 0;
  int n;

  comp = &frame->components[0];

  switch (SCHRO_FRAME_FORMAT_DEPTH(frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      {
        uint8_t *data;
        data = comp->data;
        for(j=0;j<comp->height;j++){
          for(i=0;i<comp->width;i++){
            sum += data[comp->stride * j + i];
          }
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      {
        int16_t *data;
        for(j=0;j<comp->height;j++){
          data = OFFSET(comp->data, comp->stride * j);
          for(i=0;i<comp->width;i++){
            sum += data[i];
          }
        }
      }
      break;
    default:
      SCHRO_ERROR ("unimplemented");
      break;
  }

  n = comp->height * comp->width;
  return (sum + n/2) / n;
}

static void
schro_frame_component_planar_copy_u8 (SchroFrameComponent *dest,
    SchroFrameComponent *src)
{
  int j;

  for(j=0;j<dest->height;j++) {
    memcpy (dest->data + dest->stride * j, src->data + src->stride * j,
        dest->width);
  }
}

static void
horiz_upsample (uint8_t *d, uint8_t *s, int n)
{
  int i;

  d[0] = s[0];

  for (i = 0; i < n-3; i+=2) {
    d[i + 1] = (3*s[i/2] + s[i/2+1] + 2)>>2;
    d[i + 2] = (s[i/2] + 3*s[i/2+1] + 2)>>2;
  }

  if (n&1) {
    i = n-3;
    d[n-2] = s[n/2];
    d[n-1] = s[n/2];
  } else {
    d[n-1] = s[n/2-1];
  }
}

static void
schro_frame_component_convert_420_to_444 (SchroFrameComponent *dest,
    SchroFrameComponent *src)
{
  int j;
  uint8_t *tmp;
  uint32_t weight = 128;

  SCHRO_ASSERT(dest->height <= src->height * 2);
  SCHRO_ASSERT(dest->width <= src->width * 2);

  tmp = malloc (src->width);
  for(j=0;j<dest->height;j++) {
    if (j&1) {
      oil_merge_linear_u8 (tmp,
          src->data + src->stride * ((j-1)>>1),
          src->data + src->stride * ((j+1)>>1),
          &weight,
          src->width);
      horiz_upsample (dest->data + dest->stride * j,
          tmp, dest->width);
    } else {
      horiz_upsample (dest->data + dest->stride * j,
          src->data + src->stride * (j>>1), dest->width);
    }
  }
  free(tmp);
}

SchroFrame *
schro_frame_convert_to_444 (SchroFrame *frame)
{
  SchroFrame *dest;

  SCHRO_ASSERT (frame->format == SCHRO_FRAME_FORMAT_U8_420);
  
  dest = schro_frame_new_and_alloc (SCHRO_FRAME_FORMAT_U8_444,
      frame->width, frame->height);

  schro_frame_component_planar_copy_u8 (&dest->components[0],
      &frame->components[0]);
  schro_frame_component_convert_420_to_444 (&dest->components[1],
      &frame->components[1]);
  schro_frame_component_convert_420_to_444 (&dest->components[2],
      &frame->components[2]);

  return dest;
}

