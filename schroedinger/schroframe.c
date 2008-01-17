

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/schrocog.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

#include <stdlib.h>
#include <string.h>

SchroFrame *
schro_frame_new (void)
{
  SchroFrame *frame;

  frame = schro_malloc0 (sizeof(*frame));
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

  frame->regions[0] = schro_memory_domain_alloc (NULL,
      frame->components[0].length +
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
schro_frame_new_from_data_UYVY (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new();

  frame->format = SCHRO_FRAME_FORMAT_UYVY;

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
schro_frame_new_from_data_YV12 (void *data, int width, int height)
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

  frame->components[2].width = ROUND_UP_SHIFT(width,1);
  frame->components[2].height = ROUND_UP_SHIFT(height,1);
  frame->components[2].stride = ROUND_UP_POW2(frame->components[2].width,2);
  frame->components[2].length =
    frame->components[2].stride * frame->components[2].height;
  frame->components[2].data =
    frame->components[0].data + frame->components[0].length; 
  frame->components[2].v_shift = 1;
  frame->components[2].h_shift = 1;

  frame->components[1].width = ROUND_UP_SHIFT(width,1);
  frame->components[1].height = ROUND_UP_SHIFT(height,1);
  frame->components[1].stride = ROUND_UP_POW2(frame->components[1].width,2);
  frame->components[1].length =
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].data =
    frame->components[2].data + frame->components[2].length; 
  frame->components[1].v_shift = 1;
  frame->components[1].h_shift = 1;

  return frame;
}

SchroFrame *
schro_frame_dup (SchroFrame *frame)
{
  SchroFrame *dup_frame;

  dup_frame = schro_frame_new_and_alloc (frame->format, frame->width,
      frame->height);
  schro_frame_convert (dup_frame, frame);

  return dup_frame;
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
  SCHRO_ASSERT(frame->refcount > 0);
  frame->refcount--;
  if (frame->refcount == 0) {
    if (frame->free) {
      frame->free (frame, frame->priv);
    }
    if (frame->regions[0]) {
      schro_memory_domain_memfree(NULL, frame->regions[0]);
    }
    if (frame->is_cuda_frame) {
#ifdef HAVE_CUDA
      _schro_gpuframe_free (frame);
#else
      SCHRO_ASSERT(0);
#endif
    }

    schro_free(frame);
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

  for(i=0;schro_frame_convert_func_list[i].func;i++){
    if (schro_frame_convert_func_list[i].from == src->format &&
        schro_frame_convert_func_list[i].to == dest->format) {
      schro_frame_convert_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR("conversion unimplemented (%d -> %d)",
      src->format, dest->format);
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  uint8_t *ddata;
  int16_t *sdata;
  int i;
  int y;
  int width, height;
  int16_t *tmp;
  int16_t c = 128;

  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(dest->format) == SCHRO_FRAME_FORMAT_DEPTH_U8);
  SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(src->format) == SCHRO_FRAME_FORMAT_DEPTH_S16);

  tmp = schro_malloc(dest->width*sizeof(int16_t));
  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      oil_addc_s16 (tmp, sdata, &c, width);
      oil_convert_u8_s16 (ddata, tmp, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
  }
  schro_free(tmp);

  schro_frame_edge_extend (dest, src->width, src->height);
}

static void
schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
  int16_t *ddata;
  uint8_t *sdata;
  int i;
  int y;
  int width, height;
  int16_t c = -128;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    width = MIN(dcomp->width, scomp->width);
    height = MIN(dcomp->height, scomp->height);

    for(y=0;y<height;y++){
      oil_convert_s16_u8(ddata, sdata, width);
      oil_addc_s16(ddata, ddata, &c, width);
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

    oil_packyuyv (ddata, ydata, udata, vdata, n);
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  SchroFrameData *dcomp;
  SchroFrameData *scomp;
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
  int component;
  int width;
  int height;
  int level;

  //SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16_420);

  for(component=0;component<3;component++){
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    for(level=0;level<params->transform_depth;level++) {
      SchroFrameData fd;

      fd.format = frame->format;
      fd.data = comp->data;
      fd.width = width >> level;
      fd.height = height >> level;
      fd.stride = comp->stride << level;

      schro_wavelet_transform_2d (&fd, params->wavelet_filter_index,
          tmp);
    }
  }
}

void
schro_frame_inverse_iwt_transform (SchroFrame *frame, SchroParams *params,
    int16_t *tmp)
{
  int width;
  int height;
  int level;
  int component;

  //SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16_420);

  for(component=0;component<3;component++){
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    for(level=params->transform_depth-1; level >=0;level--) {
      SchroFrameData fd;

      fd.format = frame->format;
      fd.data = comp->data;
      fd.width = width >> level;
      fd.height = height >> level;
      fd.stride = comp->stride << level;

      schro_wavelet_inverse_transform_2d (&fd, params->wavelet_filter_index,
          tmp);
    }
  }
}


void schro_frame_shift_left (SchroFrame *frame, int shift)
{
  SchroFrameData *comp;
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
  SchroFrameData *comp;
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
  SchroFrameData *comp;
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
  SchroFrameData *comp;
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

static void
mas12_edgeextend_u8 (uint8_t *dest, uint8_t *src, const int16_t *taps,
    const int16_t *offsetshift, int n, int i)
{
  int j;
  int x;

  x = 0;
  for(j=0;j<12;j++){
    //x += taps[j]*src[CLAMP(i*2 + j - 5, 0, n-1)];
    x += taps[j]*src[CLAMP(i + j, 0, n-1)];
  }
  dest[0] = CLAMP((x + 128) >> 8,0,255);
}

void
downsample_horiz_u8 (uint8_t *dest, int n_dest, uint8_t *src, int n_src,
    const int16_t *taps, const int16_t *offsetshift)
{
  int i;

  if (n_dest < 7) {
    for(i=0;i<n_dest;i++){
      mas12_edgeextend_u8 (dest + i, src, taps, offsetshift, n_src, 2*i - 5);
    }
  } else {
    mas12_edgeextend_u8 (dest + 0, src, taps, offsetshift, n_src, -5);
    mas12_edgeextend_u8 (dest + 1, src, taps, offsetshift, n_src, -3);
    mas12_edgeextend_u8 (dest + 2, src, taps, offsetshift, n_src, -1);

    oil_mas12_addc_rshift_decim2_u8 (dest + 3, src + 1, taps, offsetshift,
        n_dest - 7);

    mas12_edgeextend_u8 (dest + n_dest - 4, src, taps, offsetshift, n_src, 2*n_dest - 13);
    mas12_edgeextend_u8 (dest + n_dest - 3, src, taps, offsetshift, n_src, 2*n_dest - 11);
    mas12_edgeextend_u8 (dest + n_dest - 2, src, taps, offsetshift, n_src, 2*n_dest - 9);
    mas12_edgeextend_u8 (dest + n_dest - 1, src, taps, offsetshift, n_src, 2*n_dest - 7);
  }

}

void
schro_frame_component_downsample (SchroFrameData *dest,
    SchroFrameData *src)
{
  const int16_t taps[12] = { 4, -4, -8, 4, 46, 86, 86, 46, 4, -8, -4, 4 };
  const int16_t offsetshift[2] = { 128, 8 };
  int i,j;
  uint8_t *tmp, *tmp0, *tmp1;
  uint8_t *tmplist[12];

  tmp = schro_malloc(dest->width * 12);
  for(i=0;i<12;i++){
    tmplist[i] = tmp + dest->width * i;
  }

  for(i=0;i<7;i++){
    downsample_horiz_u8 (tmplist[i+5], dest->width,
        src->data + src->stride * CLAMP(i, 0, src->height - 1), src->width,
        taps, offsetshift);
  }
  for(i=0;i<5;i++){
    memcpy (tmplist[i], tmplist[5], dest->width);
  }
  oil_mas12across_addc_rshift_u8 (dest->data + dest->stride * 0, tmplist,
      taps, offsetshift, dest->width);

  for (j=1;j<dest->height;j++){
    tmp0 = tmplist[0];
    tmp1 = tmplist[1];
    for(i=0;i<10;i++){
      tmplist[i] = tmplist[i+2];
    }
    tmplist[10] = tmp0;
    tmplist[11] = tmp1;

    downsample_horiz_u8 (tmplist[10], dest->width,
        src->data + src->stride * CLAMP(j*2+5,0,src->height-1), src->width,
        taps, offsetshift);
    downsample_horiz_u8 (tmplist[11], dest->width,
        src->data + src->stride * CLAMP(j*2+6,0,src->height-1), src->width,
        taps, offsetshift);
    
    oil_mas12across_addc_rshift_u8 (dest->data + dest->stride * j, tmplist,
        taps, offsetshift, dest->width);
  }

  schro_free (tmp);
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

void
schro_frame_upsample_horiz (SchroFrame *dest, SchroFrame *src)
{
  int j, k;
  SchroFrameData *dcomp;
  SchroFrameData *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH(src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR("unimplemented");
    return;
  }

  for(k=0;k<3;k++){
    static const int16_t taps[8] = { -1, 3, -7, 21, 21, -7, 3, -1 };

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    for(j=0;j<dcomp->height;j++){
      schro_cog_mas8_u8_edgeextend (
          OFFSET(dcomp->data, dcomp->stride * j),
          OFFSET(scomp->data, scomp->stride * j),
          taps, 16, 5, 3, scomp->width);
    }
  }
}

void
schro_frame_upsample_vert (SchroFrame *dest, SchroFrame *src)
{
  int i, j, k;
  SchroFrameData *dcomp;
  SchroFrameData *scomp;

  if (SCHRO_FRAME_FORMAT_DEPTH(dest->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      SCHRO_FRAME_FORMAT_DEPTH(src->format) != SCHRO_FRAME_FORMAT_DEPTH_U8 ||
      src->format != dest->format) {
    SCHRO_ERROR("unimplemented");
    return;
  }

  for(k=0;k<3;k++){
    static const int16_t taps[8] = { -1, 3, -7, 21, 21, -7, 3, -1 };
    uint8_t *list[8];
    const int16_t offsetshift[2] = { 16, 5 };

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    for(j=0;j<dcomp->height;j++){
      for(i=0;i<8;i++){
        list[i] = SCHRO_FRAME_DATA_GET_LINE(scomp,
            CLAMP(i+j-3,0,scomp->height-1));
      }
      oil_mas8_across_u8 (SCHRO_FRAME_DATA_GET_LINE(dcomp, j), list,
          taps, offsetshift, scomp->width);
    }
  }
}

double
schro_frame_calculate_average_luma (SchroFrame *frame)
{
  SchroFrameData *comp;
  int j;
  int sum = 0;
  int n;

  comp = &frame->components[0];

  switch (SCHRO_FRAME_FORMAT_DEPTH(frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      for(j=0;j<comp->height;j++){
        int32_t linesum;
        oil_sum_s32_u8 (&linesum, OFFSET(comp->data, comp->stride * j),
            comp->width);
        sum += linesum;
      }
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      for(j=0;j<comp->height;j++){
        int32_t linesum;
        oil_sum_s32_s16 (&linesum, OFFSET(comp->data, comp->stride * j),
            comp->width);
        sum += linesum;
      }
      break;
    default:
      SCHRO_ERROR ("unimplemented");
      break;
  }

  n = comp->height * comp->width;
  return (double)sum / n;
}

static void
schro_frame_component_planar_copy_u8 (SchroFrameData *dest,
    SchroFrameData *src)
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
schro_frame_component_convert_420_to_444 (SchroFrameData *dest,
    SchroFrameData *src)
{
  int j;
  uint8_t *tmp;
  uint32_t weight = 128;

  SCHRO_ASSERT(dest->height <= src->height * 2);
  SCHRO_ASSERT(dest->width <= src->width * 2);

  tmp = schro_malloc (src->width);
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
  schro_free(tmp);
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

void
schro_frame_md5 (SchroFrame *frame, uint32_t *state)
{
  uint8_t *line;
  int x,y,k;

  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;
  
  x = 0;
  y = 0;
  k = 0;
  for(k=0;k<3;k++){
    for(y=0;y<frame->components[k].height;y++){
      line = OFFSET(frame->components[k].data,
          frame->components[k].stride * y);
      for(x=0;x+63<frame->components[k].width;x+=64){
        oil_md5 (state, (uint32_t *)(line + x));
      }
      if (x < frame->components[k].width) {
        uint8_t tmp[64];
        int left;
        left = frame->components[k].width - x;
        memcpy (tmp, line + x, left);
        memset (tmp + left, 0, 64 - left);
        oil_md5 (state, (uint32_t *)tmp);
      }
    }
  }

  SCHRO_DEBUG("md5 %02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
      state[0]&0xff, (state[0]>>8)&0xff, (state[0]>>16)&0xff,
      (state[0]>>24)&0xff,
      state[1]&0xff, (state[1]>>8)&0xff, (state[1]>>16)&0xff,
      (state[1]>>24)&0xff,
      state[2]&0xff, (state[2]>>8)&0xff, (state[2]>>16)&0xff,
      (state[2]>>24)&0xff,
      state[3]&0xff, (state[3]>>8)&0xff, (state[3]>>16)&0xff,
      (state[3]>>24)&0xff);
}

void
schro_frame_mark (SchroFrame *frame, int value)
{
  uint8_t *line;
  int y;
  int i;

  for(y=0;y<MIN(10,frame->components[0].height);y++){
    line = OFFSET(frame->components[0].data,
        frame->components[0].stride * y);
    for(i=0;i<10;i++){
      line[i] = value;
    }
  }

}

void
schro_frame_data_get_codeblock (SchroFrameData *dest, SchroFrameData *src,
    int x, int y, int horiz_codeblocks, int vert_codeblocks)
{
  int xmin = (src->width*x)/horiz_codeblocks;
  int xmax = (src->width*(x+1))/horiz_codeblocks;
  int ymin = (src->height*y)/vert_codeblocks;
  int ymax = (src->height*(y+1))/vert_codeblocks;

  dest->format = src->format;
  dest->data = OFFSET(src->data,
      src->stride * ymin + sizeof(int16_t) * xmin);
  dest->stride = src->stride;
  dest->width = xmax - xmin;
  dest->height = ymax - ymin;
  dest->length = 0;
  dest->h_shift = src->h_shift;
  dest->v_shift = src->v_shift;
}


/* upsampled frame */

SchroUpsampledFrame *
schro_upsampled_frame_new (SchroFrame *frame)
{
  SchroUpsampledFrame *df;

  df = schro_malloc0 (sizeof(SchroUpsampledFrame));

  df->frames[0] = frame;

  return df;
}

void
schro_upsampled_frame_free (SchroUpsampledFrame *df)
{
  int i;
  for(i=0;i<4;i++){
    if (df->frames[i]) {
      schro_frame_unref (df->frames[i]);
    }
  }
  schro_free(df);
}

void
schro_upsampled_frame_upsample (SchroUpsampledFrame *df)
{
  if (df->frames[1]) return;

  df->frames[1] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  df->frames[2] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  df->frames[3] = schro_frame_new_and_alloc (df->frames[0]->format,
      df->frames[0]->width, df->frames[0]->height);
  schro_frame_upsample_horiz (df->frames[1], df->frames[0]);
  schro_frame_upsample_vert (df->frames[2], df->frames[0]);
  schro_frame_upsample_horiz (df->frames[3], df->frames[2]);
}

int
schro_upsampled_frame_get_pixel_prec0 (SchroUpsampledFrame *upframe, int k,
    int x, int y)
{
  SchroFrameData *comp;
  uint8_t *line;

  comp = upframe->frames[0]->components + k;
  x = CLAMP(x, 0, comp->width - 1);
  y = CLAMP(y, 0, comp->height - 1);

  line = OFFSET(comp->data, y*comp->stride);

  return line[x];
}

int
schro_upsampled_frame_get_pixel_prec1 (SchroUpsampledFrame *upframe, int k,
    int x, int y)
{
  SchroFrameData *comp;
  uint8_t *line;
  int i;

  comp = upframe->frames[0]->components + k;
  x = CLAMP(x, 0, comp->width * 2 - 1);
  y = CLAMP(y, 0, comp->height * 2 - 1);

  i = ((y&1)<<1) | (x&1);
  x >>= 1;
  y >>= 1;

  comp = upframe->frames[i]->components + k;
  line = OFFSET(comp->data, y*comp->stride);

  return line[x];
}

int
schro_upsampled_frame_get_pixel_prec3 (SchroUpsampledFrame *upframe, int k,
    int x, int y)
{
  int hx, hy;
  int rx, ry;
  int w00, w01, w10, w11;
  int value;

  hx = x >> 2;
  hy = y >> 2;

  rx = x & 0x3;
  ry = y & 0x3;

  w00 = (4 - ry) * (4 - rx);
  w01 = (4 - ry) * rx;
  w10 = ry * (4 - rx);
  w11 = ry * rx;

  value = w00 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx, hy);
  value += w01 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx + 1, hy);
  value += w10 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx, hy + 1);
  value += w11 * schro_upsampled_frame_get_pixel_prec1 (upframe, k, hx + 1, hy + 1);

  return ROUND_SHIFT(value, 4);
}

int
schro_upsampled_frame_get_pixel_precN (SchroUpsampledFrame *upframe, int k,
    int x, int y, int prec)
{
  switch (prec) {
    case 0:
      return schro_upsampled_frame_get_pixel_prec0 (upframe, k, x, y);
    case 1:
      return schro_upsampled_frame_get_pixel_prec1 (upframe, k, x, y);
    case 2:
      return schro_upsampled_frame_get_pixel_prec3 (upframe, k, x<<1, y<<1);
    case 3:
      return schro_upsampled_frame_get_pixel_prec3 (upframe, k, x, y);
  }

  SCHRO_ASSERT(0);
}

