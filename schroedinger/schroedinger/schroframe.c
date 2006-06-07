

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schroframe.h>
#include <liboil/liboil.h>

#include <stdlib.h>
#include <string.h>

SchroFrame *
schro_frame_new (void)
{
  SchroFrame *frame;

  frame = malloc (sizeof(*frame));
  memset (frame, 0, sizeof(*frame));

  return frame;
}

SchroFrame *
schro_frame_new_and_alloc (SchroFrameFormat format, int width, int height, int sub_x, int sub_y)
{
  SchroFrame *frame = schro_frame_new();
  int bytes_pp;
  
  frame->format = format;

  switch (format) {
    case SCHRO_FRAME_FORMAT_U8:
      bytes_pp = 1;
      break;
    case SCHRO_FRAME_FORMAT_S16:
      bytes_pp = 2;
      break;
    default:
      bytes_pp = 0;
      break;
  }

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = width * bytes_pp;
  frame->components[0].length = 
    frame->components[0].stride * frame->components[0].height;

  frame->components[1].width = width / sub_x;
  frame->components[1].height = height / sub_y;
  frame->components[1].stride = frame->components[1].width * bytes_pp;
  frame->components[1].length = 
    frame->components[1].stride * frame->components[0].height;

  frame->components[2].width = width / sub_x;
  frame->components[2].height = height / sub_y;
  frame->components[2].stride = frame->components[2].width * bytes_pp;
  frame->components[2].length = 
    frame->components[2].stride * frame->components[0].height;

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
schro_frame_new_I420 (void *data, int width, int height)
{
  SchroFrame *frame = schro_frame_new();

  /* FIXME: This isn't 100% correct */

  frame->format = SCHRO_FRAME_FORMAT_U8;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = width;
  frame->components[0].data = data;
  frame->components[0].length =
    frame->components[0].height * frame->components[0].stride;

  frame->components[1].width = width / 2;
  frame->components[1].height = height / 2;
  frame->components[1].stride = frame->components[1].width;
  frame->components[1].length =
    frame->components[1].height * frame->components[1].stride;
  frame->components[1].data =
    frame->components[0].data + frame->components[0].length; 

  frame->components[2].width = width / 2;
  frame->components[2].height = height / 2;
  frame->components[2].stride = frame->components[2].width;
  frame->components[2].length =
    frame->components[2].height * frame->components[2].stride;
  frame->components[2].data =
    frame->components[1].data + frame->components[1].length; 

  return frame;
}

void
schro_frame_free (SchroFrame *frame)
{
  if (frame->free) {
    frame->free (frame, frame->priv);
  }
  if (frame->regions[0]) {
    free(frame->regions[0]);
  }

  free(frame);
}

void schro_frame_set_free_callback (SchroFrame *frame,
    SchroFrameFreeFunc free_func, void *priv)
{
  frame->free = free_func;
  frame->priv = priv;
}

static void schro_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src);
//static void schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src);

void
schro_frame_convert (SchroFrame *dest, SchroFrame *src)
{
  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  dest->frame_number = src->frame_number;

  if (dest->format == SCHRO_FRAME_FORMAT_U8 &&
      src->format == SCHRO_FRAME_FORMAT_S16) {
    schro_frame_convert_u8_s16 (dest, src);
    return;
  }
  if (dest->format == SCHRO_FRAME_FORMAT_S16 &&
      src->format == SCHRO_FRAME_FORMAT_U8) {
    schro_frame_convert_s16_u8 (dest, src);
    return;
  }
#if 0
  if (dest->format == SCHRO_FRAME_FORMAT_U8 &&
      src->format == SCHRO_FRAME_FORMAT_U8) {
    schro_frame_convert_u8_u8 (dest, src);
    return;
  }
#endif

  SCHRO_ERROR("unimplemented");
}

static void schro_frame_add_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_add_s16_u8 (SchroFrame *dest, SchroFrame *src);

void
schro_frame_add (SchroFrame *dest, SchroFrame *src)
{
  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  if (dest->format == SCHRO_FRAME_FORMAT_S16 &&
      src->format == SCHRO_FRAME_FORMAT_S16) {
    schro_frame_add_s16_s16 (dest, src);
    return;
  }
  if (dest->format == SCHRO_FRAME_FORMAT_S16 &&
      src->format == SCHRO_FRAME_FORMAT_U8) {
    schro_frame_add_s16_u8 (dest, src);
    return;
  }

  SCHRO_ERROR("unimplemented");
}

static void schro_frame_subtract_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_frame_subtract_s16_u8 (SchroFrame *dest, SchroFrame *src);

void
schro_frame_subtract (SchroFrame *dest, SchroFrame *src)
{
  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  if (dest->format == SCHRO_FRAME_FORMAT_S16 &&
      src->format == SCHRO_FRAME_FORMAT_S16) {
    schro_frame_subtract_s16_s16 (dest, src);
    return;
  }
  if (dest->format == SCHRO_FRAME_FORMAT_S16 &&
      src->format == SCHRO_FRAME_FORMAT_U8) {
    schro_frame_subtract_s16_u8 (dest, src);
    return;
  }

  SCHRO_ERROR("unimplemented");
}

#if 0
static void
oil_convert4_s16_u8 (int16_t *d_n, uint8_t *s_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = s_n[i];
  }
}
#endif

#define CLAMP(x,a,b) ((x)<(a) ? (a) : ((x)>(b) ? (b) : (x)))

#if 0
static void
oil_convert4_u8_s16 (uint8_t *d_n, int16_t *s_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = CLAMP(s_n[i],0,255);
  }
}
#endif

#define OFFSET(ptr,offset) ((void *)(((uint8_t *)(ptr)) + (offset)))
#define MIN(a,b) ((a)<(b) ? (a) : (b))

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

#if 0
    if (dcomp->width <= scomp->width && dcomp->height <= scomp->height) {
      for(y=0;y<dcomp->height;y++){
        oil_convert4_u8_s16 (ddata, sdata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
    } else {
      void *last_ddata;

      if (dcomp->width < scomp->width || dcomp->height < scomp->height) {
        SCHRO_ERROR("unimplemented");
      }

      for(y=0;y<scomp->height;y++){
        oil_convert4_u8_s16 (ddata, sdata, scomp->width);
        oil_splat_u8_ns (ddata + scomp->width,
            ddata + scomp->width - 1,
            dcomp->width - scomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
      last_ddata = OFFSET(ddata, -dcomp->stride);
      for(;y<dcomp->height;y++){
        oil_memcpy (ddata, last_ddata, dcomp->width * sizeof (int16_t));
        ddata = OFFSET(ddata, dcomp->stride);
      }
    }
#endif
  }

  schro_frame_edge_extend (dest, src->components[0].width,
      src->components[0].height);
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
#if 0
    if (dcomp->width <= scomp->width && dcomp->height <= scomp->height) {
      for(y=0;y<dcomp->height;y++){
        oil_convert4_s16_u8 (ddata, sdata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
    } else {
      void *last_ddata;

      if (dcomp->width < scomp->width || dcomp->height < scomp->height) {
        SCHRO_ERROR("unimplemented");
      }

      for(y=0;y<scomp->height;y++){
        oil_convert4_s16_u8 (ddata, sdata, scomp->width);
        oil_splat_u16_ns ((uint16_t *)ddata + scomp->width,
            (uint16_t *)ddata + scomp->width - 1,
            dcomp->width - scomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
      last_ddata = OFFSET(ddata, -dcomp->stride);
      for(;y<dcomp->height;y++){
        oil_memcpy (ddata, last_ddata, dcomp->width * sizeof (int16_t));
        ddata = OFFSET(ddata, dcomp->stride);
      }
    }
#endif
  }
}


#if 0
static void
schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  uint8_t *ddata;
  uint8_t *sdata;
  int i;
  int y;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    if (dcomp->width <= scomp->width && dcomp->height <= scomp->height) {
      for(y=0;y<dcomp->height;y++){
        oil_memcpy (ddata, sdata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
    } else {
      void *last_ddata;

      if (dcomp->width < scomp->width || dcomp->height < scomp->height) {
        SCHRO_ERROR("unimplemented");
      }

      for(y=0;y<scomp->height;y++){
        oil_memcpy (ddata, sdata, scomp->width);
        oil_splat_u8_ns (ddata + scomp->width, ddata + scomp->width - 1,
            dcomp->width - scomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
      last_ddata = OFFSET(ddata, -dcomp->stride);
      for(;y<dcomp->height;y++){
        oil_memcpy (ddata, last_ddata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
      }
    }
  }
}
#endif

static void
oil_add_s16(int16_t *d_n, int16_t *s1_n, int16_t *s2_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = s1_n[i] + s2_n[i];
  }
}

static void
oil_add_s16_u8(int16_t *d_n, int16_t *s1_n, uint8_t *s2_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = s1_n[i] + s2_n[i];
  }
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
oil_subtract_s16(int16_t *d_n, int16_t *s1_n, int16_t *s2_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = s1_n[i] - s2_n[i];
  }
}

static void
oil_subtract_s16_u8(int16_t *d_n, int16_t *s1_n, uint8_t *s2_n, int n)
{
  int i;

  for(i=0;i<n;i++){
    d_n[i] = s1_n[i] - (s2_n[i]<<4);
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

  SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16);

  for(component=0;component<3;component++){
    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)frame->components[component].data;
    for(level=0;level<params->transform_depth;level++) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = width << level;

      schro_wavelet_transform_2d (params->wavelet_filter_index,
          frame_data, stride*2, w, h, tmp);
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

  SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16);

  for(component=0;component<3;component++){
    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)frame->components[component].data;
    for(level=params->transform_depth-1; level >=0;level--) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = width << level;

      schro_wavelet_inverse_transform_2d (params->wavelet_filter_index,
          frame_data, stride*2, w, h, tmp);
    }
  }
}


void
oil_leftshift_s16(int16_t *data, int *shift, int n)
{
  int i;
  for(i=0;i<n;i++){
    data[i] <<= *shift;
  }
}

void schro_frame_shift_left (SchroFrame *frame, int shift)
{
  SchroFrameComponent *comp;
  int16_t *data;
  int i;
  int y;

  for(i=0;i<3;i++){
    comp = &frame->components[i];
    data = comp->data;

    for(y=0;y<comp->height;y++){
      oil_leftshift_s16 (data, &shift, comp->width);
      data = OFFSET(data, comp->stride);
    }
  }
}

void
oil_divpow2_s16(int16_t *data, int *shift, int n)
{
  int i;
  int16_t round;
 
  if (*shift > 1) {
    round = 1<<(*shift-1);
  } else {
    round = 0;
  }

  for(i=0;i<n;i++){
    data[i] = (data[i] + round) >> *shift;
  }
}

void schro_frame_shift_right (SchroFrame *frame, int shift)
{
  SchroFrameComponent *comp;
  int16_t *data;
  int i;
  int y;

  for(i=0;i<3;i++){
    comp = &frame->components[i];
    data = comp->data;

    for(y=0;y<comp->height;y++){
      oil_divpow2_s16 (data, &shift, comp->width);
      data = OFFSET(data, comp->stride);
    }
  }
}


static void
oil_splat_s16_ns (int16_t *dest, int16_t *src, int n)
{
  oil_splat_u16_ns ((uint16_t *)dest, (uint16_t *)src, n);
}

void
schro_frame_edge_extend (SchroFrame *frame, int width, int height)
{
  SchroFrameComponent *comp;
  int i;
  int y;

  switch(frame->format) {
    case SCHRO_FRAME_FORMAT_U8:
      for(i=0;i<3;i++){
        uint8_t *data;
        int w,h;

        comp = &frame->components[i];
        data = comp->data;

        if (i>0) {
          w = width/2;
          h = height/2;
        } else {
          w = width;
          h = height;
        }
        if (w < comp->width) {
          for(y = 0; y<h; y++) {
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
    case SCHRO_FRAME_FORMAT_S16:
      for(i=0;i<3;i++){
        int16_t *data;
        int w,h;

        comp = &frame->components[i];
        data = comp->data;

        if (i>0) {
          w = width/2;
          h = height/2;
        } else {
          w = width;
          h = height;
        }
        if (w < comp->width) {
          for(y = 0; y<h; y++) {
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
      break;
  }
}

