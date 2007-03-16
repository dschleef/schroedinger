

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schrointernal.h>
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
schro_frame_new_and_alloc2 (SchroFrameFormat format, int width, int height,
    int width2, int height2)
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

  frame->components[1].width = width2;
  frame->components[1].height = height2;
  frame->components[1].stride = frame->components[1].width * bytes_pp;
  frame->components[1].length = 
    frame->components[1].stride * frame->components[0].height;

  frame->components[2].width = width2;
  frame->components[2].height = height2;
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

  frame->format = SCHRO_FRAME_FORMAT_U8;

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_POW2(width,2);
  frame->components[0].data = data;
  frame->components[0].length = frame->components[0].stride *
    ROUND_UP_POW2(frame->components[0].height,1);

  frame->components[1].width = ROUND_UP_SHIFT(width,1);
  frame->components[1].height = ROUND_UP_SHIFT(height,1);
  frame->components[1].stride = ROUND_UP_POW2(frame->components[1].width,2);
  frame->components[1].length =
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].data =
    frame->components[0].data + frame->components[0].length; 

  frame->components[2].width = ROUND_UP_SHIFT(width,1);
  frame->components[2].height = ROUND_UP_SHIFT(height,1);
  frame->components[2].stride = ROUND_UP_POW2(frame->components[2].width,2);
  frame->components[2].length =
    frame->components[2].stride * frame->components[2].height;
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
static void schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src);

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
  if (dest->format == SCHRO_FRAME_FORMAT_U8 &&
      src->format == SCHRO_FRAME_FORMAT_U8) {
    schro_frame_convert_u8_u8 (dest, src);
    return;
  }

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
  }

  schro_frame_edge_extend (dest, src->components[0].width,
      src->components[0].height);
}

static void
schro_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src)
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
      memcpy (ddata, sdata, width);
      ddata = OFFSET(ddata, dcomp->stride);
      sdata = OFFSET(sdata, scomp->stride);
    }
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
  }

  schro_frame_edge_extend (dest, src->components[0].width,
      src->components[0].height);
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

  SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16);

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

  SCHRO_ASSERT(frame->format == SCHRO_FRAME_FORMAT_S16);

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

  SCHRO_DEBUG("extending %d %d -> %d %d", width, height,
      frame->components[0].width, frame->components[0].height);

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
      break;
  }
}

void
schro_frame_zero_extend (SchroFrame *frame, int width, int height)
{
  SchroFrameComponent *comp;
  int i;
  int y;

  SCHRO_DEBUG("extending %d %d -> %d %d", width, height,
      frame->components[0].width, frame->components[0].height);

  switch(frame->format) {
    case SCHRO_FRAME_FORMAT_U8:
      for(i=0;i<3;i++){
        uint8_t zero = 0;
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
            oil_splat_u8_ns (data + w, &zero, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_splat_u8_ns (OFFSET(comp->data, comp->stride * y), &zero,
              comp->width);
        }
      }
      break;
    case SCHRO_FRAME_FORMAT_S16:
      for(i=0;i<3;i++){
        int16_t *data;
        int w,h;
        int16_t zero = 0;

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
            oil_splat_s16_ns (data + w, &zero, comp->width - w);
          }
        }
        for(y=h; y < comp->height; y++) {
          oil_splat_s16_ns (OFFSET(comp->data, comp->stride * y), &zero,
              comp->width * 2);
        }
      }
      break;
    default:
      SCHRO_ERROR("unimplemented case");
      break;
  }
}

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

  if (dest->format != SCHRO_FRAME_FORMAT_U8 ||
      src->format != SCHRO_FRAME_FORMAT_U8) {
    SCHRO_ERROR("unimplemented");
    return;
  }

  for(k=0;k<3;k++){
    uint8_t *sdata;
    uint8_t *ddata;

    dcomp = &dest->components[k];
    scomp = &src->components[k];

    SCHRO_ASSERT(ROUND_UP_SHIFT(scomp->width,1) == dcomp->width);
    SCHRO_ASSERT(ROUND_UP_SHIFT(scomp->height,1) == dcomp->height);

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

void
schro_frame_h_upsample (SchroFrame *dest, SchroFrame *src)
{
  int i, j, k, l;
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;

  if (dest->format != SCHRO_FRAME_FORMAT_U8 ||
      src->format != SCHRO_FRAME_FORMAT_U8) {
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
      for(;i<dcomp->width-1;i++){
        x = 128;
        for(l=0;l<10;l++){
          x += taps[l] * sdata[scomp->stride * j +
            CLAMP(i - 4 + l,0,scomp->width-1)];
        }
        x >>= 8;
        ddata[dcomp->stride * j + i] = CLAMP(x,0,255);
      }
      i = dcomp->width - 1;
      ddata[dcomp->stride * j + i] = sdata[scomp->stride * j + i];
    }
  }
}

void
schro_frame_v_upsample (SchroFrame *dest, SchroFrame *src)
{
  int i, j, k, l;
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;

  if (dest->format != SCHRO_FRAME_FORMAT_U8 ||
      src->format != SCHRO_FRAME_FORMAT_U8) {
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
    for(j=dcomp->height-5;j<dcomp->height - 1;j++){
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
    j = dcomp->height - 1;
    for(i=0;i<dcomp->width;i++){
      ddata[dcomp->stride * j + i] = sdata[scomp->stride * j + i];
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

  if (frame->format == SCHRO_FRAME_FORMAT_U8) {
    uint8_t *data;
    data = comp->data;
    for(j=0;j<comp->height;j++){
      for(i=0;i<comp->width;i++){
        sum += data[comp->stride * j + i];
      }
    }
  } else if (frame->format == SCHRO_FRAME_FORMAT_S16) {
    int16_t *data;
    for(j=0;j<comp->height;j++){
      data = OFFSET(comp->data, comp->stride * j);
      for(i=0;i<comp->width;i++){
        sum += data[i];
      }
    }
  } else {
    SCHRO_ERROR ("unimplemented");
  }

  n = comp->height * comp->width;
  return (sum + n/2) / n;
}

