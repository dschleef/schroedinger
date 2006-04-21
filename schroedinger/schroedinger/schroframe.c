

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schro/schro.h>
#include <schro/schroframe.h>
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

#define OFFSET(ptr,offset) ((void *)(((uint8_t *)(ptr)) + (offset)))

static void
schro_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src)
{
  SchroFrameComponent *dcomp;
  SchroFrameComponent *scomp;
  uint8_t *ddata;
  int16_t *sdata;
  int i;
  int y;

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    if (dcomp->width <= scomp->width && dcomp->height <= scomp->height) {
      for(y=0;y<dcomp->height;y++){
        oil_convert_u8_s16 (ddata, sdata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
    } else {
      void *last_ddata;

      if (dcomp->width < scomp->width || dcomp->height < scomp->height) {
        SCHRO_ERROR("unimplemented");
      }

      for(y=0;y<scomp->height;y++){
        oil_convert_u8_s16 (ddata, sdata, scomp->width);
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
  }
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

  for(i=0;i<3;i++){
    dcomp = &dest->components[i];
    scomp = &src->components[i];
    ddata = dcomp->data;
    sdata = scomp->data;

    if (dcomp->width <= scomp->width && dcomp->height <= scomp->height) {
      for(y=0;y<dcomp->height;y++){
        oil_convert_s16_u8 (ddata, sdata, dcomp->width);
        ddata = OFFSET(ddata, dcomp->stride);
        sdata = OFFSET(sdata, scomp->stride);
      }
    } else {
      void *last_ddata;

      if (dcomp->width < scomp->width || dcomp->height < scomp->height) {
        SCHRO_ERROR("unimplemented");
      }

      for(y=0;y<scomp->height;y++){
        oil_convert_s16_u8 (ddata, sdata, scomp->width);
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
  }
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



