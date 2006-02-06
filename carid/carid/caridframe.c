

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <carid/carid.h>
#include <carid/caridframe.h>
#include <liboil/liboil.h>

#include <stdlib.h>
#include <string.h>

CaridFrame *
carid_frame_new (void)
{
  CaridFrame *frame;

  frame = malloc (sizeof(*frame));
  memset (frame, 0, sizeof(*frame));

  return frame;
}

CaridFrame *
carid_frame_new_and_alloc (CaridFrameFormat format, int width, int height, int sub_x, int sub_y)
{
  CaridFrame *frame = carid_frame_new();
  int bytes_pp;
  
  frame->format = format;

  switch (format) {
    case CARID_FRAME_FORMAT_U8:
      bytes_pp = 1;
      break;
    case CARID_FRAME_FORMAT_S16:
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

CaridFrame *
carid_frame_new_I420 (void *data, int width, int height)
{
  CaridFrame *frame = carid_frame_new();

  /* FIXME: This isn't 100% correct */

  frame->format = CARID_FRAME_FORMAT_U8;

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
carid_frame_free (CaridFrame *frame)
{
  if (frame->regions[0]) {
    free(frame->regions[0]);
  }

  free(frame);
}

static void carid_frame_convert_u8_s16 (CaridFrame *dest, CaridFrame *src);
static void carid_frame_convert_s16_u8 (CaridFrame *dest, CaridFrame *src);

void
carid_frame_convert (CaridFrame *dest, CaridFrame *src)
{
  CARID_ASSERT(dest != NULL);
  CARID_ASSERT(src != NULL);

  if (dest->format == CARID_FRAME_FORMAT_U8 &&
      src->format == CARID_FRAME_FORMAT_S16) {
    carid_frame_convert_u8_s16 (dest, src);
    return;
  }
  if (dest->format == CARID_FRAME_FORMAT_S16 &&
      src->format == CARID_FRAME_FORMAT_U8) {
    carid_frame_convert_s16_u8 (dest, src);
    return;
  }

  CARID_ERROR("unimplemented");
}

#define OFFSET(ptr,offset) ((void *)(((uint8_t *)(ptr)) + (offset)))

static void
carid_frame_convert_u8_s16 (CaridFrame *dest, CaridFrame *src)
{
  CaridFrameComponent *dcomp;
  CaridFrameComponent *scomp;
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
        CARID_ERROR("unimplemented");
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
carid_frame_convert_s16_u8 (CaridFrame *dest, CaridFrame *src)
{
  CaridFrameComponent *dcomp;
  CaridFrameComponent *scomp;
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
        CARID_ERROR("unimplemented");
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


