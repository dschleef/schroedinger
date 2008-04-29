
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SCHRO_ENABLE_UNSTABLE_API 1

#include "schrovirtframe.h"
#include <schroedinger/schro.h>
#include <schroedinger/schroutils.h>
#include <string.h>


SchroFrame *
schro_frame_new_virtual (SchroMemoryDomain *domain, SchroFrameFormat format,
    int width, int height)
{
  SchroFrame *frame = schro_frame_new();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  int i;

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->domain = domain;

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

  frame->components[0].format = format;
  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_4(width * bytes_pp);
  frame->components[0].length =
    frame->components[0].stride * frame->components[0].height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].format = format;
  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride = ROUND_UP_4(chroma_width * bytes_pp);
  frame->components[1].length =
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].format = format;
  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride = ROUND_UP_4(chroma_width * bytes_pp);
  frame->components[2].length =
    frame->components[2].stride * frame->components[2].height;
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  for(i=0;i<3;i++){
    SchroFrameData *comp = &frame->components[i];
    int j;

    frame->regions[i] = malloc (comp->stride * SCHRO_FRAME_CACHE_SIZE);
    for(j=0;j<SCHRO_FRAME_CACHE_SIZE;j++){
      frame->cached_lines[i][j] = -1;
    }
  }
  frame->is_virtual = TRUE;

  return frame;
}

void *
schro_virt_frame_get_line (SchroFrame *frame, int component, int i)
{
  SchroFrameData *comp = &frame->components[component];
  int j;
  int min;
  int min_j;

  SCHRO_ASSERT(i >= 0);
  //SCHRO_ASSERT(i < comp->height);

  if (!frame->is_virtual) {
    return SCHRO_FRAME_DATA_GET_LINE(&frame->components[component], i);
  }

  for(j=0;j<SCHRO_FRAME_CACHE_SIZE;j++){
    if (frame->cached_lines[component][j] == i) {
      return SCHRO_OFFSET(frame->regions[component], comp->stride * j);
    }
  }

  min_j = 0;
  min = frame->cached_lines[component][0];
  for(j=1;j<SCHRO_FRAME_CACHE_SIZE;j++){
    if (frame->cached_lines[component][j] < min) {
      min = frame->cached_lines[component][j];
      min_j = j;
    }
  }

  schro_virt_frame_render_line (frame,
      SCHRO_OFFSET(frame->regions[component], comp->stride * min_j), component, i);

  return SCHRO_OFFSET(frame->regions[component], comp->stride * min_j);
}

void
schro_virt_frame_render_line (SchroFrame *frame, void *dest,
    int component, int i)
{
  frame->render_line (frame, dest, component, i);
}

void
schro_virt_frame_render (SchroFrame *frame, SchroFrame *dest)
{
  int i,k;

  SCHRO_ASSERT(frame->width == dest->width);
  SCHRO_ASSERT(frame->height == dest->height);

  for(k=0;k<3;k++){
    SchroFrameData *comp = dest->components + k;

    for(i=0;i<frame->components[k].height;i++){
      schro_virt_frame_render_line (frame,
          SCHRO_FRAME_DATA_GET_LINE (comp, i), k, i);
    }
  }
}




void
schro_virt_frame_render_downsample_horiz (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  for(j=0;j<frame->components[component].width;j++){
    dest[j] = src[j*2];
  }
}

SchroFrame *
schro_virt_frame_new_horiz_downsample (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width/2, vf->height);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = schro_virt_frame_render_downsample_horiz;

  return virt_frame;
}

void
schro_virt_frame_render_downsample_vert (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i*2);

  memcpy (dest, src, frame->components[component].width);
}

SchroFrame *
schro_virt_frame_new_vert_downsample (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width, vf->height/2);
  virt_frame->virt_frame1 = vf;
  virt_frame->render_line = schro_virt_frame_render_downsample_vert;

  return virt_frame;
}

