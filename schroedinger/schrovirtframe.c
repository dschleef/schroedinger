
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define SCHRO_ENABLE_UNSTABLE_API 1

#include "schrovirtframe.h"
#include <schroedinger/schro.h>
#include <schroedinger/schroutils.h>
#include <liboil/liboil.h>
#include <string.h>
#include <math.h>


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

  if (SCHRO_FRAME_IS_PACKED (format)) {
    frame->components[0].format = format;
    frame->components[0].width = width;
    frame->components[0].height = height;
    if (format == SCHRO_FRAME_FORMAT_AYUV) {
      frame->components[0].stride = width * 4;
    } else {
      frame->components[0].stride = ROUND_UP_POW2(width,1) * 2;
    }
    frame->components[0].length = frame->components[0].stride * height;

    frame->components[0].data = frame->regions[0];
    frame->components[0].v_shift = 0;
    frame->components[0].h_shift = 0;

    frame->regions[0] = malloc (frame->components[0].stride * SCHRO_FRAME_CACHE_SIZE);
    for(i=0;i<SCHRO_FRAME_CACHE_SIZE;i++){
      frame->cached_lines[0][i] = -1;
    }
    frame->is_virtual = TRUE;

    return frame;
  }

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
  frame->cached_lines[component][min_j] = i;

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
schro_virt_frame_render_downsample_horiz_cosite (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  n_src = frame->virt_frame1->components[component].width;

  for(j=0;j<frame->components[component].width;j++){
    int x = 0;
    x +=  1*src[CLAMP(j*2 - 1, 0, n_src-1)];
    x +=  2*src[CLAMP(j*2 + 0, 0, n_src-1)];
    x +=  1*src[CLAMP(j*2 + 1, 0, n_src-1)];
    dest[j] = CLAMP((x+2)>>2, 0, 255);
  }
}

void
schro_virt_frame_render_downsample_horiz_halfsite (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;
  int taps = 4;
  int k;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  n_src = frame->virt_frame1->components[component].width;

  switch (taps) {
    case 4:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        x +=  6*src[CLAMP(j*2 - 1, 0, n_src-1)];
        x += 26*src[CLAMP(j*2 + 0, 0, n_src-1)];
        x += 26*src[CLAMP(j*2 + 1, 0, n_src-1)];
        x +=  6*src[CLAMP(j*2 + 2, 0, n_src-1)];
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    case 6:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        x += -3*src[CLAMP(j*2 - 2, 0, n_src-1)];
        x +=  8*src[CLAMP(j*2 - 1, 0, n_src-1)];
        x += 27*src[CLAMP(j*2 + 0, 0, n_src-1)];
        x += 27*src[CLAMP(j*2 + 1, 0, n_src-1)];
        x +=  8*src[CLAMP(j*2 + 2, 0, n_src-1)];
        x += -3*src[CLAMP(j*2 + 3, 0, n_src-1)];
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
    case 8:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        const int taps8[8] = { -2, -4, 9, 29, 29, 9, -4, -2 };
        for(k=0;k<8;k++){
          x += taps8[k]*src[CLAMP(j*2 - 3 + k, 0, n_src-1)];
        }
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    case 10:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        const int taps10[10] = { 1, -2, -5, 9, 29, 29, 9, -5, -2, 1 };
        for(k=0;k<10;k++){
          x += taps10[k]*src[CLAMP(j*2 - 4 + k, 0, n_src-1)];
        }
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    default:
      break;
  }
}

SchroFrame *
schro_virt_frame_new_horiz_downsample (SchroFrame *vf, int cosite)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width/2, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  if (cosite) {
    virt_frame->render_line = schro_virt_frame_render_downsample_horiz_cosite;
  } else {
    virt_frame->render_line = schro_virt_frame_render_downsample_horiz_halfsite;
  }

  return virt_frame;
}

void
schro_virt_frame_render_downsample_vert_cosite (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  int j;
  int n_src;

  n_src = frame->virt_frame1->components[component].height;
  src1 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(i*2 - 1, 0, n_src - 1));
  src2 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(i*2 + 0, 0, n_src - 1));
  src3 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(i*2 + 1, 0, n_src - 1));

  for(j=0;j<frame->components[component].width;j++){
    int x = 0;
    x +=  1*src1[j];
    x +=  2*src2[j];
    x +=  1*src3[j];
    dest[j] = CLAMP((x+2)>>2, 0, 255);
  }
}


void
schro_virt_frame_render_downsample_vert_halfsite (SchroFrame *frame,
    void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src[10];
  int j;
  int n_src;
  int taps = 4;
  int k;

  n_src = frame->virt_frame1->components[component].height;
  for(j=0;j<taps;j++){
    src[j] = schro_virt_frame_get_line (frame->virt_frame1, component,
        CLAMP(i*2 - (taps-2)/2 + j, 0, n_src - 1));
  }

  switch (taps) {
    case 4:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        x +=  6*src[0][j];
        x += 26*src[1][j];
        x += 26*src[2][j];
        x +=  6*src[3][j];
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    case 6:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        x += -3*src[0][j];
        x +=  8*src[1][j];
        x += 27*src[2][j];
        x += 27*src[3][j];
        x +=  8*src[4][j];
        x += -3*src[5][j];
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    case 8:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        const int taps8[8] = { -2, -4, 9, 29, 29, 9, -4, -2 };
        for(k=0;k<8;k++){
          x += taps8[k] * src[k][j];
        }
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    case 10:
      for(j=0;j<frame->components[component].width;j++){
        int x = 0;
        const int taps10[10] = { 1, -2, -5, 9, 29, 29, 9, -5, -2, 1 };
        //const int taps10[10] = { -1, 1, 6, 11, 15, 15, 11, 6, 1, -1 };
        for(k=0;k<10;k++){
          x += taps10[k] * src[k][j];
        }
        dest[j] = CLAMP((x+32)>>6, 0, 255);
      }
      break;
    default:
      SCHRO_ASSERT(0);
      break;
  }
}

SchroFrame *
schro_virt_frame_new_vert_downsample (SchroFrame *vf, int cosite)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width, vf->height/2);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  if (cosite) {
    virt_frame->render_line = schro_virt_frame_render_downsample_vert_cosite;
  } else {
    virt_frame->render_line = schro_virt_frame_render_downsample_vert_halfsite;
  }

  return virt_frame;
}

void
get_taps (double *taps, double x)
{
  taps[3] = x * x * (x - 1);
  taps[2] = x * (- x * x + x + 1);
  x = 1 - x;
  taps[1] = x * (- x * x + x + 1);
  taps[0] = x * x * (x - 1);
}

void
schro_virt_frame_render_resample_vert (SchroFrame *frame, void *_dest,
    int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  uint8_t *src4;
  int j;
  int n_src;
  double taps[4];
  double *scale = (double *)frame->virt_priv;
  double x;
  int src_i;

  x = (*scale) * i;
  src_i = floor (x);
  get_taps (taps, x - floor(x));

  n_src = frame->virt_frame1->components[component].height;
  src1 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(src_i - 1, 0, n_src - 1));
  src2 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(src_i + 0, 0, n_src - 1));
  src3 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(src_i + 1, 0, n_src - 1));
  src4 = schro_virt_frame_get_line (frame->virt_frame1, component,
      CLAMP(src_i + 2, 0, n_src - 1));

  for(j=0;j<frame->components[component].width;j++){
    double x = 0;
    x += taps[0]*src1[j];
    x += taps[1]*src2[j];
    x += taps[2]*src3[j];
    x += taps[3]*src4[j];
    dest[j] = CLAMP(rint(x), 0, 255);
  }
}

SchroFrame *
schro_virt_frame_new_vert_resample (SchroFrame *vf, int height)
{
  SchroFrame *virt_frame;
  double *scale;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, vf->width, height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = schro_virt_frame_render_resample_vert;

  scale = malloc(sizeof(double));
  virt_frame->virt_priv = scale;

  *scale = (double)vf->height / height;

  return virt_frame;
}

void
schro_virt_frame_render_resample_horiz (SchroFrame *frame, void *_dest,
    int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;
  int n_src;
  double taps[4];
  double *scale = (double *)frame->virt_priv;
  int src_i;

  n_src = frame->virt_frame1->components[component].width;
  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  for(j=0;j<frame->components[component].width;j++){
    double x;
    double y = 0;

    x = (*scale) * j;
    src_i = floor (x);
    get_taps (taps, x - floor(x));

    y = 0;
    y += taps[0]*src[CLAMP(src_i - 1, 0, n_src - 1)];
    y += taps[1]*src[CLAMP(src_i + 0, 0, n_src - 1)];
    y += taps[2]*src[CLAMP(src_i + 1, 0, n_src - 1)];
    y += taps[3]*src[CLAMP(src_i + 2, 0, n_src - 1)];
    dest[j] = CLAMP(rint(y), 0, 255);
  }
}

SchroFrame *
schro_virt_frame_new_horiz_resample (SchroFrame *vf, int width)
{
  SchroFrame *virt_frame;
  double *scale;

  virt_frame = schro_frame_new_virtual (NULL, vf->format, width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = schro_virt_frame_render_resample_horiz;

  scale = malloc(sizeof(double));
  virt_frame->virt_priv = scale;

  *scale = (double)vf->width / width;

  return virt_frame;
}

static void
unpack_yuyv (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      for(j=0;j<frame->width;j++){
        dest[j] = src[j*2];
      }
      break;
    case 1:
      for(j=0;j<frame->width/2;j++){
        dest[j] = src[j*4 + 1];
      }
      break;
    case 2:
      for(j=0;j<frame->width/2;j++){
        dest[j] = src[j*4 + 3];
      }
  }
}

static void
unpack_uyvy (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      for(j=0;j<frame->width;j++){
        dest[j] = src[j*2 + 1];
      }
      break;
    case 1:
      for(j=0;j<frame->width/2;j++){
        dest[j] = src[j*4 + 0];
      }
      break;
    case 2:
      for(j=0;j<frame->width/2;j++){
        dest[j] = src[j*4 + 2];
      }
  }
}

static void
unpack_ayuv (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, 0, i);

  switch (component) {
    case 0:
      for(j=0;j<frame->width;j++){
        dest[j] = src[j*4 + 1];
      }
      break;
    case 1:
      for(j=0;j<frame->width;j++){
        dest[j] = src[j*4 + 2];
      }
      break;
    case 2:
      for(j=0;j<frame->width;j++){
        dest[j] = src[j*4 + 3];
      }
  }
}

SchroFrame *
schro_virt_frame_new_unpack (SchroFrame *vf)
{
  SchroFrame *virt_frame;
  SchroFrameFormat format;
  SchroFrameRenderFunc render_line;

  switch (vf->format) {
    case SCHRO_FRAME_FORMAT_YUYV:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_yuyv;
      break;
    case SCHRO_FRAME_FORMAT_UYVY:
      format = SCHRO_FRAME_FORMAT_U8_422;
      render_line = unpack_uyvy;
      break;
    case SCHRO_FRAME_FORMAT_AYUV:
      format = SCHRO_FRAME_FORMAT_U8_444;
      render_line = unpack_ayuv;
      break;
    default:
      return schro_frame_ref (vf);
  }

  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = render_line;

  return virt_frame;
}


static void
pack_yuyv (SchroFrame *frame, void *_dest, int component, int i)
{
  uint32_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  oil_packyuyv (dest, src_y, src_u, src_v, frame->width/2);
}


SchroFrame *
schro_virt_frame_new_pack_YUY2 (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_YUYV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref (vf);
  virt_frame->render_line = pack_yuyv;

  return virt_frame;
}

static void
pack_uyvy (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for(j=0;j<frame->width/2;j++){
    dest[j*4+1] = src_y[j*2+0];
    dest[j*4+3] = src_y[j*2+1];
    dest[j*4+0] = src_u[j];
    dest[j*4+2] = src_v[j];
  }
}

SchroFrame *
schro_virt_frame_new_pack_UYVY (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_YUYV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = pack_uyvy;

  return virt_frame;
}

static void
pack_ayuv (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src_y;
  uint8_t *src_u;
  uint8_t *src_v;
  int j;

  src_y = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src_u = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src_v = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  for(j=0;j<frame->width;j++){
    dest[j*4+0] = 0xff;
    dest[j*4+1] = src_y[j];
    dest[j*4+2] = src_u[j];
    dest[j*4+3] = src_v[j];
  }
}

SchroFrame *
schro_virt_frame_new_pack_AYUV (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_YUYV,
      vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = pack_ayuv;

  return virt_frame;
}

static void
color_matrix (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src1;
  uint8_t *src2;
  uint8_t *src3;
  double m1, m2, m3;
  double offset;
  int j;

  src1 = schro_virt_frame_get_line (frame->virt_frame1, 0, i);
  src2 = schro_virt_frame_get_line (frame->virt_frame1, 1, i);
  src3 = schro_virt_frame_get_line (frame->virt_frame1, 2, i);

  switch (component) {
    case 0:
      m1 = 0.25679;
      m2 = 0.50413;
      m3 = 0.097906;
      offset = 16;
      break;
    case 1:
      m1 = -0.14822;
      m2 = -0.29099;
      m3 = 0.43922;
      offset = 128;
      break;
    case 2:
      m1 = 0.43922;
      m2 = -0.36779;
      m3 = -0.071427;
      offset = 128;
      break;
    default:
      m1 = 0.0;
      m2 = 0.0;
      m3 = 0.0;
      offset = 0;
      break;
  }

  for(j=0;j<frame->width;j++){
    dest[j] = floor (src1[j]*m1 + src2[j]*m2 + src3[j]*m3 + offset + 0.5);
  }

}

SchroFrame *
schro_virt_frame_new_color_matrix (SchroFrame *vf)
{
  SchroFrame *virt_frame;

  virt_frame = schro_frame_new_virtual (NULL, SCHRO_FRAME_FORMAT_U8_444,
      vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = color_matrix;

  return virt_frame;
}

static void
convert_444_422 (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  src = schro_virt_frame_get_line (frame->virt_frame1, component, i);

  if (component == 0) {
    memcpy (dest, src, frame->width);
  } else {
    for(j=0;j<frame->components[component].width;j++){
      dest[j] = src[j*2];
    }
  }
}

static void
convert_444_420 (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;
  int j;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
    memcpy (dest, src, frame->components[component].width);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i*2);
    for(j=0;j<frame->components[component].width;j++){
      dest[j] = src[j*2];
    }
  }
}

static void
convert_422_420 (SchroFrame *frame, void *_dest, int component, int i)
{
  uint8_t *dest = _dest;
  uint8_t *src;

  if (component == 0) {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i);
  } else {
    src = schro_virt_frame_get_line (frame->virt_frame1, component, i*2);
  }
  memcpy (dest, src, frame->components[component].width);
}


SchroFrame *
schro_virt_frame_new_subsample (SchroFrame *vf, SchroFrameFormat format)
{
  SchroFrame *virt_frame;
  SchroFrameRenderFunc render_line;

  if (vf->format == SCHRO_FRAME_FORMAT_U8_422 &&
      format == SCHRO_FRAME_FORMAT_U8_420) {
    render_line = convert_422_420;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_444 &&
      format == SCHRO_FRAME_FORMAT_U8_420) {
    render_line = convert_444_420;
  } else if (vf->format == SCHRO_FRAME_FORMAT_U8_444 &&
      format == SCHRO_FRAME_FORMAT_U8_422) {
    render_line = convert_444_422;
  } else {
    return NULL;
  }
  virt_frame = schro_frame_new_virtual (NULL, format, vf->width, vf->height);
  virt_frame->virt_frame1 = schro_frame_ref(vf);
  virt_frame->render_line = render_line;

  return virt_frame;
}




SchroFrame *
schro_virt_frame_new_horiz_downsample_take (SchroFrame *vf, int cosite)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_horiz_downsample (vf, cosite);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_vert_downsample_take (SchroFrame *vf, int cosite)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_vert_downsample (vf, cosite);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_vert_resample_take (SchroFrame *vf, int height)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_vert_resample (vf, height);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_horiz_resample_take (SchroFrame *vf, int width)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_horiz_resample (vf, width);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_unpack_take (SchroFrame *vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_unpack (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_YUY2_take (SchroFrame *vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_YUY2 (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_UYVY_take (SchroFrame *vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_UYVY (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

SchroFrame *
schro_virt_frame_new_pack_AYUV_take (SchroFrame *vf)
{
  SchroFrame *virt_frame;
  virt_frame = schro_virt_frame_new_pack_AYUV (vf);
  schro_frame_unref (vf);
  return virt_frame;
}

