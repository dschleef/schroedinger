
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <liboil/liboil.h>

/*
static void schro_opengl_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_s16_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_422_yuyv (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_422_uyvy (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_444_ayuv (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_yuyv_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_uyvy_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_ayuv_u8_444 (SchroFrame *dest, SchroFrame *src);

typedef void (*SchroOpenGLFrameBinaryFunc) (SchroFrame *dest, SchroFrame *src);

struct binary_struct {
  SchroFrameFormat from;
  SchroFrameFormat to;
  SchroOpenGLFrameBinaryFunc func;
};

static struct binary_struct schro_opengl_frame_convert_func_list[] = {
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, schro_opengl_frame_convert_u8_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, schro_opengl_frame_convert_u8_s16 },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, schro_opengl_frame_convert_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, schro_opengl_frame_convert_s16_u8 },
  { SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, schro_opengl_frame_convert_s16_u8 },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_U8_444, schro_opengl_frame_convert_u8_u8 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_u8 },
  { SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_U8_420, schro_opengl_frame_convert_u8_u8 },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, schro_opengl_frame_convert_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, schro_opengl_frame_convert_s16_s16 },
  { SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, schro_opengl_frame_convert_s16_s16 },

  { SCHRO_FRAME_FORMAT_YUYV, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_422_yuyv },
  { SCHRO_FRAME_FORMAT_UYVY, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_422_uyvy },
  { SCHRO_FRAME_FORMAT_AYUV, SCHRO_FRAME_FORMAT_U8_444, schro_opengl_frame_convert_u8_444_ayuv },

  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV, schro_opengl_frame_convert_yuyv_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_UYVY, schro_opengl_frame_convert_uyvy_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_AYUV, schro_opengl_frame_convert_ayuv_u8_444 },
  { 0 }
};*/

void
schro_opengl_frame_convert (SchroFrame *dest, SchroFrame *src)
{
/*  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);

  for (i = 0; schro_opengl_frame_convert_func_list[i].func; i++) {
    if (schro_opengl_frame_convert_func_list[i].from == src->format &&
        schro_opengl_frame_convert_func_list[i].to == dest->format) {
      schro_opengl_frame_convert_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR("conversion unimplemented (%d -> %d)", src->format, dest->format);
  SCHRO_ASSERT(FALSE);*/
}

/*
static const char* code_s16_u8 =
"uniform sampler2DRect texture;\n"
"void main() {\n"
"  if (mod(gl_TexCoord[0].x, 2.0) == 0.5) {\n"
"    gl_FragColor = texture2DRect(texture, gl_TexCoord[0]);\n"
"  } else {\n"
"    gl_FragColor = vec4(0.0);\n"
"  }\n"
"}\n\0";*/
