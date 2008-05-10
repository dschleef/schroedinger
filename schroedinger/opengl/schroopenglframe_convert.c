
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <liboil/liboil.h>

static void schro_opengl_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_s16_s16 (SchroFrame *dest, SchroFrame *src);/*
static void schro_opengl_frame_convert_u8_422_yuyv (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_422_uyvy (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_u8_444_ayuv (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_yuyv_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_uyvy_u8_422 (SchroFrame *dest, SchroFrame *src);
static void schro_opengl_frame_convert_ayuv_u8_444 (SchroFrame *dest, SchroFrame *src);*/

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
/*
  { SCHRO_FRAME_FORMAT_YUYV, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_422_yuyv },
  { SCHRO_FRAME_FORMAT_UYVY, SCHRO_FRAME_FORMAT_U8_422, schro_opengl_frame_convert_u8_422_uyvy },
  { SCHRO_FRAME_FORMAT_AYUV, SCHRO_FRAME_FORMAT_U8_444, schro_opengl_frame_convert_u8_444_ayuv },

  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV, schro_opengl_frame_convert_yuyv_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_UYVY, schro_opengl_frame_convert_uyvy_u8_422 },
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_AYUV, schro_opengl_frame_convert_ayuv_u8_444 },*/

  { 0 }
};

void
schro_opengl_frame_convert (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (src));

  for (i = 0; schro_opengl_frame_convert_func_list[i].func; ++i) {
    if (schro_opengl_frame_convert_func_list[i].from == src->format
        && schro_opengl_frame_convert_func_list[i].to == dest->format) {
      schro_opengl_frame_convert_func_list[i].func (dest, src);
      return;
    }
  }

  SCHRO_ERROR ("conversion unimplemented (%d -> %d)", src->format,
      dest->format);
  SCHRO_ASSERT (0);
}

static void
schro_opengl_frame_convert_with_shader (SchroFrame *dest, SchroFrame *src,
    int shader_index)
{
  int i;
  int width, height;
  SchroOpenGLFrameData *dest_opengl_data = NULL;
  SchroOpenGLFrameData *src_opengl_data = NULL;
  SchroOpenGLShader *shader;

  schro_opengl_lock ();

  shader = schro_opengl_shader_get (shader_index);

  SCHRO_ASSERT (shader);

  for (i = 0; i < 3; ++i) {
    dest_opengl_data = (SchroOpenGLFrameData *) dest->components[i].data;
    src_opengl_data = (SchroOpenGLFrameData *) src->components[i].data;

    SCHRO_ASSERT (dest_opengl_data != NULL);
    SCHRO_ASSERT (src_opengl_data != NULL);

    width = MAX (dest->components[i].width, src->components[i].width);
    height = MAX (dest->components[i].height, src->components[i].height);

    glViewport (0, 0, width, height);

    glLoadIdentity ();
    glOrtho (0, width, 0, height, -1, 1);

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, dest_opengl_data->framebuffer);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, src_opengl_data->texture.handle);

    SCHRO_OPENGL_CHECK_ERROR

    glUseProgramObjectARB (shader->program);
    glUniform1iARB (shader->texture, 0);

    glBegin (GL_QUADS);
    glTexCoord2f (width, 0);      glVertex3f (width, 0,      0);
    glTexCoord2f (0,     0);      glVertex3f (0,     0,      0);
    glTexCoord2f (0,     height); glVertex3f (0,     height, 0);
    glTexCoord2f (width, height); glVertex3f (width, height, 0);
    glEnd ();

    glUseProgramObjectARB (0);

    glFlush ();

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
  }

  schro_opengl_unlock ();
}

static void
schro_opengl_frame_convert_u8_s16 (SchroFrame *dest, SchroFrame *src)
{
  schro_opengl_frame_convert_with_shader (dest, src,
      SCHRO_OPENGL_SHADER_CONVERT_U8_S16);
}

static void
schro_opengl_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src)
{
  schro_opengl_frame_convert_with_shader (dest, src,
      SCHRO_OPENGL_SHADER_CONVERT_S16_U8);
}

static void
schro_opengl_frame_convert_u8_u8 (SchroFrame *dest, SchroFrame *src)
{
  schro_opengl_frame_convert_with_shader (dest, src,
      SCHRO_OPENGL_SHADER_CONVERT_U8_U8);
}

static void
schro_opengl_frame_convert_s16_s16 (SchroFrame *dest, SchroFrame *src)
{
  schro_opengl_frame_convert_with_shader (dest, src,
      SCHRO_OPENGL_SHADER_CONVERT_S16_S16);
}

