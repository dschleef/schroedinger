
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>

void
schro_opengl_wavelet_transform_2d (SchroFrameData *frame_data, int type)
{
}

void
schro_opengl_wavelet_inverse_transform_2d (SchroFrameData *frame_data,
    int type)
{
  int width, height;
  SchroOpenGLFrameData *opengl_data = NULL;
  SchroOpenGLShader *shader_vertical_filter_xlp;
  SchroOpenGLShader *shader_vertical_filter_xhp;
  SchroOpenGLShader *shader_vertical_interleave;
  SchroOpenGLShader *shader_horizontal_filter_lp;
  SchroOpenGLShader *shader_horizontal_filter_hp;
  SchroOpenGLShader *shader_horizontal_interleave;

  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_S16);
  SCHRO_ASSERT (frame_data->width % 2 == 0);
  SCHRO_ASSERT (frame_data->height % 2 == 0);

  /*switch (type) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      break;
    case SCHRO_WAVELET_HAAR_0:
      break;
    case SCHRO_WAVELET_HAAR_1:
      break;
    case SCHRO_WAVELET_FIDELITY:
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      break;
  }*/

  schro_opengl_lock ();

  shader_vertical_filter_xlp
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_VERTICAL_FILTER_XLp);
  shader_vertical_filter_xhp
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_VERTICAL_FILTER_XHp);
  shader_vertical_interleave
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_VERTICAL_INTERLEAVE);

  SCHRO_ASSERT (shader_vertical_filter_xlp);
  SCHRO_ASSERT (shader_vertical_filter_xhp);
  SCHRO_ASSERT (shader_vertical_interleave);

  shader_horizontal_filter_lp
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_HORIZONTAL_FILTER_Lp);
  shader_horizontal_filter_hp
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_HORIZONTAL_FILTER_Hp);
  shader_horizontal_interleave
      = schro_opengl_shader_get
      (SCHRO_OPENGL_SHADER_INVERSE_WAVELET_HORIZONTAL_INTERLEAVE);

  SCHRO_ASSERT (shader_horizontal_filter_lp);
  SCHRO_ASSERT (shader_horizontal_filter_hp);
  SCHRO_ASSERT (shader_horizontal_interleave);

  width = frame_data->width;
  height = frame_data->height;
  opengl_data = (SchroOpenGLFrameData *) frame_data->data;

  SCHRO_ASSERT (opengl_data != NULL);

  schro_opengl_setup_viewport (width, height);

  /* pass 1: primary -> secondary, vertical filtering => XL' */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[1]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[0]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_vertical_filter_xlp->program);
  glUniform1iARB (shader_vertical_filter_xlp->textures[0], 0);
  glUniform2fARB (shader_vertical_filter_xlp->offset, 0, height / 2);

  schro_opengl_render_quad (0, 0, width, height / 2);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 2: secondary -> primary, vertical filtering => XH' */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[0]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[1]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_vertical_filter_xhp->program);
  glUniform1iARB (shader_vertical_filter_xhp->textures[0], 0);
  glUniform2fARB (shader_vertical_filter_xhp->offset, 0, height / 2);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, 0, width, height / 2);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 3: primary -> secondary, vertical interleave */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[1]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[0]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_vertical_interleave->program);
  glUniform1iARB (shader_vertical_interleave->textures[0], 0);
  glUniform2fARB (shader_vertical_interleave->offset, 0, height / 2);

  schro_opengl_render_quad (0, 0, width, height);

  glUseProgramObjectARB (0);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 4: secondary -> primary, horizontal filtering => L' */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[0]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[1]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_horizontal_filter_lp->program);
  glUniform1iARB (shader_horizontal_filter_lp->textures[0], 0);
  glUniform2fARB (shader_horizontal_filter_lp->offset, width / 2, 0);

  schro_opengl_render_quad (0, 0, width / 2, height);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (width / 2, 0, width / 2, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 5: primary -> secondary, horizontal filtering => H' */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[1]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[0]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_horizontal_filter_hp->program);
  glUniform1iARB (shader_horizontal_filter_hp->textures[0], 0);
  glUniform2fARB (shader_horizontal_filter_hp->offset, width / 2, 0);

  SCHRO_OPENGL_CHECK_ERROR

  schro_opengl_render_quad (width / 2, 0, width / 2, height);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, 0, width / 2, height);

  SCHRO_OPENGL_CHECK_ERROR // failes

  glFlush ();

  /* pass 6: secondary -> primary, horizontal interleave */
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[0]);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[1]);

  SCHRO_OPENGL_CHECK_ERROR

  glUseProgramObjectARB (shader_horizontal_interleave->program);
  glUniform1iARB (shader_horizontal_interleave->textures[0], 0);
  glUniform2fARB (shader_horizontal_interleave->offset, width / 2, 0);

  schro_opengl_render_quad (0, 0, width, height);

  glUseProgramObjectARB (0);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock ();
}

