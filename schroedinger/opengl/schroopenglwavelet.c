
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
  int framebuffer_index, texture_index;
  int filter_shift = FALSE;
  SchroOpenGLFrameData *opengl_data = NULL;
  SchroOpenGL *opengl = NULL;
  SchroOpenGLShader *shader_vertical_deinterleave_xl;
  SchroOpenGLShader *shader_vertical_deinterleave_xh;
  SchroOpenGLShader *shader_vertical_filter_xlp;
  SchroOpenGLShader *shader_vertical_filter_xhp;
  SchroOpenGLShader *shader_vertical_interleave;
  SchroOpenGLShader *shader_horizontal_filter_lp;
  SchroOpenGLShader *shader_horizontal_filter_hp;
  SchroOpenGLShader *shader_horizontal_interleave;
  SchroOpenGLShader *shader_filter_shift;

  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_S16);
  SCHRO_ASSERT (frame_data->width % 2 == 0);
  SCHRO_ASSERT (frame_data->height % 2 == 0);

  switch (type) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_HAAR_0:
      filter_shift = FALSE;
      break;
    case SCHRO_WAVELET_HAAR_1:
      filter_shift = TRUE;
      break;
    case SCHRO_WAVELET_FIDELITY:
      filter_shift = FALSE;
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      filter_shift = TRUE;
      break;
    default:
      SCHRO_ERROR ("unknown type %i", type);
      SCHRO_ASSERT (0);
      break;
  }

  width = frame_data->width;
  height = frame_data->height;
  opengl_data = (SchroOpenGLFrameData *) frame_data->data;

  SCHRO_ASSERT (opengl_data != NULL);

  opengl = opengl_data->opengl;

  schro_opengl_lock (opengl);

  shader_vertical_deinterleave_xl = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_DEINTERLEAVE_XL);
  shader_vertical_deinterleave_xh = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_DEINTERLEAVE_XH);
  shader_vertical_filter_xlp = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_FILTER_XLp);
  shader_vertical_filter_xhp = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_FILTER_XHp);
  shader_vertical_interleave = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_INTERLEAVE);

  SCHRO_ASSERT (shader_vertical_filter_xlp);
  SCHRO_ASSERT (shader_vertical_filter_xhp);
  SCHRO_ASSERT (shader_vertical_interleave);

  shader_horizontal_filter_lp = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_FILTER_Lp);
  shader_horizontal_filter_hp = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_FILTER_Hp);
  shader_horizontal_interleave = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_INTERLEAVE);

  SCHRO_ASSERT (shader_horizontal_filter_lp);
  SCHRO_ASSERT (shader_horizontal_filter_hp);
  SCHRO_ASSERT (shader_horizontal_interleave);

  if (filter_shift) {
    shader_filter_shift = schro_opengl_shader_get (opengl,
        SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_FILTER_SHIFT);

    SCHRO_ASSERT (shader_filter_shift);
  } else {
    shader_filter_shift = NULL;
  }

  schro_opengl_setup_viewport (width, height);

  SCHRO_OPENGL_CHECK_ERROR

  #define SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES \
      framebuffer_index = 1 - framebuffer_index; \
      texture_index = 1 - texture_index; \
      SCHRO_ASSERT (framebuffer_index != texture_index);

  #define BIND_FRAMEBUFFER_AND_TEXTURE \
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, \
          opengl_data->framebuffers[framebuffer_index]); \
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, \
          opengl_data->texture.handles[texture_index]); \
      SCHRO_OPENGL_CHECK_ERROR

  framebuffer_index = 1;
  texture_index = 0;

  /* pass 0: vertical deinterleave */
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_deinterleave_xl->program);
  glUniform1iARB (shader_vertical_deinterleave_xl->textures[0], 0);

  schro_opengl_render_quad (0, 0, width, height / 2);

  glUseProgramObjectARB (shader_vertical_deinterleave_xh->program);
  glUniform1iARB (shader_vertical_deinterleave_xh->textures[0], 0);
  glUniform2fARB (shader_vertical_deinterleave_xh->offset, 0, height / 2);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  glUseProgramObjectARB (0);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 1: vertical filtering => XL + f(XH) = XL' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_filter_xlp->program);
  glUniform1iARB (shader_vertical_filter_xlp->textures[0], 0);
  glUniform2fARB (shader_vertical_filter_xlp->offset, 0, height / 2);

  schro_opengl_render_quad (0, 0, width, height / 2);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 2: vertical filtering => f(XL') + XH = XH' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_filter_xhp->program);
  glUniform1iARB (shader_vertical_filter_xhp->textures[0], 0);
  glUniform2fARB (shader_vertical_filter_xhp->offset, 0, height / 2);

  schro_opengl_render_quad (0, height / 2, width, height / 2);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, 0, width, height / 2);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 3: vertical interleave => i(LL', LH') = L, i(HL', HH') = H */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_vertical_interleave->program);
  glUniform1iARB (shader_vertical_interleave->textures[0], 0);
  glUniform2fARB (shader_vertical_interleave->offset, 0, height / 2);

  schro_opengl_render_quad (0, 0, width, height);

  glUseProgramObjectARB (0);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 4: horizontal filtering => L + f(H) = L' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_horizontal_filter_lp->program);
  glUniform1iARB (shader_horizontal_filter_lp->textures[0], 0);
  glUniform2fARB (shader_horizontal_filter_lp->offset, width / 2, 0);

  schro_opengl_render_quad (0, 0, width / 2, height);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (width / 2, 0, width / 2, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 5: horizontal filtering => f(L') + H = H' */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_horizontal_filter_hp->program);
  glUniform1iARB (shader_horizontal_filter_hp->textures[0], 0);
  glUniform2fARB (shader_horizontal_filter_hp->offset, width / 2, 0);

  SCHRO_OPENGL_CHECK_ERROR

  schro_opengl_render_quad (width / 2, 0, width / 2, height);

  glUseProgramObjectARB (0);

  schro_opengl_render_quad (0, 0, width / 2, height);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 6: horizontal interleave => i(L', H') = LL */
  SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  BIND_FRAMEBUFFER_AND_TEXTURE

  glUseProgramObjectARB (shader_horizontal_interleave->program);
  glUniform1iARB (shader_horizontal_interleave->textures[0], 0);
  glUniform2fARB (shader_horizontal_interleave->offset, width / 2, 0);

  schro_opengl_render_quad (0, 0, width, height);

  glUseProgramObjectARB (0);

  SCHRO_OPENGL_CHECK_ERROR

  glFlush ();

  /* pass 7: filter shift */
  if (filter_shift) {
    SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
    BIND_FRAMEBUFFER_AND_TEXTURE

    glUseProgramObjectARB (shader_filter_shift->program);
    glUniform1iARB (shader_filter_shift->textures[0], 0);

    schro_opengl_render_quad (0, 0, width, height);

    glUseProgramObjectARB (0);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush ();
  }

  /* pass 8: transfer data from secondary to primary framebuffer if previous
             pass result wasn't rendered into the primary framebuffer */
  if (framebuffer_index != 0) {
    SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
    BIND_FRAMEBUFFER_AND_TEXTURE

    schro_opengl_render_quad (0, 0, width, height);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush ();
  }

  #undef SWITCH_FRAMEBUFFER_AND_TEXTURE_INDICES
  #undef BIND_FRAMEBUFFER_AND_TEXTURE

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock (opengl);
}

