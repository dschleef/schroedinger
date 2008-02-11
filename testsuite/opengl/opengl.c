
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>

#include "schroopengl.h"
#include "../common.h"

#include "glextensions.h"




int main (int argc, char *argv[])
{
  SchroOpenGL *gl;
  int ret;
  SchroFrame *frame;
  SchroFrame *frame16;
  SchroFrame *frame16_test;
  char name[TEST_PATTERN_NAME_SIZE];
  GLuint fbo;
  GLuint texture;
  GLuint rgba_texture;

  schro_init();

  gl = schro_opengl_new ();
  SCHRO_ERROR("gl %p", gl);

  ret = schro_opengl_connect (gl, NULL);
  SCHRO_ERROR("ret %d", ret);

  schro_opengl_lock (gl);

  frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_U8_444, 16, 16);

  test_pattern_generate (frame->components + 0, name, 8);
  test_pattern_generate (frame->components + 1, name, 8);
  test_pattern_generate (frame->components + 2, name, 8);

  frame16 = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444, 16, 16);
  schro_frame_convert (frame16, frame);
  {
    int i,j;
    for(j=0;j<16;j+=1){
      int16_t *d = SCHRO_FRAME_DATA_GET_LINE(frame16->components+0, j);
      for(i=0;i<16;i++){
        d[i] = 10*128 + i*16;
      }
    }
  }

  glGenTextures (1, &texture);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, texture);

  glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_LUMINANCE16, 16, 16,
      0, GL_LUMINANCE, GL_SHORT, NULL);
  glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, 16, 16,
      GL_LUMINANCE, GL_SHORT, frame16->components[0].data);

  schro_opengl_unlock (gl);

  /* download */

  schro_opengl_lock (gl);

  frame16_test = schro_frame_new_and_alloc (NULL,
      SCHRO_FRAME_FORMAT_S16_444, 16, 16);

  glGenFramebuffersEXT (1, &fbo);
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, fbo);
  glGenTextures (1, &rgba_texture);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, rgba_texture);

  glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, 16, 16,
      0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT,
      GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, rgba_texture, 0);

  glDrawBuffer (GL_COLOR_ATTACHMENT1_EXT);
  glReadBuffer (GL_COLOR_ATTACHMENT1_EXT);
  schro_opengl_check_error (gl, __LINE__);

  SCHRO_ASSERT (glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT) ==
         GL_FRAMEBUFFER_COMPLETE_EXT);

  glViewport (0, 0, 16, 16);
  glClearColor (0.3, 0.3, 0.3, 1.0);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();

  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();

  glDisable (GL_CULL_FACE);
  glEnableClientState (GL_TEXTURE_COORD_ARRAY);

  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, texture);
  glEnable (GL_TEXTURE_RECTANGLE_ARB);

  {
#if 0
    const double matrix[16] = {
      1, 1, 1, 0,
      0, -0.344 * 1, 1.770 * 1, 0,
      1.403 * 1, -0.714 * 1, 0, 0,
      0, 0, 0, 1
    };
    glMatrixMode (GL_COLOR);
    glLoadMatrixd (matrix);
#endif
    glPixelTransferf (GL_POST_COLOR_MATRIX_RED_SCALE, 4);
    //glPixelTransferf (GL_POST_COLOR_MATRIX_GREEN_BIAS, (0.344 + 0.714) / 2);
    //glPixelTransferf (GL_POST_COLOR_MATRIX_BLUE_BIAS, -1.770 / 2);
  }

  glColor4f (0.5, 0, 0, 0);

  glBegin (GL_QUADS);
  glNormal3f (0, 0, -1);
  glTexCoord2f (16.0, 0);
  glVertex3f (1.0, -1.0, 0);
  glTexCoord2f (0, 0);
  glVertex3f (-1.0, -1.0, 0);
  glTexCoord2f (0, 16.0);
  glVertex3f (-1.0, 1.0, 0);
  glTexCoord2f (16.0, 16.0);
  glVertex3f (1.0, 1.0, 0);
  glEnd ();
  glFlush();

  {
    unsigned char *data;
    int i, j;

    data = malloc (16 * 16 * 4);
    glReadPixels (0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, data);
    for(j=0;j<16;j++) {
      int16_t *d = SCHRO_FRAME_DATA_GET_LINE(frame16_test->components+0, j);
      for(i=0;i<16;i++) {
        d[i] = data[4*(i+16*j)];
      }
    }
    free(data);
  }

  glDeleteFramebuffersEXT (1, &fbo);
  schro_opengl_unlock (gl);

  frame_dump (frame16_test, frame16);

  return 0;
}


