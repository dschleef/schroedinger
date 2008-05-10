
#ifndef __SCHRO_OPENGL_SHADER_H__
#define __SCHRO_OPENGL_SHADER_H__

#include <schroedinger/schro.h>
#include <GL/gl.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroOpenGLShader SchroOpenGLShader;

struct _SchroOpenGLShader {
  GLhandleARB program;
  GLint texture;
};

#define SCHRO_OPENGL_SHADER_IDENTITY         0
#define SCHRO_OPENGL_SHADER_CONVERT_U8_S16   1
#define SCHRO_OPENGL_SHADER_CONVERT_S16_U8   2
#define SCHRO_OPENGL_SHADER_CONVERT_U8_U8    3
#define SCHRO_OPENGL_SHADER_CONVERT_S16_S16  4

SchroOpenGLShader *schro_opengl_shader_new (const char* code);
void schro_opengl_shader_free (SchroOpenGLShader *shader);
SchroOpenGLShader *schro_opengl_shader_get (int index);

SCHRO_END_DECLS

#endif

