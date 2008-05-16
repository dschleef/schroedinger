
#ifndef __SCHRO_OPENGL_SHADER_H__
#define __SCHRO_OPENGL_SHADER_H__

#include <schroedinger/schro.h>
#include <GL/glew.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroOpenGLShader SchroOpenGLShader;

struct _SchroOpenGLShader {
  GLhandleARB program;
  GLint textures[3];
};

#define SCHRO_OPENGL_SHADER_IDENTITY             0
#define SCHRO_OPENGL_SHADER_CONVERT_U8_S16       1
#define SCHRO_OPENGL_SHADER_CONVERT_S16_U8       2
#define SCHRO_OPENGL_SHADER_CONVERT_U8_U8        3
#define SCHRO_OPENGL_SHADER_CONVERT_S16_S16      4
#define SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV   5
#define SCHRO_OPENGL_SHADER_CONVERT_U8_U2_YUYV   6
#define SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV   7
#define SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY   8
#define SCHRO_OPENGL_SHADER_CONVERT_U8_U2_UYVY   9
#define SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY  10
#define SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV  11
#define SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV  12
#define SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV  13
#define SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422 14
#define SCHRO_OPENGL_SHADER_CONVERT_UYVY_U8_422 15
#define SCHRO_OPENGL_SHADER_CONVERT_AYUV_U8_444 16

SchroOpenGLShader *schro_opengl_shader_new (const char* code, int textures);
void schro_opengl_shader_free (SchroOpenGLShader *shader);
SchroOpenGLShader *schro_opengl_shader_get (int index);

SCHRO_END_DECLS

#endif

