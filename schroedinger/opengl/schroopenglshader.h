
#ifndef __SCHRO_OPENGL_SHADER_H__
#define __SCHRO_OPENGL_SHADER_H__

#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <GL/glew.h>

SCHRO_BEGIN_DECLS

typedef struct _SchroOpenGLShader SchroOpenGLShader;

struct _SchroOpenGLShader {
  GLhandleARB program;
  GLint textures[3];
  GLint offset;
  GLint four_decrease;
  GLint three_decrease;
  GLint two_decrease;
  GLint one_decrease;
  GLint one_increase;
  GLint two_increase;
  GLint three_increase;
  GLint four_increase;
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
#define SCHRO_OPENGL_SHADER_ADD_S16_U8          17
#define SCHRO_OPENGL_SHADER_ADD_S16_S16         18
#define SCHRO_OPENGL_SHADER_SUBTRACT_S16_U8     19
#define SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16    20
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Lp  21
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Hp  22
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Lp            23
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Hp            24
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Lp 25
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Hp 26
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Lp                   27
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Hp                   28
#define SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_L          29
#define SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_H          30
#define SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_INTERLEAVE              31
#define SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_INTERLEAVE            32
#define SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_SHIFT                     33
#define SCHRO_OPENGL_SHADER_UPSAMPLE_U8                               34

#define SCHRO_OPENGL_SHADER_COUNT \
    ((SCHRO_OPENGL_SHADER_UPSAMPLE_U8) + 1)

SchroOpenGLShaderLibrary *schro_opengl_shader_library_new (SchroOpenGL *opengl);
void schro_opengl_shader_library_free (SchroOpenGLShaderLibrary *library);
SchroOpenGLShader *schro_opengl_shader_get (SchroOpenGL *opengl, int index);

SCHRO_END_DECLS

#endif

