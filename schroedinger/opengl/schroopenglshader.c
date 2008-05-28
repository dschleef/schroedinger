
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/opengl/schroopenglshader.h>

static int
schro_opengl_shader_check_status (GLhandleARB handle, GLenum status,
    const char* code)
{
  GLint result;
  GLint length;
  char* infolog;

  glGetObjectParameterivARB(handle, status, &result);

  if (result != 0) {
    return TRUE;
  }

  glGetObjectParameterivARB(handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);

  if (length < 1) {
    SCHRO_ERROR ("invalid infolog length %d", length);
    return FALSE;
  }

  infolog = schro_malloc0(length * sizeof (char));

  glGetInfoLogARB(handle, length, &length, infolog);

  SCHRO_ERROR ("\n%s\n%s", code, infolog);

  schro_free(infolog);

  return FALSE;
}

SchroOpenGLShader *
schro_opengl_shader_new (const char* code, int textures, int offset)
{
  SchroOpenGLShader *shader;
  GLhandleARB handle;
  int ok;

  shader = schro_malloc0 (sizeof (SchroOpenGLShader));
  handle = glCreateShaderObjectARB (GL_FRAGMENT_SHADER_ARB);

  glShaderSourceARB (handle, 1, (const char**)&code, 0);
  glCompileShaderARB (handle);

  ok = schro_opengl_shader_check_status (handle, GL_OBJECT_COMPILE_STATUS_ARB,
      code);

  SCHRO_ASSERT (ok);

  shader->program = glCreateProgramObjectARB ();

  glAttachObjectARB (shader->program, handle);
  glDeleteObjectARB (handle);
  glLinkProgramARB (shader->program);

  ok = schro_opengl_shader_check_status (shader->program,
      GL_OBJECT_LINK_STATUS_ARB, code);

  SCHRO_ASSERT (ok);

  glValidateProgramARB (shader->program);

  ok = schro_opengl_shader_check_status (shader->program,
      GL_OBJECT_VALIDATE_STATUS_ARB, code);

  SCHRO_ASSERT (ok);

  switch (textures) {
    case 1:
      shader->textures[0] = glGetUniformLocationARB (shader->program,
          "texture");
      break;
    case 2:
      shader->textures[0] = glGetUniformLocationARB (shader->program,
          "texture1");
      shader->textures[1] = glGetUniformLocationARB (shader->program,
          "texture2");
      break;
    case 3:
      shader->textures[0] = glGetUniformLocationARB (shader->program,
          "texture1");
      shader->textures[1] = glGetUniformLocationARB (shader->program,
          "texture2");
      shader->textures[2] = glGetUniformLocationARB (shader->program,
          "texture3");
      break;
    default:
      SCHRO_ERROR ("unhandled count of texture unforms: %i", textures);
      SCHRO_ASSERT (0);
      break;
  }

  if (offset) {
    shader->offset = glGetUniformLocationARB (shader->program, "offset");
  }

  return shader;
}

void
schro_opengl_shader_free (SchroOpenGLShader *shader)
{
  SCHRO_ASSERT (shader != NULL);

  glDeleteObjectARB (shader->program);

  schro_free (shader);
}

struct IndexToShader {
  int index;
  int textures;
  int offset;
  SchroOpenGLShader *shader;
  const char *code;
};

static struct IndexToShader schro_opengl_shader_list[] = {
  { SCHRO_OPENGL_SHADER_IDENTITY, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* S16 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect (texture, gl_TexCoord[0].xy) - bias) / scale;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = texture2DRect (texture, gl_TexCoord[0]) * scale + bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 yuyv = texture2DRect (texture, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = yuyv.r;\n"
      "  } else {\n"
      "    gl_FragColor = yuyv.b;\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_YUYV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = yuyv.g;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = yuyv.a;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* UYVY */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 uyvy = texture2DRect (texture, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = uyvy.g;\n"
      "  } else {\n"
      "    gl_FragColor = uyvy.a;\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_UYVY, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = uyvy.r;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = uyvy.b;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = ayuv.g;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = ayuv.b;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV, 1, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = ayuv.a;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422, 3, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 yuyv;\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord1 = vec2 (floor (x) + 0.5, y);\n" /* FIXME: floor */
      "  vec2 coord2 = vec2 (floor (x) + 1.5, y);\n" /* FIXME: floor */
      "  yuyv.r = texture2DRect (texture1, coord1).r;\n"
      "  yuyv.g = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  yuyv.b = texture2DRect (texture1, coord2).r;\n"
      "  yuyv.a = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  gl_FragColor = yuyv;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_UYVY_U8_422, 3, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 uyvy;\n"
      /* round x coordinate down from texel center n.5 to n.0 and scale up to
         double width */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord1 = vec2 (floor (x) + 0.5, y);\n" /* FIXME: floor */
      "  vec2 coord2 = vec2 (floor (x) + 1.5, y);\n" /* FIXME: floor */
      "  uyvy.r = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  uyvy.g = texture2DRect (texture1, coord1).r;\n"
      "  uyvy.b = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  uyvy.a = texture2DRect (texture1, coord2).r;\n"
      "  gl_FragColor = uyvy;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_AYUV_U8_444, 3, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U4 */
      "uniform sampler2DRect texture3;\n" /* V4 */
      "void main (void) {\n"
      "  vec4 ayuv;\n"
      "  ayuv.r = 1.0;\n"
      "  ayuv.g = texture2DRect (texture1, gl_TexCoord[0].xy).r;\n"
      "  ayuv.b = texture2DRect (texture2, gl_TexCoord[0].xy).r;\n"
      "  ayuv.a = texture2DRect (texture3, gl_TexCoord[0].xy).r;\n"
      "  gl_FragColor = ayuv;\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_ADD_S16_U8, 2, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_ADD_S16_S16, 2, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_U8, 2, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n" },
  /* FIXME: CPU overflows, GPU clamps, is this a problem? */
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16, 2, FALSE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0].xy) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0].xy) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_DEINTERLEAVE_XL, 1, FALSE,
      NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (x, floor (y * 2.0) + 0.5);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture, coord);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_DEINTERLEAVE_XH, 1, TRUE,
      NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* height of subband XL */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y) - offset.y;\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (x, floor (y * 2.0) + 1.5);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture, coord);\n"
      "}\n" },

#define FILTER_HAAR0_STEP1 \
      "float filter_haar0_step1 (float value) {\n" \
      /* scale with 65536.0 (instead of 65535.0, to get correct mappings when \
         applying floor) from FP [-0.5..0.5] to real S16 [-32768..32767] */ \
      "  float input = floor (value * 65536.0);\n" \
      /* apply haar filter */ \
      "  float output = -floor ((input + 1.0) / 2.0);\n" \
      /* scale from real S16 [-32768..32767] to FP [-0.5..0.5] */ \
      "  return output / 65535.0;\n" \
      "}\n"

  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_FILTER_XLp, 1, TRUE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "const float bias = -32768.0 / 65535.0;\n"
      FILTER_HAAR0_STEP1
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  float xl = texture2DRect (texture, gl_TexCoord[0].xy).r + bias;\n"
      "  float xh = texture2DRect (texture, gl_TexCoord[0].xy + offset).r\n"
      "      + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (xl + filter_haar0_step1 (xh)) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_FILTER_XHp, 1, TRUE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* vertical distance between two corresponding texels from subband XL'
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "const float bias = -32768.0 / 65535.0;\n"
      "float filter_haar0_step2 (float value) {\n"
      "  return value;\n"
      "}\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  float xlp = texture2DRect (texture, gl_TexCoord[0].xy - offset).r\n"
      "      + bias;\n"
      "  float xh = texture2DRect (texture, gl_TexCoord[0].xy).r + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (filter_haar0_step2 (xlp) + xh) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_VERTICAL_INTERLEAVE, 1, TRUE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      "  if (mod (y, 2.0) < 0.5) {\n"
      "    y = floor (y / 2.0);\n"
      "  } else {\n"
      "    y = floor (y / 2.0) + offset.y;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (x, floor (y) + 0.5);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture, coord);\n"
      "}\n" },
  /* FIXME: may merge with vertical version, the code is the same only the
     value bound to offset differs */
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_FILTER_Lp, 1, TRUE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* horizontal distance between two corresponding texels from subband L
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "const float bias = -32768.0 / 65535.0;\n"
      FILTER_HAAR0_STEP1
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  float lx = texture2DRect (texture, gl_TexCoord[0].xy).r + bias;\n"
      "  float hx = texture2DRect (texture, gl_TexCoord[0].xy + offset).r\n"
      "      + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (lx + filter_haar0_step1 (hx)) - bias;\n"
      "}\n" },
  /* FIXME: may merge with vertical version, the code is the same only the
     value bound to offset differs */
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_FILTER_Hp, 1, TRUE, NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* horizontal distance between two corresponding texels from subband L'
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "const float bias = -32768.0 / 65535.0;\n"
      "float filter_haar0_step2 (float value) {\n"
      "  return value;\n"
      "}\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  float lxp = texture2DRect (texture, gl_TexCoord[0].xy - offset).r\n"
      "      + bias;\n"
      "  float hx = texture2DRect (texture, gl_TexCoord[0].xy).r + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (filter_haar0_step2 (lxp) + hx) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_INTERLEAVE, 1, TRUE,
      NULL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture;\n"
      /* horizontal distance between two corresponding texels from subband L'
         and H' in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x);\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  if (mod (x, 2.0) < 0.5) {\n"
      "    x = floor (x / 2.0);\n"
      "  } else {\n"
      "    x = floor (x / 2.0) + offset.x;\n"
      "  }\n"
      /* shift y coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture, coord);\n"
      "}\n" },

  { -1, 0, FALSE, NULL, NULL }
};

SchroOpenGLShader *
schro_opengl_shader_get (int index)
{
  int i;

  SCHRO_ASSERT (index >= SCHRO_OPENGL_SHADER_IDENTITY);
  SCHRO_ASSERT (index
     <= SCHRO_OPENGL_SHADER_INVERSE_WAVELET_S16_HORIZONTAL_INTERLEAVE);

  for (i = 0; schro_opengl_shader_list[i].code; ++i) {
    if (schro_opengl_shader_list[i].index == index) {
      if (!schro_opengl_shader_list[i].shader) {
          schro_opengl_shader_list[i].shader
              = schro_opengl_shader_new (schro_opengl_shader_list[i].code,
              schro_opengl_shader_list[i].textures,
              schro_opengl_shader_list[i].offset);
      }

      return schro_opengl_shader_list[i].shader;
    }
  }

  SCHRO_ERROR ("no shader for index %i", index);
  SCHRO_ASSERT (0);

  return NULL;
}

