
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/opengl/schroopenglshader.h>

SchroOpenGLShader *
schro_opengl_shader_new (const char* code, int textures)
{
  SchroOpenGLShader *shader;
  GLhandleARB handle;
  GLint status;

  shader = schro_malloc0 (sizeof (SchroOpenGLShader));
  handle = glCreateShaderObjectARB (GL_FRAGMENT_SHADER_ARB);

  glShaderSourceARB (handle, 1, (const char**)&code, 0);
  glCompileShaderARB (handle);
  glGetObjectParameterivARB (handle, GL_OBJECT_COMPILE_STATUS_ARB, &status);

  SCHRO_ASSERT (status != 0);

  shader->program = glCreateProgramObjectARB ();

  glAttachObjectARB (shader->program, handle);
  glDeleteObjectARB (handle);
  glLinkProgramARB (shader->program);
  glGetObjectParameterivARB (shader->program, GL_OBJECT_LINK_STATUS_ARB,
      &status);

  SCHRO_ASSERT (status != 0);

  glValidateProgramARB (shader->program);
  glGetObjectParameterivARB (shader->program, GL_OBJECT_VALIDATE_STATUS_ARB,
      &status);

  SCHRO_ASSERT (status != 0);

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
  SchroOpenGLShader *shader;
  const char *code;
};

static struct IndexToShader schro_opengl_shader_list[] = {
  { SCHRO_OPENGL_SHADER_IDENTITY, 1, NULL,
      "uniform sampler2DRect texture;\n"
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0]);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16, 1, NULL,
      "uniform sampler2DRect texture;\n" /* S16 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect (texture, gl_TexCoord[0]) - bias) / scale;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8, 1, NULL,
      "uniform sampler2DRect texture;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = texture2DRect (texture, gl_TexCoord[0]) * scale + bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8, 1, NULL,
      "uniform sampler2DRect texture;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0]);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16, 1, NULL,
      "uniform sampler2DRect texture;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture, gl_TexCoord[0]);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV, 1, NULL,
      "uniform sampler2DRect texture;\n"
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 yuyv = texture2DRect (texture, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = yuyv.r;\n"
      "  } else {\n"
      "    gl_FragColor = yuyv.b;\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_YUYV, 1, NULL,
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = yuyv.g;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV, 1, NULL,
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = yuyv.a;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY, 1, NULL,
      "uniform sampler2DRect texture;\n" /* YUYV */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 uyvy = texture2DRect (texture, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = uyvy.g;\n"
      "  } else {\n"
      "    gl_FragColor = uyvy.a;\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_UYVY, 1, NULL,
      "uniform sampler2DRect texture;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = uyvy.r;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY, 1, NULL,
      "uniform sampler2DRect texture;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = uyvy.b;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV, 1, NULL,
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = ayuv.g;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV, 1, NULL,
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = ayuv.b;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV, 1, NULL,
      "uniform sampler2DRect texture;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture, gl_TexCoord[0]);\n"
      "  gl_FragColor = ayuv.a;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422, 3, NULL,
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 yuyv;\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  vec2 coord1 = vec2 (floor (x) + 0.5, y);\n"
      "  vec2 coord2 = vec2 (floor (x) + 1.5, y);\n"
      "  yuyv.r = texture2DRect (texture1, coord1).r;\n"
      "  yuyv.g = texture2DRect (texture2, gl_TexCoord[0]).r;\n"
      "  yuyv.b = texture2DRect (texture1, coord2).r;\n"
      "  yuyv.a = texture2DRect (texture3, gl_TexCoord[0]).r;\n"
      "  gl_FragColor = yuyv;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_UYVY_U8_422, 3, NULL,
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U2 */
      "uniform sampler2DRect texture3;\n" /* V2 */
      "void main (void) {\n"
      "  vec4 uyvy;\n"
      /* round x coordinate down from texel center n.5 to n.0 and scale up to
         double width */
      "  float x = floor (gl_TexCoord[0].x) * 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      "  vec2 coord1 = vec2 (floor (x) + 0.5, y);\n"
      "  vec2 coord2 = vec2 (floor (x) + 1.5, y);\n"
      "  uyvy.r = texture2DRect (texture2, gl_TexCoord[0]).r;\n"
      "  uyvy.g = texture2DRect (texture1, coord1).r;\n"
      "  uyvy.b = texture2DRect (texture3, gl_TexCoord[0]).r;\n"
      "  uyvy.a = texture2DRect (texture1, coord2).r;\n"
      "  gl_FragColor = uyvy;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_AYUV_U8_444, 3, NULL,
      "uniform sampler2DRect texture1;\n" /* Y4 */
      "uniform sampler2DRect texture2;\n" /* U4 */
      "uniform sampler2DRect texture3;\n" /* V4 */
      "void main (void) {\n"
      "  vec4 ayuv;\n"
      "  ayuv.r = 1.0;\n"
      "  ayuv.g = texture2DRect (texture1, gl_TexCoord[0]).r;\n"
      "  ayuv.b = texture2DRect (texture2, gl_TexCoord[0]).r;\n"
      "  ayuv.a = texture2DRect (texture3, gl_TexCoord[0]).r;\n"
      "  gl_FragColor = ayuv;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_ADD_S16_U8, 2, NULL, // FIXME: CPU overflows, GPU clamps, is this a problem?
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0]) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0]) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_ADD_S16_S16, 2, NULL, // FIXME: CPU overflows, GPU clamps, is this a problem?
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] = [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0]) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0]) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a + b) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_U8, 2, NULL, // FIXME: CPU overflows, GPU clamps, is this a problem?
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0]) + bias;\n"
      /* scale from U8 [0..255] == [0..1] to S16 [..0..255..] ~= [0..0.004]
         so that both inputs from S16 and U8 are mapped equivalent to FP and
         U8 zero == S16 zero == FP zero holds */
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0]) * scale;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] = [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16, 2, NULL, // FIXME: CPU overflows, GPU clamps, is this a problem?
      "uniform sampler2DRect texture1;\n" /* S16 */
      "uniform sampler2DRect texture2;\n" /* S16 */
      "const float bias = -32768.0 / 65535.0;\n"
      "void main (void) {\n"
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */
      "  vec4 a = texture2DRect (texture1, gl_TexCoord[0]) + bias;\n"
      "  vec4 b = texture2DRect (texture2, gl_TexCoord[0]) + bias;\n"
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */
      "  gl_FragColor = (a - b) - bias;\n"
      "}\n" },

  { -1, 0, NULL, NULL }
};

SchroOpenGLShader *
schro_opengl_shader_get (int index)
{
  int i;

  SCHRO_ASSERT (index >= SCHRO_OPENGL_SHADER_IDENTITY);
  SCHRO_ASSERT (index <= SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16);

  for (i = 0; schro_opengl_shader_list[i].code; ++i) {
    if (schro_opengl_shader_list[i].index == index) {
      if (!schro_opengl_shader_list[i].shader) {
          schro_opengl_shader_list[i].shader
              = schro_opengl_shader_new (schro_opengl_shader_list[i].code,
              schro_opengl_shader_list[i].textures);
      }

      return schro_opengl_shader_list[i].shader;
    }
  }

  SCHRO_ERROR ("no shader for index %i", index);
  SCHRO_ASSERT (0);

  return NULL;
}

