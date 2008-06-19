
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/opengl/schroopenglshader.h>
#include <stdio.h>
#include <string.h>

static char*
schro_opengl_shader_add_linenumbers (const char* code)
{
  const char *src = code;
  char *dst;
  char *linenumbered_code;
  char number[16];
  int lines = 1;
  int size;

  while (*src) {
    if (*src == '\n') {
      ++lines;
    }

    ++src;
  }

  snprintf (number, sizeof (number) - 1, "%3i: ", lines);

  size = strlen (code) + 1 + lines * strlen(number);
  linenumbered_code = schro_malloc0 (size);
  src = code;
  dst = linenumbered_code;

  strcpy (dst, "  1: ");

  dst += strlen("  1: ");
  lines = 2;

  while (*src) {
    *dst++ = *src;

    if (*src == '\n') {
      snprintf (number, sizeof (number) - 1, "%3i: ", lines);
      strcpy (dst, number);

      dst += strlen(number);
      ++lines;
    }

    ++src;
  }

  return linenumbered_code;
}

static int
schro_opengl_shader_check_status (GLhandleARB handle, GLenum status,
    const char* code)
{
  GLint result;
  GLint length;
  char* infolog;
  char* linenumbered_code;

  glGetObjectParameterivARB(handle, status, &result);
  glGetObjectParameterivARB(handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);

  if (length < 1) {
    SCHRO_ERROR ("invalid infolog length %i", length);
    return FALSE;
  }

  infolog = schro_malloc0(length * sizeof (char));

  glGetInfoLogARB(handle, length, &length, infolog);

  if (length > 0) {
    linenumbered_code = schro_opengl_shader_add_linenumbers (code);

    SCHRO_ERROR ("\nshadercode:\n%s\ninfolog:\n%s", linenumbered_code,
        infolog);

    schro_free (linenumbered_code);
  }

  schro_free (infolog);

  return result != 0;
}

static SchroOpenGLShader *
schro_opengl_shader_new (const char* code)
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

  if (strstr (code, "uniform sampler2DRect texture1;")) {
      shader->textures[0] = glGetUniformLocationARB (shader->program,
          "texture1");
  }

  if (strstr (code, "uniform sampler2DRect texture2;")) {
      shader->textures[1] = glGetUniformLocationARB (shader->program,
          "texture2");
  }

  if (strstr (code, "uniform sampler2DRect texture3;")) {
      shader->textures[2] = glGetUniformLocationARB (shader->program,
          "texture3");
  }

  if (strstr (code, "uniform vec2 offset;")) {
      shader->offset = glGetUniformLocationARB (shader->program, "offset");
  }

  if (strstr (code, "uniform vec2 one;")) {
      shader->one = glGetUniformLocationARB (shader->program, "one");
  }

  return shader;
}

static void
schro_opengl_shader_free (SchroOpenGLShader *shader)
{
  SCHRO_ASSERT (shader != NULL);

  glDeleteObjectARB (shader->program);

  schro_free (shader);
}

struct IndexToShader {
  int index;
  const char *code;
};

static struct IndexToShader schro_opengl_shader_list[] = {
  { SCHRO_OPENGL_SHADER_IDENTITY,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n"
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect (texture1, gl_TexCoord[0].xy) - bias) / scale;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = texture2DRect (texture1, gl_TexCoord[0].xy) * scale + bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 yuyv = texture2DRect (texture1, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = vec4 (yuyv.r);\n"
      "  } else {\n"
      "    gl_FragColor = vec4 (yuyv.b);\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_YUYV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      /* round x coordinate down from texel center n.5 to texel edge n.0 */
      "  float x = floor (gl_TexCoord[0].x) / 2.0;\n"
      "  float y = gl_TexCoord[0].y;\n"
      /* shift x coordinate from texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (floor (x) + 0.5, y);\n"
      "  vec4 uyvy = texture2DRect (texture1, coord);\n"
      "  if (fract (x) < 0.25) {\n"
      "    gl_FragColor = vec4 (uyvy.g);\n"
      "  } else {\n"
      "    gl_FragColor = vec4 (uyvy.a);\n"
      "  }\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U2_UYVY,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.r);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422,
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
  { SCHRO_OPENGL_SHADER_CONVERT_UYVY_U8_422,
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
  { SCHRO_OPENGL_SHADER_CONVERT_AYUV_U8_444,
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
  { SCHRO_OPENGL_SHADER_ADD_S16_U8,
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
  { SCHRO_OPENGL_SHADER_ADD_S16_S16,
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
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_U8,
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
  { SCHRO_OPENGL_SHADER_SUBTRACT_S16_S16,
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
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_XL,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n"
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y);\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (x, floor (y * 2.0) + 0.5);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture1, coord);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_XH,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n"
      /* height of subband XL */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "void main (void) {\n"
      "  float x = gl_TexCoord[0].x;\n"
      /* round y coordinate down from texel center n.5 to texel edge n.0 */
      "  float y = floor (gl_TexCoord[0].y) - offset.y;\n"
      /* scale y coordinate to the destination coordinate and shift it from
         texel edge n.0 to texel center n.5 */
      "  vec2 coord = vec2 (x, floor (y * 2.0) + 1.5);\n" /* FIXME: floor */
      "  gl_FragColor = texture2DRect (texture1, coord);\n"
      "}\n" },

  #define IIWT_S16_READ_WRITE_BIASED \
      "uniform sampler2DRect texture1;\n" \
      "const float bias = -32768.0 / 65535.0;\n" \
      "float read_biased (vec2 offset = vec2 (0.0)) {\n" \
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero - S16 zero != S16 zero if calculation is done in
         FP space */ \
      "  return texture2DRect (texture1, gl_TexCoord[0].xy + offset).r\n" \
      "      + bias;\n" \
      "}\n" \
      "void write_biased (float value) {\n" \
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */ \
      "  gl_FragColor = vec4 (value - bias);\n" \
      "}\n"

  #define IIWT_S16_SCALE_UP_DOWN \
      "float scale_up (float value) {\n" \
      /* scale from FP [-0.5..0.5] to real S16 [-32768..32767] and apply
         proper rounding */ \
      "  return floor (value * 65535.0 + 0.5);\n" \
      "}\n" \
      "float scale_down (float value) {\n" \
      /* scale from real S16 [-32768..32767] to FP [-0.5..0.5] */ \
      "  return value / 65535.0;\n" \
      "}\n"

  /* 1 = Deslauriers-Debuc (9,7)
     2 = LeGall (5,3)
     3 = Deslauriers-Debuc (13,7)
     4 = Haar 0/1
     5 = Fidelity
     6 = Daubechies (9,7)

     offset = height / 2

     +----------------+
     |                |
     |     L/even     |
     |                |
     |              o | A[2 * n - 6]    - - - - o -
     |              o | A[2 * n - 4]    - - - - o -
     |              o | A[2 * n - 2]    o - o - o -
     |           /> X | A[2 * n    ] >  X X X X X X
     |          /   o | A[2 * n + 2]    o o o - o o
     |          |   o | A[2 * n + 4]    o - o - o -
     |          |   o | A[2 * n + 6]    - - - - o -
     |   offset |   o | A[2 * n + 8]    - - - - o -
     |          |     |
     +----------|-----+                 1 2 3 4 5 6
     |          |     |
     |          |   o | A[2 * n - 7]    - - - - o -
     |          |   o | A[2 * n - 5]    - - - - o -
     |          |   o | A[2 * n - 3]    - - o - o -
     |          \   o | A[2 * n - 1]    o o o - o o
     |           \> X | A[2 * n + 1] >  X X X X X X
     |              o | A[2 * n + 3]    - - o - o -
     |              o | A[2 * n + 5]    - - - - o -
     |              o | A[2 * n + 7]    - - - - o -
     |                |
     |     H/odd      |
     |                |
     +----------------+ */

  #define IIWT_S16_FILTER_HAAR_STEP1 \
      IIWT_S16_SCALE_UP_DOWN \
      "float filter (float value) {\n" \
      "  float input = scale_up (value);\n" /* A[2 ∗ n + 1] */ \
      "  float output = floor ((input + 1.0) / 2.0);\n" \
      "  return scale_down (output);\n" \
      "}\n"

  #define IIWT_S16_FILTER_HAAR_STEP2 \
      "float filter (float value) {\n" \
      "  return value;\n" /* A[2 ∗ n] */ \
      "}\n"

  #define IIWT_S16_FILTER_LE_GALL_5_3_STEP1 \
      IIWT_S16_SCALE_UP_DOWN \
      "float filter (float value1, float value2) {\n" \
      "  float input1 = scale_up (value1);\n" /* A[2 ∗ n - 1] */ \
      "  float input2 = scale_up (value2);\n" /* A[2 ∗ n + 1] */ \
      "  float output = floor ((input1 + input2 + 2.0) / 4.0);\n" \
      "  return scale_down (output);\n" \
      "}\n"

  #define IIWT_S16_FILTER_LE_GALL_5_3_STEP2 \
      IIWT_S16_SCALE_UP_DOWN \
      "float filter (float value1, float value2) {\n" \
      "  float input1 = scale_up (value1);\n" /* A[2 ∗ n] */ \
      "  float input2 = scale_up (value2);\n" /* A[2 ∗ n + 2] */ \
      "  float output = floor ((input1 + input2 + 1.0) / 2.0);\n" \
      "  return scale_down (output);\n" \
      "}\n"

  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_FILTER_LE_GALL_5_3_XLp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "uniform vec2 one;\n"
      IIWT_S16_FILTER_LE_GALL_5_3_STEP1
      "void main (void) {\n"
      "  float xl = read_biased ();\n" /* A[2 ∗ n] */
      "  float xh1 = read_biased (offset - one);\n" /* A[2 ∗ n - 1] */
      "  float xh2 = read_biased (offset);\n" /* A[2 ∗ n + 1] */
      "  write_biased (xl - filter (xh1, xh2));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_FILTER_LE_GALL_5_3_XHp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* vertical distance between two corresponding texels from subband XL'
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      "uniform vec2 one;\n"
      IIWT_S16_FILTER_LE_GALL_5_3_STEP2
      "void main (void) {\n"
      "  float xlp1 = read_biased (-offset);\n" /* A[2 ∗ n] */
      "  float xlp2 = read_biased (-offset + one);\n" /* A[2 ∗ n + 2] */
      "  float xh = read_biased ();\n" /* A[2 ∗ n + 1] */
      "  write_biased (filter (xlp1, xlp2) + xh);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_FILTER_HAAR_XLp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* vertical distance between two corresponding texels from subband XL
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      IIWT_S16_FILTER_HAAR_STEP1
      "void main (void) {\n"
      "  float xl = read_biased ();\n" /* A[2 ∗ n] */
      "  float xh = read_biased (offset);\n" /* A[2 ∗ n + 1] */
      "  write_biased (xl - filter (xh));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_FILTER_HAAR_XHp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* vertical distance between two corresponding texels from subband XL'
         and XH in texels */
      "uniform vec2 offset;\n" /* = vec2 (0.0, height / 2.0) */
      IIWT_S16_FILTER_HAAR_STEP2
      "void main (void) {\n"
      "  float xlp = read_biased (-offset);\n" /* A[2 ∗ n] */
      "  float xh = read_biased ();\n" /* A[2 ∗ n + 1] */
      "  write_biased (filter (xlp) + xh);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_INTERLEAVE,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n"
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
      "  gl_FragColor = texture2DRect (texture1, coord);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_FILTER_LE_GALL_5_3_Lp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* horizontal distance between two corresponding texels from subband L
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "uniform vec2 one;\n"
      IIWT_S16_FILTER_LE_GALL_5_3_STEP1
      "void main (void) {\n"
      "  float lx = read_biased ();\n" /* A[2 ∗ n] */
      "  float hx1 = read_biased (offset - one);\n" /* A[2 ∗ n - 1] */
      "  float hx2 = read_biased (offset);\n" /* A[2 ∗ n + 1] */
      "  write_biased (lx - filter (hx1, hx2));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_FILTER_LE_GALL_5_3_Hp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* horizontal distance between two corresponding texels from subband L'
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      "uniform vec2 one;\n"
      IIWT_S16_FILTER_LE_GALL_5_3_STEP2
      "void main (void) {\n"
      "  float lxp1 = read_biased (-offset);\n" /* A[2 ∗ n] */
      "  float lxp2 = read_biased (-offset + one);\n" /* A[2 ∗ n + 2] */
      "  float hx = read_biased ();\n" /* A[2 ∗ n + 1] */
      "  write_biased (filter (lxp1, lxp2) + hx);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_FILTER_HAAR_Lp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* horizontal distance between two corresponding texels from subband L
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      IIWT_S16_FILTER_HAAR_STEP1
      "void main (void) {\n"
      "  float lx = read_biased ();\n" /* A[2 ∗ n] */
      "  float hx = read_biased (offset);\n" /* A[2 ∗ n + 1] */
      "  write_biased (lx - filter (hx));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_FILTER_HAAR_Hp,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      /* horizontal distance between two corresponding texels from subband L'
         and H in texels */
      "uniform vec2 offset;\n" /* = vec2 (width / 2.0, 0.0) */
      IIWT_S16_FILTER_HAAR_STEP2
      "void main (void) {\n"
      "  float lxp = read_biased (-offset);\n" /* A[2 ∗ n] */
      "  float hx = read_biased ();\n" /* A[2 ∗ n + 1] */
      "  write_biased (filter (lxp) + hx);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_INTERLEAVE,
      "#extension GL_ARB_texture_rectangle : enable\n"
      "uniform sampler2DRect texture1;\n"
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
      "  gl_FragColor = texture2DRect (texture1, coord);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_SHIFT,
      "#extension GL_ARB_texture_rectangle : enable\n"
      IIWT_S16_READ_WRITE_BIASED
      "float rshift (float value) {\n" \
      /* scale from FP [-0.5..0.5] to real S16 [-32768..32767] and apply \
         proper rounding */ \
      "  float input = floor (value * 65535.0 + 0.5);\n" \
      /* add 1 and right shift by 1 */ \
      "  float output = floor ((input + 1.0) / 2.0);\n" \
      /* scale from real S16 [-32768..32767] to FP [-0.5..0.5] */ \
      "  return output / 65535.0;\n" \
      "}\n"
      "void main (void) {\n"
      "  float value = read_biased ();\n"
      "  write_biased (rshift (value));\n"
      "}\n" },

  #undef IIWT_S16_READ_WRITE_BIASED
  #undef IIWT_S16_SCALE_UP_DOWN
  #undef IIWT_S16_FILTER_HAAR_STEP1
  #undef IIWT_S16_FILTER_HAAR_STEP2
  #undef IIWT_S16_FILTER_LE_GALL_5_3_STEP1
  #undef IIWT_S16_FILTER_LE_GALL_5_3_STEP2

  { -1, NULL }
};

struct _SchroOpenGLShaderLibrary {
  SchroOpenGL *opengl;
  SchroOpenGLShader *shaders[SCHRO_OPENGL_SHADER_COUNT];
};

SchroOpenGLShaderLibrary *
schro_opengl_shader_library_new (SchroOpenGL *opengl)
{
  SchroOpenGLShaderLibrary* library
      = schro_malloc0 (sizeof (SchroOpenGLShaderLibrary));

  library->opengl = opengl;

  return library;
}

void
schro_opengl_shader_library_free (SchroOpenGLShaderLibrary *library)
{
  int i;

  SCHRO_ASSERT (library != NULL);

  schro_opengl_lock (library->opengl);

  for (i = 0; i < ARRAY_SIZE (library->shaders); ++i) {
    if (library->shaders[i]) {
      schro_opengl_shader_free (library->shaders[i]);
    }
  }

  schro_opengl_unlock (library->opengl);

  schro_free (library);
}

SchroOpenGLShader *
schro_opengl_shader_get (SchroOpenGL *opengl, int index)
{
  int i;
  SchroOpenGLShaderLibrary* library;

  SCHRO_ASSERT (index >= 0);
  SCHRO_ASSERT (index <= SCHRO_OPENGL_SHADER_COUNT);

  library = schro_opengl_get_library (opengl);

  for (i = 0; schro_opengl_shader_list[i].code; ++i) {
    if (schro_opengl_shader_list[i].index == index) {
      if (!library->shaders[index]) {
          schro_opengl_lock (opengl);

          library->shaders[index]
              = schro_opengl_shader_new (schro_opengl_shader_list[i].code);

          schro_opengl_unlock (opengl);
      }

      return library->shaders[index];
    }
  }

  SCHRO_ERROR ("no shader for index %i", index);
  SCHRO_ASSERT (0);

  return NULL;
}

