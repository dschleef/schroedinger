
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

  size = strlen (code) + 1 + lines * strlen (number);
  linenumbered_code = schro_malloc0 (size);
  src = code;
  dst = linenumbered_code;

  strcpy (dst, "  1: ");

  dst += strlen ("  1: ");
  lines = 2;

  while (*src) {
    *dst++ = *src;

    if (*src == '\n') {
      snprintf (number, sizeof (number) - 1, "%3i: ", lines);
      strcpy (dst, number);

      dst += strlen (number);
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

  glGetObjectParameterivARB (handle, status, &result);
  glGetObjectParameterivARB (handle, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);

  if (length < 1) {
    SCHRO_ERROR ("invalid infolog length %i", length);
    return FALSE;
  }

  infolog = schro_malloc0 (length * sizeof (char));

  glGetInfoLogARB (handle, length, &length, infolog);

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

  #define UNIFORM_LOCATION(_type, _name, _member) \
      if (strstr (code, "uniform "#_type" "#_name";")) { \
        shader->_member = glGetUniformLocationARB (shader->program, #_name); \
      } else { \
        shader->_member = -1; \
      }

  UNIFORM_LOCATION (sampler2DRect, texture1, textures[0])
  UNIFORM_LOCATION (sampler2DRect, texture2, textures[1])
  UNIFORM_LOCATION (sampler2DRect, texture3, textures[2])
  UNIFORM_LOCATION (vec2, offset, offset)
  UNIFORM_LOCATION (vec2, one_decrease, one_decrease)
  UNIFORM_LOCATION (vec2, one_increase, one_increase)
  UNIFORM_LOCATION (vec2, two_decrease, two_decrease)
  UNIFORM_LOCATION (vec2, two_increase, two_increase)

  #undef UNIFORM_LOCATION

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

#define SHADER_HEADER \
    "#version 110\n" \
    "#extension GL_ARB_texture_rectangle : require\n"

static struct IndexToShader schro_opengl_shader_list[] = {
  { SCHRO_OPENGL_SHADER_IDENTITY,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n"
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect (texture1, gl_TexCoord[0].xy) - bias) / scale;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* U8 */
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main (void) {\n"
      "  gl_FragColor\n"
      "      = texture2DRect (texture1, gl_TexCoord[0].xy) * scale + bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* U8 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* S16 */
      "void main (void) {\n"
      "  gl_FragColor = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_YUYV,
      SHADER_HEADER
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
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_YUYV,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* YUYV */
      "void main (void) {\n"
      "  vec4 yuyv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (yuyv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_UYVY,
      SHADER_HEADER
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
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.r);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V2_UYVY,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* UYVY */
      "void main (void) {\n"
      "  vec4 uyvy = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (uyvy.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_Y4_AYUV,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.g);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U4_AYUV,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.b);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_V4_AYUV,
      SHADER_HEADER
      "uniform sampler2DRect texture1;\n" /* AYUV */
      "void main (void) {\n"
      "  vec4 ayuv = texture2DRect (texture1, gl_TexCoord[0].xy);\n"
      "  gl_FragColor = vec4 (ayuv.a);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_YUYV_U8_422,
      SHADER_HEADER
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
      SHADER_HEADER
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
      SHADER_HEADER
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
      SHADER_HEADER
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
      SHADER_HEADER
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
      SHADER_HEADER
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
      SHADER_HEADER
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

  #define SHADER_IIWT_S16_READ_WRITE_BIASED \
      "uniform sampler2DRect texture1;\n" \
      "const float bias = -32768.0 / 65535.0;\n" \
      "float read_biased (vec2 offset = vec2 (0.0)) {\n" \
      /* bias from [-32768..32767] == [0..1] to [-32768..32767] ~= [-0.5..0.5]
         so that S16 zero maps to FP zero, otherwise S16 zero maps to FP ~0.5
         leading to S16 zero <op> S16 zero != S16 zero if calculation is done in
         FP space */ \
      "  return texture2DRect (texture1, gl_TexCoord[0].xy + offset).r\n" \
      "      + bias;\n" \
      "}\n" \
      "void write_biased (float value) {\n" \
      /* bias from [-32768..32767] ~= [-0.5..0.5] to [-32768..32767] == [0..1]
         to undo the initial bias */ \
      "  gl_FragColor = vec4 (value - bias);\n" \
      "}\n"

  #define SHADER_IIWT_S16_SCALE_UP_DOWN \
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

     +---------------+                read for...
     |               |
     |       L       |                L'            H'
     |               |
     |             o | A[2 * n - 6]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n - 4]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n - 2]   - - - - ? ?   o - o - ? ?
     |          /> X | A[2 * n    ]   = = = = ? ?   X X X X ? ?
     |         /   o | A[2 * n + 2]   - - - - ? ?   o o o - ? ?
     |         |   o | A[2 * n + 4]   - - - - ? ?   o - o - ? ?
     |         |   o | A[2 * n + 6]   - - - - ? ?   - - - - ? ?
     |  offset |   o | A[2 * n + 8]   - - - - ? ?   - - - - ? ?
     |         |     |
     +---------|-----+                1 2 3 4 5 6   1 2 3 4 5 6
     |         |     |
     |         |   o | A[2 * n - 7]   - - - - ? ?   - - - - ? ?
     |         |   o | A[2 * n - 5]   - - - - ? ?   - - - - ? ?
     |         |   o | A[2 * n - 3]   - - o - ? ?   - - - - ? ?
     |         \   o | A[2 * n - 1]   o o o - ? ?   - - - - ? ?
     |          \> X | A[2 * n + 1]   X X X X ? ?   = = = = ? ?
     |             o | A[2 * n + 3]   - - o - ? ?   - - - - ? ?
     |             o | A[2 * n + 5]   - - - - ? ?   - - - - ? ?
     |             o | A[2 * n + 7]   - - - - ? ?   - - - - ? ?
     |               |
     |       H       |
     |               |
     +---------------+ */

  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Lp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "float filter (float h1m, float h0) {\n"
      "  float sh1m = scale_up (h1m);\n" /* A[2 ∗ n - 1] */
      "  float sh0 = scale_up (h0);\n"   /* A[2 ∗ n + 1] */
      "  float output = floor ((sh1m + sh0 + 2.0) / 4.0);\n"
      "  return scale_down (output);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_biased ();\n"                       /* A[2 ∗ n] */
      "  float h1m = read_biased (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_biased (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_biased (l0 - filter (h1m, h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_9_7_Hp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "float filter (float l1m, float l0, float l1p, float l2p) {\n"
#if 1
      "  float sl1m = scale_up (l1m);\n" /* A[2 ∗ n - 2] */
      "  float sl0 = scale_up (l0);\n"   /* A[2 ∗ n] */
      "  float sl1p = scale_up (l1p);\n" /* A[2 ∗ n + 2] */
      "  float sl2p = scale_up (l2p);\n" /* A[2 ∗ n + 4] */
      "  float output = floor ((-sl1m + 9.0 * (sl0 + sl1p) - sl2p + 8.0) / 16.0);\n"
      //"  float output = floor (-sl1m / 16.0 + (9.0 / 16.0) * sl0 + (9.0 / 16.0) * sl1p - sl2p / 16.0 + 8.0 / 16.0);\n"
      "  return scale_down (output);\n"
#else
      "  float output = floor (scale_up (-l1m + 9.0 * (l0 + l1p) - l2p + scale_down (8.0)) / 16.0);\n"
      "  return scale_down (output);\n"
#endif
      "}\n"
      "void main (void) {\n"
      "  float l1m = read_biased (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  float l0 = read_biased (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_biased (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float l2p = read_biased (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  float h0 = read_biased ();\n"                        /* A[2 ∗ n + 1] */
      "  write_biased (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Lp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "float filter (float h1m, float h0) {\n"
      "  float sh1m = scale_up (h1m);\n" /* A[2 ∗ n - 1] */
      "  float sh0 = scale_up (h0);\n"   /* A[2 ∗ n + 1] */
      "  float output = floor ((sh1m + sh0 + 2.0) / 4.0);\n"
      "  return scale_down (output);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_biased ();\n"                       /* A[2 ∗ n] */
      "  float h1m = read_biased (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_biased (offset);\n"                 /* A[2 ∗ n + 1] */
      "  write_biased (l0 - filter (h1m, h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_LE_GALL_5_3_Hp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_increase;\n"
      "float filter (float l0, float l1p) {\n"
      "  float sl0 = scale_up (l0);\n"   /* A[2 ∗ n] */
      "  float sl1p = scale_up (l1p);\n" /* A[2 ∗ n + 2] */
      "  float output = floor ((sl0 + sl1p + 1.0) / 2.0);\n"
      "  return scale_down (output);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_biased (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_biased (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float h0 = read_biased ();\n"                        /* A[2 ∗ n + 1] */
      "  write_biased (h0 + filter (l0, l1p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Lp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_decrease;\n"
      "float filter (float h2m, float h1m, float h0, float h1p) {\n"
#if 1
      "  float sh2m = scale_up (h2m);\n" /* A[2 ∗ n - 3] */
      "  float sh1m = scale_up (h1m);\n" /* A[2 ∗ n - 1] */
      "  float sh0 = scale_up (h0);\n"   /* A[2 ∗ n + 1] */
      "  float sh1p = scale_up (h1p);\n" /* A[2 ∗ n + 3] */
      "  float output = floor ((-sh2m + 9.0 * (sh1m + sh0) - sh1p + 16.0) / 32.0);\n"
      //"  float output = floor (-sh2m / 32.0 + (9.0 / 32.0) * sh1m + (9.0 / 32.0) * sh0 - sh1p / 32.0 + 16.0 / 32.0);\n"
      "  return scale_down (output);\n"
#else
      "  float output = floor (scale_up (-h2m + 9.0 * (h1m + h0) - h1p + scale_down (16.0)) / 32.0);\n"
      "  return scale_down (output);\n"
#endif
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_biased ();\n"                       /* A[2 ∗ n] */
      "  float h2m = read_biased (offset - two_decrease);\n" /* A[2 ∗ n - 3] */
      "  float h1m = read_biased (offset - one_decrease);\n" /* A[2 ∗ n - 1] */
      "  float h0 = read_biased (offset);\n"                 /* A[2 ∗ n + 1] */
      "  float h1p = read_biased (offset + one_increase);\n" /* A[2 ∗ n + 3] */
      "  write_biased (l0 - filter (h2m, h1m, h0, h1p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_DESLAURIERS_DUBUC_13_7_Hp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "uniform vec2 one_decrease;\n"
      "uniform vec2 one_increase;\n"
      "uniform vec2 two_increase;\n"
      "float filter (float l1m, float l0, float l1p, float l2p) {\n"
#if 1
      "  float sl1m = scale_up (l1m);\n" /* A[2 ∗ n - 2] */
      "  float sl0 = scale_up (l0);\n"   /* A[2 ∗ n] */
      "  float sl1p = scale_up (l1p);\n" /* A[2 ∗ n + 2] */
      "  float sl2p = scale_up (l2p);\n" /* A[2 ∗ n + 4] */
      "  float output = floor ((-sl1m + 9.0 * (sl0 + sl1p) - sl2p + 8.0) / 16.0);\n"
      //"  float output = floor (-sl1m / 16.0 + (9.0 / 16.0) * sl0 + (9.0 / 16.0) * sl1p - sl2p / 16.0 + 8.0 / 16.0);\n"
      "  return scale_down (output);\n"
#else
      "  float output = floor (scale_up (-l1m + 9.0 * (l0 + l1p) - l2p + scale_down (8.0)) / 16.0);\n"
      "  return scale_down (output);\n"
#endif
      "}\n"
      "void main (void) {\n"
      "  float l1m = read_biased (-offset - one_decrease);\n" /* A[2 ∗ n - 2] */
      "  float l0 = read_biased (-offset);\n"                 /* A[2 ∗ n] */
      "  float l1p = read_biased (-offset + one_increase);\n" /* A[2 ∗ n + 2] */
      "  float l2p = read_biased (-offset + two_increase);\n" /* A[2 ∗ n + 4] */
      "  float h0 = read_biased ();\n"                        /* A[2 ∗ n + 1] */
      "  write_biased (h0 + filter (l1m, l0, l1p, l2p));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Lp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      /* distance between two corresponding texels from subbands L and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "float filter (float h0) {\n"
      "  float sh0 = scale_up (h0);\n"       /* A[2 ∗ n + 1] */
      "  float output = floor ((sh0 + 1.0) / 2.0);\n"
      "  return scale_down (output);\n"
      "}\n"
      "void main (void) {\n"
      "  float l0 = read_biased ();\n"       /* A[2 ∗ n] */
      "  float h0 = read_biased (offset);\n" /* A[2 ∗ n + 1] */
      "  write_biased (l0 - filter (h0));\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_IIWT_S16_FILTER_HAAR_Hp,
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      /* distance between two corresponding texels from subbands L' and H in
         texels = vec2 (width / 2.0, 0.0) or vec2 (0.0, height / 2.0) */
      "uniform vec2 offset;\n"
      "void main (void) {\n"
      "  float l0 = read_biased (-offset);\n" /* A[2 ∗ n] */
      "  float h0 = read_biased ();\n"        /* A[2 ∗ n + 1] */
      "  write_biased (h0 + l0);\n"
      "}\n" },

  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_L,
      SHADER_HEADER
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
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_DEINTERLEAVE_H,
      SHADER_HEADER
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
  { SCHRO_OPENGL_SHADER_IIWT_S16_VERTICAL_INTERLEAVE,
      SHADER_HEADER
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
  { SCHRO_OPENGL_SHADER_IIWT_S16_HORIZONTAL_INTERLEAVE,
      SHADER_HEADER
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
      SHADER_HEADER
      SHADER_IIWT_S16_READ_WRITE_BIASED
      SHADER_IIWT_S16_SCALE_UP_DOWN
      "float rshift (float value) {\n"
      "  float input = scale_up (value);\n"
      "  float output = floor ((input + 1.0) / 2.0);\n"
      "  return scale_down (output);\n"
      "}\n"
      "void main (void) {\n"
      "  float value = read_biased ();\n"
      "  write_biased (rshift (value));\n"
      "}\n" },

  #undef SHADER_IIWT_S16_READ_WRITE_BIASED
  #undef SHADER_IIWT_S16_SCALE_UP_DOWN

  { -1, NULL }
};

#undef SHADER_HEADER

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
  SCHRO_ASSERT (index < SCHRO_OPENGL_SHADER_COUNT);

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

