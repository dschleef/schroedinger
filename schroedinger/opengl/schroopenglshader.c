
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglshader.h>

SchroOpenGLShader *
schro_opengl_shader_new (const char* code)
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

  shader->texture = glGetUniformLocationARB (shader->program, "texture");

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
  SchroOpenGLShader *shader;
  const char *code;
};

static struct IndexToShader index_to_shader_list[] = {
  { SCHRO_OPENGL_SHADER_IDENTITY, NULL,
      "uniform sampler2DRect texture;\n"
      "void main() {\n"
      "  gl_FragColor = texture2DRect(texture, gl_TexCoord[0]);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_S16, NULL,
      "uniform sampler2DRect texture;\n"
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main() {\n"
      "  gl_FragColor\n"
      "      = (texture2DRect(texture, gl_TexCoord[0]) - bias) / scale;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_U8, NULL,
      "uniform sampler2DRect texture;\n"
      "const float scale = 255.0 / 65535.0;\n"
      "const float bias = (32767.0 - 127.0) / 65535.0;\n"
      "void main() {\n"
      "  gl_FragColor\n"
      "      = texture2DRect(texture, gl_TexCoord[0]) * scale + bias;\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_U8_U8, NULL,
      "uniform sampler2DRect texture;\n"
      "void main() {\n"
      "  gl_FragColor = texture2DRect(texture, gl_TexCoord[0]);\n"
      "}\n" },
  { SCHRO_OPENGL_SHADER_CONVERT_S16_S16, NULL,
      "uniform sampler2DRect texture;\n"
      "void main() {\n"
      "  gl_FragColor = texture2DRect(texture, gl_TexCoord[0]);\n"
      "}\n" },

  { -1, NULL, NULL }
};

SchroOpenGLShader *
schro_opengl_shader_get (int index)
{
  int i;

  SCHRO_ASSERT (index >= SCHRO_OPENGL_SHADER_CONVERT_U8_S16);
  SCHRO_ASSERT (index <= SCHRO_OPENGL_SHADER_CONVERT_S16_S16);

  for (i = 0; index_to_shader_list[i].code; ++i) {
    if (index_to_shader_list[i].index == index) {
      if (!index_to_shader_list[i].shader) {
          index_to_shader_list[i].shader
              = schro_opengl_shader_new (index_to_shader_list[i].code);
      }

      return index_to_shader_list[i].shader;
    }
  }

  SCHRO_ERROR ("no shader for index %i", index);
  SCHRO_ASSERT (0);

  return NULL;
}

