
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglshader.h>

GLhandleARB
schro_opengl_shader_new (const char* code)
{
  GLhandleARB program;
  GLhandleARB shader;
  GLint status;

  shader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

  glShaderSourceARB (shader, 1, (const char**)&code, 0);
  glCompileShaderARB (shader);
  glGetObjectParameterivARB (shader, GL_OBJECT_COMPILE_STATUS_ARB, &status);

  SCHRO_ASSERT (status != 0);

  program = glCreateProgramObjectARB ();

  glAttachObjectARB (program, shader);
  glDeleteObjectARB (shader);
  glLinkProgramARB (program);
  glGetObjectParameterivARB (program, GL_OBJECT_LINK_STATUS_ARB, &status);

  SCHRO_ASSERT (status != 0);

  glValidateProgramARB (program);
  glGetObjectParameterivARB (program, GL_OBJECT_VALIDATE_STATUS_ARB, &status);

  SCHRO_ASSERT (status != 0);

  return program;
}

void
schro_opengl_shader_free (GLhandleARB program)
{
  glDeleteObjectARB (program);
}
