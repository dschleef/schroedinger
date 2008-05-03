
#ifndef __SCHRO_OPENGL_SHADER_H__
#define __SCHRO_OPENGL_SHADER_H__

#include <schroedinger/schro.h>
#include <GL/gl.h>

SCHRO_BEGIN_DECLS

GLhandleARB schro_opengl_shader_new (const char* code);
void schro_opengl_shader_free (GLhandleARB program);

SCHRO_END_DECLS

#endif

