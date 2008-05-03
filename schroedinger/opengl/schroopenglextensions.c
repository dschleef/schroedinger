
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <string.h>

extern __GLXextFuncPtr glXGetProcAddressARB (const GLubyte *);

/* GL_EXT_framebuffer_object */
PFNGLISRENDERBUFFEREXTPROC glIsRenderbufferEXT = NULL;
PFNGLBINDRENDERBUFFEREXTPROC glBindRenderbufferEXT = NULL;
PFNGLDELETERENDERBUFFERSEXTPROC glDeleteRenderbuffersEXT = NULL;
PFNGLGENRENDERBUFFERSEXTPROC glGenRenderbuffersEXT = NULL;
PFNGLRENDERBUFFERSTORAGEEXTPROC glRenderbufferStorageEXT = NULL;
PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC glGetRenderbufferParameterivEXT = NULL;
PFNGLISFRAMEBUFFEREXTPROC glIsFramebufferEXT = NULL;
PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebufferEXT = NULL;
PFNGLDELETEFRAMEBUFFERSEXTPROC glDeleteFramebuffersEXT = NULL;
PFNGLGENFRAMEBUFFERSEXTPROC glGenFramebuffersEXT = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC glCheckFramebufferStatusEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE1DEXTPROC glFramebufferTexture1DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE2DEXTPROC glFramebufferTexture2DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE3DEXTPROC glFramebufferTexture3DEXT = NULL;
PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC glFramebufferRenderbufferEXT = NULL;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC
    glGetFramebufferAttachmentParameterivEXT = NULL;
PFNGLGENERATEMIPMAPEXTPROC glGenerateMipmapEXT = NULL;

/* GL_ARB_shader_objects */
PFNGLDELETEOBJECTARBPROC glDeleteObjectARB = NULL;
PFNGLGETHANDLEARBPROC glGetHandleARB = NULL;
PFNGLDETACHOBJECTARBPROC glDetachObjectARB = NULL;
PFNGLCREATESHADEROBJECTARBPROC glCreateShaderObjectARB = NULL;
PFNGLSHADERSOURCEARBPROC glShaderSourceARB = NULL;
PFNGLCOMPILESHADERARBPROC glCompileShaderARB = NULL;
PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB = NULL;
PFNGLATTACHOBJECTARBPROC glAttachObjectARB = NULL;
PFNGLLINKPROGRAMARBPROC glLinkProgramARB = NULL;
PFNGLUSEPROGRAMOBJECTARBPROC glUseProgramObjectARB = NULL;
PFNGLVALIDATEPROGRAMARBPROC glValidateProgramARB = NULL;
PFNGLUNIFORM1FARBPROC glUniform1fARB = NULL;
PFNGLUNIFORM2FARBPROC glUniform2fARB = NULL;
PFNGLUNIFORM3FARBPROC glUniform3fARB = NULL;
PFNGLUNIFORM4FARBPROC glUniform4fARB = NULL;
PFNGLUNIFORM1IARBPROC glUniform1iARB = NULL;
PFNGLUNIFORM2IARBPROC glUniform2iARB = NULL;
PFNGLUNIFORM3IARBPROC glUniform3iARB = NULL;
PFNGLUNIFORM4IARBPROC glUniform4iARB = NULL;
PFNGLUNIFORM1FVARBPROC glUniform1fvARB = NULL;
PFNGLUNIFORM2FVARBPROC glUniform2fvARB = NULL;
PFNGLUNIFORM3FVARBPROC glUniform3fvARB = NULL;
PFNGLUNIFORM4FVARBPROC glUniform4fvARB = NULL;
PFNGLUNIFORM1IVARBPROC glUniform1ivARB = NULL;
PFNGLUNIFORM2IVARBPROC glUniform2ivARB = NULL;
PFNGLUNIFORM3IVARBPROC glUniform3ivARB = NULL;
PFNGLUNIFORM4IVARBPROC glUniform4ivARB = NULL;
PFNGLUNIFORMMATRIX2FVARBPROC glUniformMatrix2fvARB = NULL;
PFNGLUNIFORMMATRIX3FVARBPROC glUniformMatrix3fvARB = NULL;
PFNGLUNIFORMMATRIX4FVARBPROC glUniformMatrix4fvARB = NULL;
PFNGLGETOBJECTPARAMETERFVARBPROC glGetObjectParameterfvARB = NULL;
PFNGLGETOBJECTPARAMETERIVARBPROC glGetObjectParameterivARB = NULL;
PFNGLGETINFOLOGARBPROC glGetInfoLogARB = NULL;
PFNGLGETATTACHEDOBJECTSARBPROC glGetAttachedObjectsARB = NULL;
PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocationARB = NULL;
PFNGLGETACTIVEUNIFORMARBPROC glGetActiveUniformARB = NULL;
PFNGLGETUNIFORMFVARBPROC glGetUniformfvARB = NULL;
PFNGLGETUNIFORMIVARBPROC glGetUniformivARB = NULL;
PFNGLGETSHADERSOURCEARBPROC glGetShaderSourceARB = NULL;

/* GL_ARB_vertex_buffer_object */
PFNGLBINDBUFFERARBPROC glBindBufferARB = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffersARB = NULL;
PFNGLGENBUFFERSARBPROC glGenBuffersARB = NULL;
PFNGLISBUFFERARBPROC glIsBufferARB = NULL;
PFNGLBUFFERDATAARBPROC glBufferDataARB = NULL;
PFNGLBUFFERSUBDATAARBPROC glBufferSubDataARB = NULL;
PFNGLGETBUFFERSUBDATAARBPROC glGetBufferSubDataARB = NULL;
PFNGLMAPBUFFERARBPROC glMapBufferARB = NULL;
PFNGLUNMAPBUFFERARBPROC glUnmapBufferARB = NULL;
PFNGLGETBUFFERPARAMETERIVARBPROC glGetBufferParameterivARB = NULL;
PFNGLGETBUFFERPOINTERVARBPROC glGetBufferPointervARB = NULL;

#define GET_PROC_ADDRESS(type, name) \
    ok = (((name) = (type)glXGetProcAddress ((const GLubyte *) #name)) != NULL) && ok;

unsigned int _schro_opengl_extensions = 0;

int
schro_opengl_load_extensions (void)
{
  const char *extensions;
  int ok;

  schro_opengl_lock ();

  extensions = (const char *) glGetString (GL_EXTENSIONS);

  /* check GL_EXT_framebuffer_object */
  if (!strstr (extensions, "GL_EXT_framebuffer_object")) {
    SCHRO_ERROR ("no framebuffer extension");
    return FALSE;
  }

  /* resolve GL_EXT_framebuffer_object */
  ok = TRUE;

  GET_PROC_ADDRESS (PFNGLISRENDERBUFFEREXTPROC, glIsRenderbufferEXT);
  GET_PROC_ADDRESS (PFNGLBINDRENDERBUFFEREXTPROC, glBindRenderbufferEXT);
  GET_PROC_ADDRESS (PFNGLDELETERENDERBUFFERSEXTPROC, glDeleteRenderbuffersEXT);
  GET_PROC_ADDRESS (PFNGLGENRENDERBUFFERSEXTPROC, glGenRenderbuffersEXT);
  GET_PROC_ADDRESS (PFNGLRENDERBUFFERSTORAGEEXTPROC, glRenderbufferStorageEXT);
  GET_PROC_ADDRESS (PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC,
      glGetRenderbufferParameterivEXT);
  GET_PROC_ADDRESS (PFNGLISFRAMEBUFFEREXTPROC, glIsFramebufferEXT);
  GET_PROC_ADDRESS (PFNGLBINDFRAMEBUFFEREXTPROC, glBindFramebufferEXT);
  GET_PROC_ADDRESS (PFNGLDELETEFRAMEBUFFERSEXTPROC, glDeleteFramebuffersEXT);
  GET_PROC_ADDRESS (PFNGLGENFRAMEBUFFERSEXTPROC, glGenFramebuffersEXT);
  GET_PROC_ADDRESS (PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC,
      glCheckFramebufferStatusEXT);
  GET_PROC_ADDRESS (PFNGLFRAMEBUFFERTEXTURE1DEXTPROC,
      glFramebufferTexture1DEXT);
  GET_PROC_ADDRESS (PFNGLFRAMEBUFFERTEXTURE2DEXTPROC,
      glFramebufferTexture2DEXT);
  GET_PROC_ADDRESS (PFNGLFRAMEBUFFERTEXTURE3DEXTPROC,
      glFramebufferTexture3DEXT);
  GET_PROC_ADDRESS (PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC,
      glFramebufferRenderbufferEXT);
  GET_PROC_ADDRESS (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC,
      glGetFramebufferAttachmentParameterivEXT);
  GET_PROC_ADDRESS (PFNGLGENERATEMIPMAPEXTPROC, glGenerateMipmapEXT);

  if (!ok) {
    SCHRO_ERROR ("reported framebuffer extension fails to resolve");
    return FALSE;
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_FRAMEBUFFER;
  }

  /* check GL_ARB_shader_objects */
  if (!strstr (extensions, "GL_ARB_shader_objects")) {
    SCHRO_ERROR ("no shader objects extension");
    return FALSE;
  }

  /* check GL_ARB_shading_language_100 */
  if (!strstr (extensions, "GL_ARB_shading_language_100")) {
    SCHRO_ERROR ("no shading language extension");
    return FALSE;
  }

  /* check GL_ARB_fragment_shader */
  if (!strstr (extensions, "GL_ARB_fragment_shader")) {
    SCHRO_ERROR ("no fragment shader extension");
    return FALSE;
  }

  /* resolve GL_ARB_shader_objects */
  ok = TRUE;
  
  GET_PROC_ADDRESS (PFNGLDELETEOBJECTARBPROC, glDeleteObjectARB);
  GET_PROC_ADDRESS (PFNGLGETHANDLEARBPROC, glGetHandleARB);
  GET_PROC_ADDRESS (PFNGLDETACHOBJECTARBPROC, glDetachObjectARB);
  GET_PROC_ADDRESS (PFNGLCREATESHADEROBJECTARBPROC, glCreateShaderObjectARB);
  GET_PROC_ADDRESS (PFNGLSHADERSOURCEARBPROC, glShaderSourceARB);
  GET_PROC_ADDRESS (PFNGLCOMPILESHADERARBPROC, glCompileShaderARB);
  GET_PROC_ADDRESS (PFNGLCREATEPROGRAMOBJECTARBPROC, glCreateProgramObjectARB);
  GET_PROC_ADDRESS (PFNGLATTACHOBJECTARBPROC, glAttachObjectARB);
  GET_PROC_ADDRESS (PFNGLLINKPROGRAMARBPROC, glLinkProgramARB);
  GET_PROC_ADDRESS (PFNGLUSEPROGRAMOBJECTARBPROC, glUseProgramObjectARB);
  GET_PROC_ADDRESS (PFNGLVALIDATEPROGRAMARBPROC, glValidateProgramARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM1FARBPROC, glUniform1fARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM2FARBPROC, glUniform2fARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM3FARBPROC, glUniform3fARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM4FARBPROC, glUniform4fARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM1IARBPROC, glUniform1iARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM2IARBPROC, glUniform2iARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM3IARBPROC, glUniform3iARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM4IARBPROC, glUniform4iARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM1FVARBPROC, glUniform1fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM2FVARBPROC, glUniform2fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM3FVARBPROC, glUniform3fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM4FVARBPROC, glUniform4fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM1IVARBPROC, glUniform1ivARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM2IVARBPROC, glUniform2ivARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM3IVARBPROC, glUniform3ivARB);
  GET_PROC_ADDRESS (PFNGLUNIFORM4IVARBPROC, glUniform4ivARB);
  GET_PROC_ADDRESS (PFNGLUNIFORMMATRIX2FVARBPROC, glUniformMatrix2fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORMMATRIX3FVARBPROC, glUniformMatrix3fvARB);
  GET_PROC_ADDRESS (PFNGLUNIFORMMATRIX4FVARBPROC, glUniformMatrix4fvARB);
  GET_PROC_ADDRESS (PFNGLGETOBJECTPARAMETERFVARBPROC,
      glGetObjectParameterfvARB);
  GET_PROC_ADDRESS (PFNGLGETOBJECTPARAMETERIVARBPROC,
      glGetObjectParameterivARB);
  GET_PROC_ADDRESS (PFNGLGETINFOLOGARBPROC, glGetInfoLogARB);
  GET_PROC_ADDRESS (PFNGLGETATTACHEDOBJECTSARBPROC, glGetAttachedObjectsARB);
  GET_PROC_ADDRESS (PFNGLGETUNIFORMLOCATIONARBPROC, glGetUniformLocationARB);
  GET_PROC_ADDRESS (PFNGLGETACTIVEUNIFORMARBPROC, glGetActiveUniformARB);
  GET_PROC_ADDRESS (PFNGLGETUNIFORMFVARBPROC, glGetUniformfvARB);
  GET_PROC_ADDRESS (PFNGLGETUNIFORMIVARBPROC, glGetUniformivARB);
  GET_PROC_ADDRESS (PFNGLGETSHADERSOURCEARBPROC, glGetShaderSourceARB);

  if (!ok) {
    SCHRO_ERROR ("reported shader objects extension fails to resolve");
    return FALSE;
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_FRAGMENT_SHADER;
  }

  /* check GL_ARB_texture_rectangle || GL_NV_texture_rectangle */
  if (!strstr (extensions, "GL_ARB_texture_rectangle")
      && !strstr (extensions, "GL_NV_texture_rectangle")) {
    SCHRO_ERROR ("no texture rectangle extension");
    return FALSE;
  }

  /* check GL_ARB_vertex_buffer_object */
  if (!strstr (extensions, "GL_ARB_vertex_buffer_object")) {
    SCHRO_ERROR ("no vertexbuffer extension");
  } else {
    /* resolve GL_ARB_vertex_buffer_object */
    ok = TRUE;

    GET_PROC_ADDRESS (PFNGLBINDBUFFERARBPROC, glBindBufferARB);
    GET_PROC_ADDRESS (PFNGLDELETEBUFFERSARBPROC, glDeleteBuffersARB);
    GET_PROC_ADDRESS (PFNGLGENBUFFERSARBPROC, glGenBuffersARB);
    GET_PROC_ADDRESS (PFNGLISBUFFERARBPROC, glIsBufferARB);
    GET_PROC_ADDRESS (PFNGLBUFFERDATAARBPROC, glBufferDataARB);
    GET_PROC_ADDRESS (PFNGLBUFFERSUBDATAARBPROC, glBufferSubDataARB);
    GET_PROC_ADDRESS (PFNGLGETBUFFERSUBDATAARBPROC, glGetBufferSubDataARB);
    GET_PROC_ADDRESS (PFNGLMAPBUFFERARBPROC, glMapBufferARB);
    GET_PROC_ADDRESS (PFNGLUNMAPBUFFERARBPROC, glUnmapBufferARB);
    GET_PROC_ADDRESS (PFNGLGETBUFFERPARAMETERIVARBPROC,
        glGetBufferParameterivARB);
    GET_PROC_ADDRESS (PFNGLGETBUFFERPOINTERVARBPROC, glGetBufferPointervARB);

    if (!ok) {
      SCHRO_ERROR ("reported vertex buffer extension fails to resolve");
    } else {
      /* check GL_ARB_pixel_buffer_object */
      if (!strstr (extensions, "GL_ARB_pixel_buffer_object")) {
        SCHRO_ERROR ("no pixelbuffer extension");
      } else {
        _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_PIXELBUFFER;
      }
    }
  }

  /* check GL_EXT_bgra */
  if (!strstr (extensions, "GL_EXT_bgra")) {
    SCHRO_INFO ("no RGBA extension");
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_BGRA;
  }

  /* check GL_EXT_texture_integer */
  if (!strstr (extensions, "GL_EXT_texture_integer")) {
    SCHRO_INFO ("no texture integer extension");
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_TEXTURE_INTEGER;
  }

  /* check GL_ARB_texture_float || GL_ATI_texture_float*/
  if (!strstr (extensions, "GL_ARB_texture_float") 
      && !strstr (extensions, "GL_ATI_texture_float")) {
    SCHRO_INFO ("no texture float extension");
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_TEXTURE_FLOAT;
  }

  /* check GL_NV_float_buffer */
  if (!strstr (extensions, "GL_NV_float_buffer")) {
    SCHRO_INFO ("no NVIDIA floatbuffer extension");
  } else {
    _schro_opengl_extensions |= SCHRO_OPENGL_EXTENSION_NVIDIA_FLOAT_BUFFER;
  }

  schro_opengl_unlock ();

  return TRUE;
}

