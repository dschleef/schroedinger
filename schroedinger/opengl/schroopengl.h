
#ifndef __SCHRO_OPENGL_H__
#define __SCHRO_OPENGL_H__

SCHRO_BEGIN_DECLS

#define SCHRO_OPENGL_CHECK_ERROR \
    schro_opengl_check_error (__FILE__, __LINE__);

#define SCHRO_OPENGL_CHECK_FRAMEBUFFER \
    schro_opengl_check_framebuffer (__FILE__, __LINE__);

void schro_opengl_init (void);
int schro_opengl_is_usable (void);
void schro_opengl_lock (void);
void schro_opengl_unlock (void);
void schro_opengl_check_error (const char* file, int line);
void schro_opengl_check_framebuffer (const char* file, int line);
void schro_opengl_set_visible (int visible);

SchroMemoryDomain *schro_memory_domain_new_opengl (void);

SCHRO_END_DECLS

#endif

