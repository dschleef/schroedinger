
#include <schroedinger/schro.h>

#include <GL/gl.h>
#include <GL/glx.h>


Display *display;

void
schro_opengl_init (void)
{

  display = XOpenDisplay (NULL);
  if (display == NULL) {
    SCHRO_ERROR("failed to open display");
    return;
  }



}

static void *
schro_opengl_alloc (int size)
{
  return NULL;
}

static void *
schro_opengl_alloc_2d (int format, int width, int height)
{

  return NULL;
}

static void
schro_opengl_free (void *ptr, int size)
{

}

SchroMemoryDomain *
schro_memory_domain_new_opengl (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new();
  domain->flags = SCHRO_MEMORY_DOMAIN_OPENGL;
  domain->alloc = schro_opengl_alloc;
  domain->alloc_2d = schro_opengl_alloc_2d;
  domain->free = schro_opengl_free;

  return domain;
}

