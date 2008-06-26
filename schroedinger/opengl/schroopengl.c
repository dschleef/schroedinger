
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <limits.h>
#include <GL/glew.h>
#include <GL/glxew.h>
#ifdef HAVE_PTHREAD
#include <pthread.h>

typedef pthread_mutex_t SchroMutex;
#define schro_mutex_init_recursive(_mutex) \
    do { \
      pthread_mutexattr_t mutexattr; \
      pthread_mutexattr_init (&mutexattr); \
      pthread_mutexattr_settype (&mutexattr, PTHREAD_MUTEX_RECURSIVE); \
      pthread_mutex_init (&(_mutex), &mutexattr); \
      pthread_mutexattr_destroy (&mutexattr); \
    } while (0)
#define schro_mutex_destroy(_mutex) pthread_mutex_destroy (&(_mutex))
#define schro_mutex_lock(_mutex) pthread_mutex_lock (&(_mutex))
#define schro_mutex_unlock(_mutex) pthread_mutex_unlock (&(_mutex))
#elif defined HAVE_GTHREAD
#error add code for a gthread recursive mutex here
#define schro_mutex_init_recursive(_mutex) do {} while (0)
#define schro_mutex_destroy(_mutex) do {} while (0)
#define schro_mutex_lock(_mutex) do {} while (0)
#define schro_mutex_unlock(_mutex) do {} while (0)
//#elif defined WIN32
// FIXME: add code for a critical section here
#else
#define schro_mutex_init_recursive(_mutex) do {} while (0)
#define schro_mutex_destroy(_mutex) do {} while (0)
#define schro_mutex_lock(_mutex) do {} while (0)
#define schro_mutex_unlock(_mutex) do {} while (0)
#endif

struct _SchroOpenGL {
  int is_usable;
  int is_visible;
  int lock_count;
  SchroMutex mutex;
  Display *display;
  Window root;
  int screen;
  XVisualInfo *visual_info;
  GLXContext context;
  Window window;
  SchroOpenGLShaderLibrary *library;
  void *tmp;
  int tmp_size;
};

static int
schro_opengl_x_error_handler (Display *display, XErrorEvent *event)
{
  char errormsg[512];

  XGetErrorText (display, event->error_code, errormsg, sizeof (errormsg));
  SCHRO_ERROR ("Xlib error: %s", errormsg);

  return 0;
}

static int
schro_opengl_open_display (SchroOpenGL *opengl, const char *display_name)
{
  SCHRO_ASSERT (opengl->display == NULL);

  opengl->display = XOpenDisplay (display_name);

  if (opengl->display == NULL) {
    SCHRO_ERROR ("failed to open display %s", display_name);
    return FALSE;
  }

  XSynchronize (opengl->display, False);
  XSetErrorHandler (schro_opengl_x_error_handler);

  opengl->root = DefaultRootWindow (opengl->display);
  opengl->screen = DefaultScreen (opengl->display);

  return TRUE;
}

static void
schro_opengl_close_display (SchroOpenGL *opengl)
{
  if (opengl->display) {
    XCloseDisplay (opengl->display);
    opengl->display = NULL;
  }
}

static int
schro_opengl_create_window (SchroOpenGL *opengl)
{
  int error_base;
  int event_base;
  int result;
  int visual_attr[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8,
      GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
  int mask;
  XSetWindowAttributes window_attr;

  SCHRO_ASSERT (opengl->display != NULL);
  SCHRO_ASSERT (opengl->root != None);

  result = glXQueryExtension (opengl->display, &error_base, &event_base);

  if (!result) {
    SCHRO_ERROR ("missing GLX extension");
    return FALSE;
  }

  opengl->visual_info = glXChooseVisual (opengl->display, opengl->screen,
      visual_attr);

  if (opengl->visual_info == NULL) {
    SCHRO_ERROR ("no usable visual");
    return FALSE;
  }

  opengl->context = glXCreateContext (opengl->display, opengl->visual_info,
      NULL, True);

  if (opengl->context == NULL) {
    SCHRO_ERROR ("failed to create direct GLX context");

    XFree (opengl->visual_info);
    opengl->visual_info = NULL;

    return FALSE;
  }

  mask = CWBackPixel | CWBorderPixel | CWColormap | CWOverrideRedirect;

  window_attr.background_pixel = 0;
  window_attr.border_pixel = 0;
  window_attr.colormap = XCreateColormap (opengl->display, opengl->root,
      opengl->visual_info->visual, AllocNone);
  window_attr.override_redirect = False;

  opengl->window = XCreateWindow (opengl->display, opengl->root, 0, 0,
      100, 100, 0, opengl->visual_info->depth, InputOutput,
      opengl->visual_info->visual, mask, &window_attr);

  if (opengl->window == None) {
    SCHRO_ERROR ("failed to create window with visual %ld",
        opengl->visual_info->visualid);

    glXDestroyContext (opengl->display, opengl->context);
    opengl->context = NULL;

    XFree (opengl->visual_info);
    opengl->visual_info = NULL;

    return FALSE;
  }

  XSync (opengl->display, FALSE);

  return TRUE;
}

static void
schro_opengl_destroy_window (SchroOpenGL *opengl)
{
  if (opengl->window != None) {
    XDestroyWindow (opengl->display, opengl->window);
    opengl->window = None;
  }

  if (opengl->context) {
    glXDestroyContext (opengl->display, opengl->context);
    opengl->context = NULL;
  }

  if (opengl->visual_info) {
    XFree (opengl->visual_info);
    opengl->visual_info = NULL;
  }
}

static int
schro_opengl_init_glew (SchroOpenGL *opengl)
{
  int ok = TRUE;
  int major, minor, micro;
  GLenum error;

  schro_opengl_lock (opengl);

  error = glewInit ();

  if (error != GLEW_OK) {
    SCHRO_ERROR ("GLEW error: %s", glewGetErrorString (error));
    ok = FALSE;
  }

  major = atoi ((const char*) glewGetString (GLEW_VERSION_MAJOR));
  minor = atoi ((const char*) glewGetString (GLEW_VERSION_MINOR));
  micro = atoi ((const char*) glewGetString (GLEW_VERSION_MICRO));

  if (major < 1) {
    SCHRO_ERROR ("missing GLEW >= 1.5.0");
    ok = FALSE;
  } else if (major == 1 && minor < 5) {
    SCHRO_ERROR ("missing GLEW >= 1.5.0");
    ok = FALSE;
  } else if (major == 1 && minor == 5 && micro < 0) {
    SCHRO_ERROR ("missing GLEW >= 1.5.0");
    ok = FALSE;
  }

  schro_opengl_unlock (opengl);

  return ok;
}

static int
schro_opengl_check_essential_extensions (SchroOpenGL *opengl)
{
  int ok = TRUE;

  schro_opengl_lock (opengl);

  #define CHECK_EXTENSION(_name) \
    if (!GLEW_##_name) { \
      SCHRO_ERROR ("missing essential extension GL_" #_name); \
      ok = FALSE; \
    }

  #define CHECK_EXTENSION_GROUPS(_group1, _group2, _name) \
    if (!GLEW_##_group1##_##_name && !GLEW_##_group2##_##_name) { \
      SCHRO_ERROR ("missing essential extension GL_{" #_group1 "|" #_group2 "}_" #_name); \
      ok = FALSE; \
    }

  CHECK_EXTENSION (EXT_framebuffer_object)
  CHECK_EXTENSION_GROUPS (ARB, NV, texture_rectangle)
  CHECK_EXTENSION (ARB_multitexture)
  CHECK_EXTENSION (ARB_shader_objects)
  CHECK_EXTENSION (ARB_shading_language_100)
  CHECK_EXTENSION (ARB_fragment_shader)

  #undef CHECK_EXTENSION
  #undef CHECK_EXTENSION_GROUPS

  schro_opengl_unlock (opengl);

  return ok;
}

void schro_opengl_init (void)
{
  XInitThreads ();
}

SchroOpenGL *
schro_opengl_new (void)
{
  SchroOpenGL *opengl = schro_malloc0 (sizeof (SchroOpenGL));

  opengl->is_usable = TRUE;
  opengl->is_visible = FALSE;
  opengl->lock_count = 0;
  opengl->display = NULL;
  opengl->root = None;
  opengl->screen = 0;
  opengl->visual_info = NULL;
  opengl->context = NULL;
  opengl->window = None;
  opengl->library = NULL;
  opengl->tmp = NULL;
  opengl->tmp_size = 0;

  schro_mutex_init_recursive (opengl->mutex);

  if (!schro_opengl_open_display (opengl, NULL)) {
    opengl->is_usable = FALSE;
    return opengl;
  }

  if (!schro_opengl_create_window (opengl)) {
    opengl->is_usable = FALSE;
    return opengl;
  }

  if (!schro_opengl_init_glew (opengl)) {
    opengl->is_usable = FALSE;
    return opengl;
  }

  if (!schro_opengl_check_essential_extensions (opengl)) {
    opengl->is_usable = FALSE;
    return opengl;
  }

  opengl->library = schro_opengl_shader_library_new (opengl);

  schro_opengl_frame_check_flags ();

  schro_opengl_lock (opengl);

  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();

  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();

  glEnable (GL_TEXTURE_RECTANGLE_ARB);

  schro_opengl_unlock (opengl);

  //schro_opengl_set_visible (opengl, TRUE);

  return opengl;
}

void
schro_opengl_free (SchroOpenGL *opengl)
{
  SCHRO_ASSERT (opengl->lock_count == 0);

  if (opengl->library) {
    schro_opengl_shader_library_free (opengl->library);
  }

  SCHRO_ASSERT (opengl->lock_count == 0);

  schro_opengl_destroy_window (opengl);
  schro_opengl_close_display (opengl);

  if (opengl->tmp) {
    schro_free (opengl->tmp);
  }

  schro_mutex_destroy (opengl->mutex);

  schro_free (opengl);
}

int
schro_opengl_is_usable (SchroOpenGL *opengl) {
  return opengl->is_usable;
}

void
schro_opengl_lock (SchroOpenGL *opengl)
{
  SCHRO_ASSERT (opengl->display != NULL);
  SCHRO_ASSERT (opengl->window != None);
  SCHRO_ASSERT (opengl->context != NULL);
  SCHRO_ASSERT (opengl->lock_count < (INT_MAX - 1));
  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);

  schro_mutex_lock (opengl->mutex);

  if (opengl->lock_count == 0) {
    XLockDisplay (opengl->display);

    if (!glXMakeCurrent (opengl->display, opengl->window, opengl->context)) {
      SCHRO_ERROR ("glXMakeCurrent failed");
    }

    XUnlockDisplay (opengl->display);
  }

  ++opengl->lock_count;

  SCHRO_OPENGL_CHECK_ERROR
}

void
schro_opengl_unlock (SchroOpenGL *opengl)
{
  SCHRO_ASSERT (opengl->display != NULL);
  SCHRO_ASSERT (opengl->lock_count > 0);

  SCHRO_OPENGL_CHECK_ERROR

  --opengl->lock_count;

  if (opengl->lock_count == 0) {
    XLockDisplay (opengl->display);

    if (!glXMakeCurrent (opengl->display, None, NULL)) {
      SCHRO_ERROR ("glXMakeCurrent failed");
    }

    XUnlockDisplay (opengl->display);
  }

  schro_mutex_unlock (opengl->mutex);
}

void
schro_opengl_check_error (const char* file, int line, const char* func)
{
  GLenum error = glGetError ();

  if (error) {
    SCHRO_ERROR ("GL Error 0x%x in %s(%d) %s", (int) error, file, line, func);
    //SCHRO_ASSERT (0);
  }
}

void
schro_opengl_check_framebuffer (const char *file, int line, const char *func)
{
  switch (glCheckFramebufferStatusEXT (GL_FRAMEBUFFER_EXT)) {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT in %s(%d) %s",
          file, line, func);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT in "
          "%s(%d) %s", file, line, func);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT in %s(%d) %s",
          file, line, func);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT in %s(%d) %s",
          file, line, func);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT in %s(%d) %s",
          file, line, func);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT in %s(%d) %s",
          file, line, func);
      break;
    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_UNSUPPORTED_EXT in %s(%d)", file, line,
         func);
      break;
    default:
      SCHRO_ERROR ("unknown error from glCheckFramebufferStatusEXT in "
          "%s(%d) %s", file, line, func);
      break;
  }
}

void
schro_opengl_set_visible (SchroOpenGL *opengl, int visible)
{
  if (opengl->is_visible == visible) {
    return;
  }

  opengl->is_visible = visible;

  if (opengl->is_visible) {
    XMapWindow (opengl->display, opengl->window);
  } else {
    XUnmapWindow (opengl->display, opengl->window);
  }

  XSync (opengl->display, FALSE);
}


SchroOpenGLShaderLibrary *
schro_opengl_get_library (SchroOpenGL *opengl)
{
  return opengl->library;
}

void *
schro_opengl_get_tmp (SchroOpenGL *opengl, int size)
{
  SCHRO_ASSERT (size > 0);

  if (opengl->tmp_size < size || !opengl->tmp) {
    opengl->tmp_size = size;

    if (!opengl->tmp) {
      opengl->tmp = schro_malloc (opengl->tmp_size);
    } else {
      opengl->tmp = schro_realloc (opengl->tmp, opengl->tmp_size);
    }
  }

  return opengl->tmp;
}

void
schro_opengl_setup_viewport (int width, int height)
{
  glViewport (0, 0, width, height);

  glLoadIdentity ();
  glOrtho (0, width, 0, height, -1, 1);
}

void
schro_opengl_render_quad (int x, int y, int width, int height)
{
  glBegin (GL_QUADS);
  glTexCoord2f (x,         y);          glVertex3f (x,         y,          0);
  glTexCoord2f (x + width, y);          glVertex3f (x + width, y,          0);
  glTexCoord2f (x + width, y + height); glVertex3f (x + width, y + height, 0);
  glTexCoord2f (x,         y + height); glVertex3f (x,         y + height, 0);
  glEnd ();
}

static void *
schro_opengl_domain_alloc (int size)
{
  //SCHRO_DEBUG ("domain is %d", schro_async_get_exec_domain ());
  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);

  return schro_malloc0 (size);
}

static void
schro_opengl_domain_free (void *ptr, int size)
{
  //SCHRO_DEBUG ("domain is %d", schro_async_get_exec_domain ());
  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);

  schro_free (ptr);
}

SchroMemoryDomain *
schro_memory_domain_new_opengl (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new ();
  domain->flags = SCHRO_MEMORY_DOMAIN_OPENGL;
  domain->alloc = schro_opengl_domain_alloc;
  domain->free = schro_opengl_domain_free;

  return domain;
}

