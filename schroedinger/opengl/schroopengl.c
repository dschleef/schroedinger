
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <GL/glew.h>
#include <GL/glxew.h>
#include <limits.h>

#define OPENGL_HANDLE_X_ERRORS

typedef struct _SchroOpenGL SchroOpenGL;

struct _SchroOpenGL {
  int usable;
  int visible;
  int lock_count;
  Display *display;
  Window root;
  int screen;
  XVisualInfo *visual_info;
  GLXContext context;
  Window window;
};

SchroOpenGL opengl;

#ifdef OPENGL_HANDLE_X_ERRORS
static int
schro_opengl_x_error_handler (Display *display, XErrorEvent *event)
{
  char errormsg[512];

  XGetErrorText(display, event->error_code, errormsg, sizeof(errormsg));
  SCHRO_ERROR("Xlib error: %s", errormsg);

  return 0;
}
#endif

static int
schro_opengl_open_display (const char *display_name)
{
  SCHRO_ASSERT (opengl.display == NULL);

  opengl.display = XOpenDisplay (display_name);

  if (opengl.display == NULL) {
    SCHRO_ERROR ("failed to open display %s", display_name);
    return FALSE;
  }

#ifdef OPENGL_HANDLE_X_ERRORS
  XSynchronize (opengl.display, True);
  XSetErrorHandler (schro_opengl_x_error_handler);
#endif

  opengl.root = DefaultRootWindow (opengl.display);
  opengl.screen = DefaultScreen (opengl.display);

  return TRUE;
}
/*
static void
schro_opengl_close_display (void)
{
  if (opengl.display) {
    XCloseDisplay (opengl.display);
  }
}*/

static int
schro_opengl_create_window (void)
{
  int error_base;
  int event_base;
  int ret;
  int visual_attr[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8,
    GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None
  };
  int mask;
  XSetWindowAttributes window_attr;

  SCHRO_ASSERT (opengl.display != NULL);
  SCHRO_ASSERT (opengl.root != None);

  ret = glXQueryExtension (opengl.display, &error_base, &event_base);

  if (!ret) {
    SCHRO_ERROR ("missing GLX extension");
    return FALSE;
  }

  opengl.visual_info = glXChooseVisual (opengl.display, opengl.screen,
      visual_attr);

  if (opengl.visual_info == NULL) {
    SCHRO_ERROR ("no usable visual");
    return FALSE;
  }

  opengl.context = glXCreateContext (opengl.display, opengl.visual_info,
      NULL, True);

  if (opengl.context == NULL) {
    SCHRO_ERROR ("failed to create direct GLX context");
    return FALSE;
  }

  mask = CWBackPixel | CWBorderPixel | CWColormap | CWOverrideRedirect;

  window_attr.background_pixel = 0;
  window_attr.border_pixel = 0;
  window_attr.colormap = XCreateColormap (opengl.display, opengl.root,
      opengl.visual_info->visual, AllocNone);
  window_attr.override_redirect = False;

  opengl.window = XCreateWindow (opengl.display, opengl.root, 0, 0,
      100, 100, 0, opengl.visual_info->depth, InputOutput,
      opengl.visual_info->visual, mask, &window_attr);

  if (opengl.window == None) {
    SCHRO_ERROR ("failed to create window with visual %ld",
        opengl.visual_info->visualid);

    glXDestroyContext (opengl.display, opengl.context);
    opengl.context = NULL;
    return FALSE;
  }

  XSync (opengl.display, FALSE);

  return TRUE;
}
/*
static void
schro_opengl_destroy_window (void)
{
  if (opengl.window != None) {
    XDestroyWindow (opengl.display, opengl.window);
    opengl.window = None;
  }

  if (opengl.context) {
    glXDestroyContext (opengl.display, opengl.context);
    opengl.context = NULL;
  }

  if (opengl.visual_info) {
    XFree (opengl.visual_info);
    opengl.visual_info = NULL;
  }
}*/

static int
schro_opengl_init_glew (void)
{
  int ok = TRUE;
  int major, minor, micro;
  GLenum error;

  schro_opengl_lock ();

  error = glewInit ();

  if (error != GLEW_OK) {
    SCHRO_ERROR("GLEW error: %s", glewGetErrorString (error));
    ok = FALSE;
  }

  major = atoi( (const char*) glewGetString (GLEW_VERSION_MAJOR));
  minor = atoi( (const char*) glewGetString (GLEW_VERSION_MINOR));
  micro = atoi( (const char*) glewGetString (GLEW_VERSION_MICRO));

  if (major < 1) {
    SCHRO_ERROR("missing GLEW >= 1.5.0");
    ok = FALSE;
  } else if (major == 1 && minor < 5) {
    SCHRO_ERROR("missing GLEW >= 1.5.0");
    ok = FALSE;
  } else if (major == 1 && minor == 5 && micro < 0) {
    SCHRO_ERROR("missing GLEW >= 1.5.0");
    ok = FALSE;
  }

  schro_opengl_unlock ();

  return ok;
}

static int
schro_opengl_check_essential_extensions (void)
{
  int ok = TRUE;

  schro_opengl_lock ();

  #define CHECK_EXTENSION(name) \
    if (!GLEW_##name) { \
      SCHRO_ERROR ("missing essential extension GL_" #name); \
      ok = FALSE; \
    }

  #define CHECK_EXTENSION_GROUPS(group1, group2, name) \
    if (!GLEW_##group1##_##name && !GLEW_##group2##_##name) { \
      SCHRO_ERROR ("missing essential extension GL_{" #group1 "|" #group2 "}_" #name); \
      ok = FALSE; \
    }

  CHECK_EXTENSION (EXT_framebuffer_object)
  CHECK_EXTENSION_GROUPS (ARB, NV, texture_rectangle)
  CHECK_EXTENSION (ARB_multitexture)
  CHECK_EXTENSION (ARB_shader_objects)
  CHECK_EXTENSION (ARB_shading_language_100)
  CHECK_EXTENSION (ARB_fragment_shader)

  #undef CHECK_EXTENSION
  #undef CHECK_EXTENSION_OR

  schro_opengl_unlock ();

  return ok;
}

void
schro_opengl_init (void)
{
  static int inited = FALSE;

  SCHRO_ASSERT(inited == FALSE);

  inited = TRUE;

  opengl.usable = TRUE;
  opengl.visible = FALSE;
  opengl.lock_count = 0;
  opengl.display = NULL;
  opengl.root = None;
  opengl.screen = 0;
  opengl.visual_info = NULL;
  opengl.context = NULL;
  opengl.window = None;

  if (!schro_opengl_open_display (NULL)) {
    opengl.usable = FALSE;
    return;
  }

  if (!schro_opengl_create_window ()) {
    opengl.usable = FALSE;
    return;
  }

  if (!schro_opengl_init_glew ()) {
    opengl.usable = FALSE;
    return;
  }

  if (!schro_opengl_check_essential_extensions ()) {
    opengl.usable = FALSE;
    return;
  }

  schro_opengl_frame_check_flags ();

  schro_opengl_lock ();

  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();

  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();

  glEnable (GL_TEXTURE_RECTANGLE_ARB);

  schro_opengl_unlock ();

  //schro_opengl_set_visible (TRUE);
}

int
schro_opengl_is_usable (void) {
  return opengl.usable;
}

void
schro_opengl_lock (void)
{
  SCHRO_ASSERT (opengl.display != NULL);
  SCHRO_ASSERT (opengl.window != None);
  SCHRO_ASSERT (opengl.context != NULL);
  SCHRO_ASSERT (opengl.lock_count < (INT_MAX - 1));

  if (opengl.lock_count == 0) {
    if (!glXMakeCurrent (opengl.display, opengl.window, opengl.context)) {
      SCHRO_ERROR ("glXMakeCurrent failed");
    }
  }

  ++opengl.lock_count;

  SCHRO_OPENGL_CHECK_ERROR
}

void
schro_opengl_unlock (void)
{
  SCHRO_ASSERT (opengl.display != NULL);
  SCHRO_ASSERT (opengl.lock_count > 0);

  SCHRO_OPENGL_CHECK_ERROR

  --opengl.lock_count;

  if (opengl.lock_count == 0) {
    if (!glXMakeCurrent (opengl.display, None, NULL)) {
      SCHRO_ERROR ("glXMakeCurrent failed");
    }
  }
}

void
schro_opengl_check_error (const char* file, int line)
{
  GLenum error = glGetError ();

  if (error) {
    SCHRO_ERROR ("GL Error 0x%x in %s(%d)", (int) error, file, line);
    //SCHRO_ASSERT (0);
  }
}

void
schro_opengl_check_framebuffer (const char* file, int line)
{
  switch (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)) {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT in %s(%d)",
          file, line);
      break;
    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
      SCHRO_ERROR ("GL_FRAMEBUFFER_UNSUPPORTED_EXT in %s(%d)", file, line);
      break;
    default:
      SCHRO_ERROR ("unknown error from glCheckFramebufferStatusEXT in %s(%d)",
          file, line);
      break;
  }
}

void
schro_opengl_set_visible (int visible)
{
  if (opengl.visible == visible) {
    return;
  }

  opengl.visible = visible;

  if (opengl.visible) {
    XMapWindow (opengl.display, opengl.window);
  } else {
    XUnmapWindow (opengl.display, opengl.window);
  }

  XSync (opengl.display, FALSE);
}

static void *
schro_opengl_alloc (int size)
{
  //SCHRO_DEBUG("domain is %d", schro_async_get_exec_domain ());
  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);

  return schro_malloc0 (size);
}

static void
schro_opengl_free (void *ptr, int size)
{
  //SCHRO_DEBUG("domain is %d", schro_async_get_exec_domain ());
  //SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);

  schro_free (ptr);
}

SchroMemoryDomain *
schro_memory_domain_new_opengl (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new();
  domain->flags = SCHRO_MEMORY_DOMAIN_OPENGL;
  domain->alloc = schro_opengl_alloc;
  domain->free = schro_opengl_free;

  return domain;
}

