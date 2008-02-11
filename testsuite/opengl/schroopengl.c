
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include "schroopengl.h"
#include "glextensions.h"

#include <string.h>

static void schro_opengl_init_tmp_window (SchroOpenGL * display);

static int schro_opengl_check_features (SchroOpenGL * display);



void
schro_opengl_free (SchroOpenGL *display)
{
  if (display->window != None) {
    XDestroyWindow (display->display, display->window);
  }
  if (display->context) {
    glXDestroyContext (display->display, display->context);
  }
  //if (display->visinfo) {
  //  XFree (display->visinfo);
  //}
  if (display->display) {
    XCloseDisplay (display->display);
  }

#if 0
  if (display->lock) {
    g_mutex_free (display->lock);
  }
#endif
}


SchroOpenGL *
schro_opengl_new (void)
{
  return schro_malloc0 (sizeof(SchroOpenGL));
}

//#define HANDLE_X_ERRORS
#ifdef HANDLE_X_ERRORS
static int
x_error_handler (Display * display, XErrorEvent * event)
{
  //g_assert_not_reached ();
}
#endif

int
schro_opengl_connect (SchroOpenGL * display, const char *display_name)
{
  int usable;
  XGCValues values;
  XPixmapFormatValues *px_formats;
  int n_formats;
  int i;

  display->display = XOpenDisplay (display_name);
  if (display->display == NULL) {
    return FALSE;
  }
#ifdef HANDLE_X_ERRORS
  XSynchronize (display->display, True);
  XSetErrorHandler (x_error_handler);
#endif

  usable = schro_opengl_check_features (display);
  if (!usable) {
    return FALSE;
  }

  display->screen = DefaultScreenOfDisplay (display->display);
  display->screen_num = DefaultScreen (display->display);
  display->visual = DefaultVisual (display->display, display->screen_num);
  display->root = DefaultRootWindow (display->display);
  display->white = XWhitePixel (display->display, display->screen_num);
  display->black = XBlackPixel (display->display, display->screen_num);
  display->depth = DefaultDepthOfScreen (display->screen);

  display->gc = XCreateGC (display->display,
      DefaultRootWindow (display->display), 0, &values);

  px_formats = XListPixmapFormats (display->display, &n_formats);
  for (i = 0; i < n_formats; i++) {
    SCHRO_ERROR ("%d: depth %d bpp %d pad %d", i,
        px_formats[i].depth,
        px_formats[i].bits_per_pixel, px_formats[i].scanline_pad);
  }

  schro_opengl_init_tmp_window (display);

  return TRUE;
}

static int
schro_opengl_check_features (SchroOpenGL * display)
{
  int ret;
  XVisualInfo *visinfo;
  Screen *screen;
  Window root;
  int scrnum;
  int attrib[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8,
    GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None
  };
  XSetWindowAttributes attr;
  int error_base;
  int event_base;
  int mask;
  const char *extstring;
  Window window;

  screen = XDefaultScreenOfDisplay (display->display);
  scrnum = XScreenNumberOfScreen (screen);
  root = XRootWindow (display->display, scrnum);

  ret = glXQueryExtension (display->display, &error_base, &event_base);
  if (!ret) {
    SCHRO_ERROR ("No GLX extension");
    return FALSE;
  }

  visinfo = glXChooseVisual (display->display, scrnum, attrib);
  if (visinfo == NULL) {
    SCHRO_ERROR ("No usable visual");
    return FALSE;
  }

  display->visinfo = visinfo;

  display->context = glXCreateContext (display->display, visinfo, NULL, True);

  attr.background_pixel = 0;
  attr.border_pixel = 0;
  attr.colormap = XCreateColormap (display->display, root,
      visinfo->visual, AllocNone);
  attr.event_mask = StructureNotifyMask | ExposureMask;
  attr.override_redirect = True;

  mask = CWBackPixel | CWBorderPixel | CWColormap | CWOverrideRedirect;

  SCHRO_ERROR ("creating window with visual %ld", visinfo->visualid);

  window = XCreateWindow (display->display, root, 0, 0,
      100, 100, 0, visinfo->depth, InputOutput, visinfo->visual, mask, &attr);

  XSync (display->display, FALSE);

  glXMakeCurrent (display->display, window, display->context);

  glGetIntegerv (GL_MAX_TEXTURE_SIZE, &display->max_texture_size);

  extstring = (const char *) glGetString (GL_EXTENSIONS);

  display->have_ycbcr_texture = FALSE;
#ifdef GL_YCBCR_MESA
  if (strstr (extstring, "GL_MESA_ycbcr_texture")) {
    display->have_ycbcr_texture = TRUE;
  }
#endif

  display->have_color_matrix = FALSE;
#ifdef GL_POST_COLOR_MATRIX_RED_BIAS
  if (strstr (extstring, "GL_SGI_color_matrix")) {
    display->have_color_matrix = TRUE;
  }
#endif

  display->have_texture_rectangle = FALSE;
#ifdef GL_TEXTURE_RECTANGLE_ARB
  if (strstr (extstring, "GL_ARB_texture_rectangle")) {
    display->have_texture_rectangle = TRUE;
  }
#endif

  glXMakeCurrent (display->display, None, NULL);
  XDestroyWindow (display->display, window);

  return TRUE;
}

void
schro_opengl_lock (SchroOpenGL * display)
{
  int ret;

  SCHRO_ASSERT (display->window != None);
  SCHRO_ASSERT (display->context != NULL);

  //g_mutex_lock (display->lock);
  ret = glXMakeCurrent (display->display, display->window, display->context);
  if (!ret) {
    SCHRO_ERROR ("glxMakeCurrent failed");
  }
  schro_opengl_check_error (display, __LINE__);
}

void
schro_opengl_unlock (SchroOpenGL * display)
{
  schro_opengl_check_error (display, __LINE__);
  glXMakeCurrent (display->display, None, NULL);
  //g_mutex_unlock (display->lock);
}

static void
schro_opengl_init_tmp_window (SchroOpenGL * display)
{
  XSetWindowAttributes attr = { 0 };
  int scrnum;
  int mask;
  Window root;
  Window parent_window;
  Screen *screen;
  int width;
  int height;

  SCHRO_ERROR ("creating temp window");

  screen = XDefaultScreenOfDisplay (display->display);
  scrnum = XScreenNumberOfScreen (screen);
  root = XRootWindow (display->display, scrnum);

  attr.background_pixel = 0;
  attr.border_pixel = 0;
  attr.colormap = XCreateColormap (display->display, root,
      display->visinfo->visual, AllocNone);
  if (display->parent_window != None) {
    XWindowAttributes parent_attr;

    attr.override_redirect = True;
    parent_window = display->parent_window;

    XGetWindowAttributes (display->display, parent_window, &parent_attr);
    width = parent_attr.width;
    height = parent_attr.height;
  } else {
    attr.override_redirect = False;
    parent_window = root;
    width = 100;
    height = 100;
  }

  mask = CWBackPixel | CWBorderPixel | CWColormap | CWOverrideRedirect;

  display->window = XCreateWindow (display->display,
      parent_window, 0, 0, width, height,
      0, display->visinfo->depth, InputOutput,
      display->visinfo->visual, mask, &attr);
  if (display->visible) {
    XMapWindow (display->display, display->window);
  }
  XSync (display->display, FALSE);
}

static void
schro_opengl_destroy_tmp_window (SchroOpenGL * display)
{
  XDestroyWindow (display->display, display->window);
}

void
schro_opengl_set_visible (SchroOpenGL * display, int visible)
{
  if (display->visible == visible)
    return;
  display->visible = visible;
  if (display->visible) {
    XMapWindow (display->display, display->window);
  } else {
    XUnmapWindow (display->display, display->window);
  }
  XSync (display->display, FALSE);
}

void
schro_opengl_set_window (SchroOpenGL * display, Window window)
{
  //g_mutex_lock (display->lock);

  if (display->display == NULL) {
    display->parent_window = window;
  } else {
    if (window != display->parent_window) {
      XSync (display->display, False);

      schro_opengl_destroy_tmp_window (display);

      display->parent_window = window;

      schro_opengl_init_tmp_window (display);
    }
  }

  //g_mutex_unlock (display->lock);
}

void
schro_opengl_update_window (SchroOpenGL * display)
{
  XWindowAttributes attr;

  //g_return_if_fail (display != NULL);

  //g_mutex_lock (display->lock);
  if (display->window != None && display->parent_window != None) {
    XSync (display->display, False);
    XGetWindowAttributes (display->display, display->parent_window, &attr);

    SCHRO_ERROR ("new size %d %d", attr.width, attr.height);

    if (display->win_width != attr.width || display->win_height != attr.height) {
      XResizeWindow (display->display, display->window,
          attr.width, attr.height);
      //XSync (display->display, False);
    }
    display->win_width = attr.width;
    display->win_height = attr.height;
  }
  //g_mutex_unlock (display->lock);
}

void
schro_opengl_update_attributes (SchroOpenGL * display)
{
  XWindowAttributes attr;

  if (display->window != None) {
    XGetWindowAttributes (display->display, display->window, &attr);

    SCHRO_ERROR ("window visual %ld display visual %ld",
        attr.visual->visualid, display->visinfo->visual->visualid);

    display->win_width = attr.width;
    display->win_height = attr.height;
  } else {
    display->win_width = 0;
    display->win_height = 0;
  }
}

void
schro_opengl_set_window_size (SchroOpenGL * display, int width, int height)
{
  if (display->win_width != width || display->win_height != height) {
    display->win_width = width;
    display->win_height = height;
    XResizeWindow (display->display, display->window, width, height);
    XSync (display->display, False);
  }
}

void
schro_opengl_clear (SchroOpenGL * display)
{
  schro_opengl_lock (display);

  glDepthFunc (GL_LESS);
  glEnable (GL_DEPTH_TEST);
  glClearColor (0.2, 0.2, 0.2, 1.0);
  glViewport (0, 0, display->win_width, display->win_height);

  schro_opengl_unlock (display);
}

void
schro_opengl_check_error (SchroOpenGL * display, int line)
{
  GLenum err = glGetError ();

  if (err) {
    SCHRO_ERROR ("GL Error 0x%x at line %d", (int) err, line);
    SCHRO_ASSERT (0);
  }
}


GLuint
schro_opengl_upload_texture_rectangle (SchroOpenGL * display,
    void *data, int width, int height)
{
#if 0
  GLuint texture;

  glGenTextures (1, &texture);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, texture);

  switch (type) {
    case GST_VIDEO_FORMAT_RGBx:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height,
          0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_RGBA, GL_UNSIGNED_BYTE, data);
      break;
    case GST_VIDEO_FORMAT_BGRx:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height,
          0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_BGRA, GL_UNSIGNED_BYTE, data);
      break;
    case GST_VIDEO_FORMAT_xRGB:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height,
          0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_BGRA, GL_UNSIGNED_INT_8_8_8_8, data);
      break;
    case GST_VIDEO_FORMAT_xBGR:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height,
          0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, data);
      break;
    case GST_VIDEO_FORMAT_YUY2:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_YCBCR_MESA, width, height,
          0, GL_YCBCR_MESA, GL_UNSIGNED_SHORT_8_8_REV_MESA, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_YCBCR_MESA, GL_UNSIGNED_SHORT_8_8_REV_MESA, data);
      break;
    case GST_VIDEO_FORMAT_UYVY:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_YCBCR_MESA, width, height,
          0, GL_YCBCR_MESA, GL_UNSIGNED_SHORT_8_8_REV_MESA, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_YCBCR_MESA, GL_UNSIGNED_SHORT_8_8_MESA, data);
      break;
    case GST_VIDEO_FORMAT_AYUV:
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height,
          0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
      glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
          GL_BGRA, GL_UNSIGNED_INT_8_8_8_8, data);
      break;
    default:
      g_assert_not_reached ();
  }

  return texture;
#endif
  return 0;
}


void
schro_opengl_draw_texture (SchroOpenGL * display, GLuint texture,
    int width, int height, int sync)
{
  //g_return_if_fail (width > 0);
  //g_return_if_fail (height > 0);
  //g_return_if_fail (texture != None);

  schro_opengl_lock (display);

  //g_assert (display->window != None);
  //g_assert (display->context != NULL);

  //schro_opengl_update_attributes (display);
#if 0
  /* Doesn't work */
  {
    int64_t ust = 1234;
    int64_t mst = 1234;
    int64_t sbc = 1234;
    int ret;

    ret = glXGetSyncValuesOML (display->display, display->window,
        &ust, &mst, &sbc);
    SCHRO_ERROR ("sync values %d %lld %lld %lld", ret, ust, mst, sbc);
  }
#endif

  if (sync) {
    glXSwapIntervalSGI (1);
  } else {
    glXSwapIntervalSGI (0);
  }

  glViewport (0, 0, display->win_width, display->win_height);

  glClearColor (0.3, 0.3, 0.3, 1.0);
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();

  glMatrixMode (GL_MODELVIEW);
  glLoadIdentity ();

  glDisable (GL_CULL_FACE);
  glEnableClientState (GL_TEXTURE_COORD_ARRAY);

  glColor4f (1, 1, 1, 1);

  glEnable (GL_TEXTURE_RECTANGLE_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, texture);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glColor4f (1, 0, 1, 1);
  schro_opengl_check_error (display, __LINE__);
  glBegin (GL_QUADS);

  glNormal3f (0, 0, -1);

  glTexCoord2f (width, 0);
  glVertex3f (1.0, 1.0, 0);
  glTexCoord2f (0, 0);
  glVertex3f (-1.0, 1.0, 0);
  glTexCoord2f (0, height);
  glVertex3f (-1.0, -1.0, 0);
  glTexCoord2f (width, height);
  glVertex3f (1.0, -1.0, 0);
  glEnd ();
  schro_opengl_check_error (display, __LINE__);

  glXSwapBuffers (display->display, display->window);

  schro_opengl_unlock (display);
}
