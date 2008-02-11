
#ifndef __SCHRO_OPENGL_H__
#define __SCHRO_OPENGL_H__

#include <GL/glx.h>
#include <GL/gl.h>

typedef struct _SchroOpenGL SchroOpenGL;

struct _SchroOpenGL {
  Display *display;
  GC gc;
  XVisualInfo *visinfo;
  GLXContext context;
  //GMutex *lock;

  Screen *screen;
  int screen_num;
  Visual *visual;
  Window root;
  uint32_t white;
  uint32_t black;
  int depth;

  int max_texture_size;

  int have_ycbcr_texture;
  int have_texture_rectangle;
  int have_color_matrix;

  Window window;
  int visible;
  Window parent_window;

  int win_width;
  int win_height;

}; 


SchroOpenGL *schro_opengl_new (void);
int schro_opengl_connect (SchroOpenGL *display,
    const char *display_name);
void schro_opengl_lock (SchroOpenGL *display);
void schro_opengl_unlock (SchroOpenGL *display);
void schro_opengl_set_window (SchroOpenGL *display, Window window);
void schro_opengl_update_attributes (SchroOpenGL *display);
void schro_opengl_clear (SchroOpenGL *display);
void schro_opengl_draw_texture (SchroOpenGL * display, GLuint texture,
    int width, int height, int sync);
void schro_opengl_check_error (SchroOpenGL *display, int line);
GLuint schro_opengl_upload_texture_rectangle (SchroOpenGL *display,
    void *data, int width, int height);
void schro_opengl_set_visible (SchroOpenGL *display, int visible);
void schro_opengl_set_window_size (SchroOpenGL *display, int width,
    int height);
void schro_opengl_update_window (SchroOpenGL * display);

#endif

