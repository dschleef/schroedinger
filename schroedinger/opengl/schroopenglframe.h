
#ifndef __SCHRO_OPENGL_FRAME_H__
#define __SCHRO_OPENGL_FRAME_H__

#include <schroedinger/schroframe.h>
#include <schroedinger/opengl/schroopengl.h>
#include <GL/glew.h>

SCHRO_BEGIN_DECLS

#define SCHRO_FRAME_IS_OPENGL(_frame) \
    ((_frame)->domain && ((_frame)->domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL))

#define SCHRO_OPENGL_TRANSFER_PIXELBUFFERS 4

typedef struct _SchroOpenGLTexture SchroOpenGLTexture;
typedef struct _SchroOpenGLTransfer SchroOpenGLTransfer;
typedef struct _SchroOpenGLCanvas SchroOpenGLCanvas;/*
typedef struct _SchroOpenGLCanvasPool SchroOpenGLCanvasPool;*/

struct _SchroOpenGLTexture {
  GLuint handles[2];
  GLenum internal_format;
  GLenum pixel_format;
  GLenum type;
  int channels;
};

struct _SchroOpenGLTransfer {
  GLenum type;
  int stride;
  GLuint pixelbuffers[SCHRO_OPENGL_TRANSFER_PIXELBUFFERS];
  int heights[SCHRO_OPENGL_TRANSFER_PIXELBUFFERS];
};

struct _SchroOpenGLCanvas {
  SchroOpenGL *opengl;
  SchroFrameFormat format;
  int width;
  int height;
  SchroOpenGLTexture texture;
  GLuint framebuffers[2];
  SchroOpenGLTransfer push;
  SchroOpenGLTransfer pull;
};

#define SCHRO_OPENGL_CANVAS_POOL_SIZE 150

// FIXME: add a mechanism to drop long time unused canvases from the pool
struct _SchroOpenGLCanvasPool {
  SchroOpenGLCanvas *canvases[SCHRO_OPENGL_CANVAS_POOL_SIZE];
  int size;
};

#define SCHRO_OPENGL_FRAME_STORE_BGRA           (1 <<  0)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8      (1 <<  1)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_F16      (1 <<  2)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_F32      (1 <<  3)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16    (1 <<  4)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_I16     (1 <<  5)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_F16     (1 <<  6)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_F32     (1 <<  7)

#define SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD     (1 <<  8)
#define SCHRO_OPENGL_FRAME_PUSH_SHADER          (1 <<  9)
#define SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS      (1 << 10)
#define SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER  (1 << 11)
#define SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32       (1 << 12)
#define SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER (1 << 13)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16      (1 << 14)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32      (1 << 15)

#define SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER     (1 << 16)
#define SCHRO_OPENGL_FRAME_PULL_U8_AS_F32       (1 << 17)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_U16      (1 << 18)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_F32      (1 << 19)

extern unsigned int _schro_opengl_frame_flags; // FIXME: s/frame/canvas

#define SCHRO_OPENGL_FRAME_IS_FLAG_SET(_flag) \
    (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_##_flag)
#define SCHRO_OPENGL_FRAME_SET_FLAG(_flag) \
    (_schro_opengl_frame_flags |= SCHRO_OPENGL_FRAME_##_flag)
#define SCHRO_OPENGL_FRAME_CLEAR_FLAG(_flag) \
    (_schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_##_flag)

void schro_opengl_frame_check_flags (void);
void schro_opengl_frame_print_flags (const char* indent);

void schro_opengl_frame_setup (SchroOpenGL *opengl, SchroFrame *frame);
void schro_opengl_frame_cleanup (SchroFrame *frame);

SchroFrame *schro_opengl_frame_new (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrameFormat format, int width,
    int height);
SchroFrame *schro_opengl_frame_clone (SchroFrame *opengl_frame);
SchroFrame *schro_opengl_frame_clone_and_push (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrame *cpu_frame);

void schro_opengl_frame_push (SchroFrame *dest, SchroFrame *src); // CPU -> GPU
void schro_opengl_frame_pull (SchroFrame *dest, SchroFrame *src); // CPU <- GPU

void schro_opengl_frame_convert (SchroFrame *dest, SchroFrame *src);
void schro_opengl_frame_add (SchroFrame *dest, SchroFrame *src);
void schro_opengl_frame_subtract (SchroFrame *dest, SchroFrame *src);

void schro_opengl_frame_inverse_iwt_transform (SchroFrame *frame,
    SchroParams *params);

void schro_opengl_upsampled_frame_upsample (SchroUpsampledFrame *upsampled_frame);

SchroOpenGLCanvas *schro_opengl_canvas_new (SchroOpenGL *opengl,
    SchroFrameFormat format, int width, int height);
void schro_opengl_canvas_free (SchroOpenGLCanvas *canvas);

SchroOpenGLCanvasPool *schro_opengl_canvas_pool_new (void);
void schro_opengl_canvas_pool_free (SchroOpenGLCanvasPool* canvas_pool);
int schro_opengl_canvas_pool_is_empty (SchroOpenGLCanvasPool* canvas_pool);
int schro_opengl_canvas_pool_is_full (SchroOpenGLCanvasPool* canvas_pool);
SchroOpenGLCanvas *schro_opengl_canvas_pool_pull (SchroOpenGLCanvasPool* canvas_pool,
    SchroFrameFormat format, int width, int height);
void schro_opengl_canvas_pool_push (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGLCanvas *canvas);

void schro_frame_print (SchroFrame *frame, const char* name);

SCHRO_END_DECLS

#endif

