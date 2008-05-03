
#ifndef __SCHRO_OPENGL_FRAME_H__
#define __SCHRO_OPENGL_FRAME_H__

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schroframe.h>
#include <GL/gl.h>

SCHRO_BEGIN_DECLS

//#define OPENGL_INTERNAL_TIME_MEASUREMENT

#define SCHRO_FRAME_IS_OPENGL(frame) \
    ((frame)->domain && ((frame)->domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL))

#define SCHRO_OPENGL_FRAME_PIXELBUFFERS 4

typedef struct _SchroOpenGLTexture SchroOpenGLTexture;
typedef struct _SchroOpenGLTransfer SchroOpenGLTransfer;
typedef struct _SchroOpenGLFrameData SchroOpenGLFrameData;

struct _SchroOpenGLTexture {
  GLuint handle;
  GLenum internal_format;
  GLenum pixel_format;
  GLenum type;
  int components;
};

struct _SchroOpenGLTransfer {
  GLenum type;
  int bytes_per_texel;
  int byte_stride;
  int texel_stride;
  GLuint pixelbuffers[SCHRO_OPENGL_FRAME_PIXELBUFFERS];
  int heights[SCHRO_OPENGL_FRAME_PIXELBUFFERS];
};

struct _SchroOpenGLFrameData {
  SchroOpenGLTexture texture;
  GLuint framebuffer;
  SchroOpenGLTransfer push;
  SchroOpenGLTransfer pull;
  int bytes_per_pixel;
};

#define SCHRO_OPENGL_FRAME_STORE_RGBA        (1 <<  0)
#define SCHRO_OPENGL_FRAME_STORE_BGRA        (1 <<  1)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8   (1 <<  2)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_F32   (1 <<  3)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16 (1 <<  4)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_I16  (1 <<  5)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_F32  (1 <<  6)

#define SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD  (1 <<  7)
#define SCHRO_OPENGL_FRAME_PUSH_SHADER       (1 <<  8)
#define SCHRO_OPENGL_FRAME_PUSH_PIXELBUFFER  (1 <<  9)
#define SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32    (1 << 10)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16   (1 << 11)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32   (1 << 12)

#define SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER  (1 << 13)
#define SCHRO_OPENGL_FRAME_PULL_U8_AS_F32    (1 << 14)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_U16   (1 << 15)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_F32   (1 << 16)

extern unsigned int _schro_opengl_frame_flags;

void schro_opengl_frame_check_flags (void);
void schro_opengl_frame_print_flags (const char* indent);

void schro_opengl_frame_setup (SchroFrame *frame);
void schro_opengl_frame_cleanup (SchroFrame *frame);

void schro_opengl_frame_push (SchroFrame *dest, SchroFrame *src); // CPU -> GPU
void schro_opengl_frame_pull (SchroFrame *dest, SchroFrame *src); // CPU <- GPU

void schro_opengl_frame_convert (SchroFrame *dest, SchroFrame *src);

void schro_opengl_frame_convert_s16_u8 (SchroFrame *dest, SchroFrame *src);

SCHRO_END_DECLS

#endif

