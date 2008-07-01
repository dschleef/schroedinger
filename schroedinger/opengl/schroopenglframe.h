
#ifndef __SCHRO_OPENGL_FRAME_H__
#define __SCHRO_OPENGL_FRAME_H__

#include <schroedinger/schro-stdint.h>
#include <schroedinger/schroframe.h>
#include <schroedinger/opengl/schroopengl.h>
#include <GL/glew.h>

SCHRO_BEGIN_DECLS

//#define OPENGL_INTERNAL_TIME_MEASUREMENT

#define SCHRO_FRAME_IS_OPENGL(_frame) \
    ((_frame)->domain && ((_frame)->domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL))

#define SCHRO_OPENGL_FRAME_PIXELBUFFERS 4

typedef struct _SchroOpenGLTexture SchroOpenGLTexture;
typedef struct _SchroOpenGLTransfer SchroOpenGLTransfer;
typedef struct _SchroOpenGLFrameData SchroOpenGLFrameData;

struct _SchroOpenGLTexture {
  GLuint handles[2];
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
  SchroOpenGL *opengl;
  SchroOpenGLTexture texture;
  GLuint framebuffers[2];
  SchroOpenGLTransfer push;
  SchroOpenGLTransfer pull;
};

#define SCHRO_OPENGL_FRAME_STORE_BGRA           (1 <<  1)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8      (1 <<  2)
#define SCHRO_OPENGL_FRAME_STORE_U8_AS_F32      (1 <<  3)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16    (1 <<  4)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_I16     (1 <<  5)
#define SCHRO_OPENGL_FRAME_STORE_S16_AS_F32     (1 <<  6)

#define SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD     (1 <<  7)
#define SCHRO_OPENGL_FRAME_PUSH_SHADER          (1 <<  8)
#define SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS      (1 <<  9)
#define SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER  (1 << 10)
#define SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32       (1 << 11)
#define SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER (1 << 12)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16      (1 << 13)
#define SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32      (1 << 14)

#define SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER     (1 << 15)
#define SCHRO_OPENGL_FRAME_PULL_U8_AS_F32       (1 << 16)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_U16      (1 << 17)
#define SCHRO_OPENGL_FRAME_PULL_S16_AS_F32      (1 << 18)

extern unsigned int _schro_opengl_frame_flags;

void schro_opengl_frame_check_flags (void);
void schro_opengl_frame_print_flags (const char* indent);

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

void schro_frame_print (SchroFrame *frame, const char* name);

SCHRO_END_DECLS

#endif

