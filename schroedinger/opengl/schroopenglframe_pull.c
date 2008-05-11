
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <liboil/liboil.h>

static void
schro_opengl_frame_pull_convert (SchroFrameData *dest, SchroFrameData *src,
    void *texture_data, int y_offset, int height)
{
  int x, y;
  int width, depth;
  int frame_byte_stride, texture_byte_stride, texture_components;
  SchroOpenGLFrameData *src_opengl_data = NULL;
  uint8_t *texture_data_u8 = NULL;
  uint16_t *texture_data_u16 = NULL;
  int16_t *texture_data_s16 = NULL;
  float *texture_data_f32 = NULL;
  uint8_t *frame_data_u8 = NULL;
  int16_t *frame_data_s16 = NULL;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (dest->format == src->format);
  SCHRO_ASSERT (texture_data != NULL);
  SCHRO_ASSERT (dest->stride == src->stride);
  SCHRO_ASSERT (dest->width == src->width);

  width = dest->width;
  depth = SCHRO_FRAME_FORMAT_DEPTH (dest->format);
  frame_byte_stride = dest->stride;
  src_opengl_data = (SchroOpenGLFrameData *) src->data;
  texture_byte_stride = src_opengl_data->pull.byte_stride;

  if (SCHRO_FRAME_IS_PACKED (src->format)) {
    texture_components = 1;
  } else {
    texture_components = src_opengl_data->texture.components;
  }

  if (depth == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_U8_AS_F32) {
      frame_data_u8 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);
      texture_data_f32 = (float *) texture_data;

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          frame_data_u8[x]
              = (uint8_t) (texture_data_f32[x * texture_components] * 255.0);
        }

        frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
        texture_data_f32 = OFFSET (texture_data_f32, texture_byte_stride);
      }
    } else {
      frame_data_u8 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);
      texture_data_u8 = (uint8_t *) texture_data;

      if (texture_components > 1) {
        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            frame_data_u8[x] = texture_data_u8[x * texture_components];
          }

          frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
          texture_data_u8 = OFFSET (texture_data_u8, texture_byte_stride);
        }
      } else {
        for (y = 0; y < height; ++y) {
          oil_memcpy (frame_data_u8, texture_data_u8, width);

          frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
          texture_data_u8 = OFFSET (texture_data_u8, texture_byte_stride);
        }
      }
    }
  } else if (depth == SCHRO_FRAME_FORMAT_DEPTH_S16) {
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_S16_AS_F32) {
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);
      texture_data_f32 = (float *) texture_data;

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          // FIXME: for some unknown reason I need to scale with 65536.0
          // instead of 65535.0 to get correct S16 value back. I also get
          // correct S16 values with rounding: round (x) := floor (x + 0.5)
          // but thats way to expensive
          frame_data_s16[x]
              = (int16_t) ((int32_t) (texture_data_f32[x * texture_components]
              * 65536.0) - 32768);
        }

        frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
        texture_data_f32 = OFFSET (texture_data_f32, texture_byte_stride);
      }
    } else if (_schro_opengl_frame_flags
        & SCHRO_OPENGL_FRAME_PULL_S16_AS_U16) {
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (dest, y_offset);
      texture_data_u16 = (uint16_t *) texture_data;

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          frame_data_s16[x]
              = (int16_t) ((int32_t) texture_data_u16[x * texture_components]
              - 32768);
        }

        frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
        texture_data_u16 = OFFSET (texture_data_u16, texture_byte_stride);
      }
    } else {
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);
      texture_data_s16 = (int16_t *) texture_data;

      if (texture_components > 1) {
        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            frame_data_s16[x] = texture_data_s16[x * texture_components];
          }

          frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
          texture_data_s16 = OFFSET (texture_data_s16, texture_byte_stride);
        }
      } else {
        for (y = 0; y < height; ++y) {
          oil_memcpy (frame_data_s16, texture_data_s16,
              width * sizeof (int16_t));

          frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
          texture_data_s16 = OFFSET (texture_data_s16, texture_byte_stride);
        }
      }
    }
  } else {
    SCHRO_ERROR ("unhandled depth");
    SCHRO_ASSERT (0);
  }
}

void
schro_opengl_frame_pull (SchroFrame *dest, SchroFrame *src)
{
  int i, k;
  int width, height;
  int components;
  int pixelbuffer_y_offset, pixelbuffer_height;
  SchroOpenGLFrameData *src_opengl_data;
  static void *texture_data = NULL; // FIXME
  static int texture_data_length = 0;
  void *mapped_data = NULL;

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (!SCHRO_FRAME_IS_OPENGL (dest));
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (src));
  SCHRO_ASSERT (dest->format == src->format);

  schro_opengl_lock ();

  if (SCHRO_FRAME_IS_PACKED (src->format)) {
    components = 1;
  } else {
    components = 3;
  }

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    src_opengl_data = (SchroOpenGLFrameData *) src->components[i].data;

    SCHRO_ASSERT (src_opengl_data != NULL);
    SCHRO_ASSERT (src_opengl_data->texture.handle != 0);
    SCHRO_ASSERT (src_opengl_data->framebuffer != 0);

    width = src->components[i].width;
    height = src->components[i].height;

    SCHRO_ASSERT (dest->components[i].width == width);
    SCHRO_ASSERT (dest->components[i].height == height);

    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER) {
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, src_opengl_data->framebuffer);

      pixelbuffer_y_offset = 0;

      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
        pixelbuffer_height = src_opengl_data->pull.heights[k];

        glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB,
            src_opengl_data->pull.pixelbuffers[k]);
        glReadPixels (0, pixelbuffer_y_offset, width, pixelbuffer_height,
            src_opengl_data->texture.pixel_format, src_opengl_data->pull.type,
            NULL);

        pixelbuffer_y_offset += pixelbuffer_height;

        SCHRO_OPENGL_CHECK_ERROR
      }

      pixelbuffer_y_offset = 0;

      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
        pixelbuffer_height = src_opengl_data->pull.heights[k];

        glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB,
            src_opengl_data->pull.pixelbuffers[k]);
        /*glBufferDataARB (GL_PIXEL_PACK_BUFFER_ARB,
            opengl_data->pull.byte_stride * pixelbuffer_height, NULL,
            GL_STATIC_READ_ARB);*/

        mapped_data = glMapBufferARB (GL_PIXEL_PACK_BUFFER_ARB,
            GL_READ_ONLY_ARB);

        schro_opengl_frame_pull_convert (dest->components + i,
            src->components + i, mapped_data, pixelbuffer_y_offset,
            pixelbuffer_height);

        glUnmapBufferARB (GL_PIXEL_PACK_BUFFER_ARB);

        pixelbuffer_y_offset += pixelbuffer_height;

        SCHRO_OPENGL_CHECK_ERROR
      }

      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
    } else {
      if (texture_data_length != src_opengl_data->pull.byte_stride * height
          || !texture_data) {
        texture_data_length = src_opengl_data->pull.byte_stride * height;

        if (!texture_data) {
          texture_data = schro_malloc (texture_data_length);
        } else {
          texture_data = schro_realloc (texture_data, texture_data_length);
        }
      }

      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, src_opengl_data->framebuffer);
      glReadPixels (0, 0, src_opengl_data->pull.texel_stride, height,
          src_opengl_data->texture.pixel_format, src_opengl_data->pull.type,
          texture_data);
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

      schro_opengl_frame_pull_convert (dest->components + i,
          src->components + i, texture_data, 0, height);
    }
  }

  schro_opengl_unlock ();
}

