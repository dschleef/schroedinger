
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <liboil/liboil.h>

static void
schro_opengl_frame_push_convert (SchroFrameData *dest, SchroFrameData *src,
    void *texture_data, int y_offset, int height)
{
  int x, y;
  int width, depth;
  int frame_byte_stride, texture_byte_stride, texture_components;
  SchroOpenGLFrameData *dest_opengl_data = NULL;
  uint8_t *frame_data_u8 = NULL;
  int16_t *frame_data_s16 = NULL;
  uint8_t *texture_data_u8 = NULL;
  uint16_t *texture_data_u16 = NULL;
  int16_t *texture_data_s16 = NULL;
  float *texture_data_f32 = NULL;

  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (dest->format == src->format);
  SCHRO_ASSERT (texture_data != NULL);
  SCHRO_ASSERT (dest->stride == src->stride);
  SCHRO_ASSERT (dest->width == src->width);

  width = src->width;
  depth = SCHRO_FRAME_FORMAT_DEPTH (src->format);
  frame_byte_stride = src->stride;
  dest_opengl_data = (SchroOpenGLFrameData *) dest->data;
  texture_byte_stride = dest_opengl_data->push.byte_stride;

  if (SCHRO_FRAME_IS_PACKED (src->format)) {
    texture_components = 1;
  } else {
    texture_components = dest_opengl_data->texture.components;
  }

  if (depth == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32) {
      texture_data_f32 = (float *) texture_data;
      frame_data_u8 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          texture_data_f32[x * texture_components]
              = (float) frame_data_u8[x] / 255.0;
        }

        texture_data_f32 = OFFSET (texture_data_f32, texture_byte_stride);
        frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
      }
    } else {
      texture_data_u8 = (uint8_t *) texture_data;
      frame_data_u8 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);

      if (texture_components > 1) {
        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            texture_data_u8[x * texture_components] = frame_data_u8[x];
          }

          texture_data_u8 = OFFSET (texture_data_u8, texture_byte_stride);
          frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
        }
      } else {
        for (y = 0; y < height; ++y) {
          oil_memcpy (texture_data_u8, frame_data_u8, width);

          texture_data_u8 = OFFSET (texture_data_u8, texture_byte_stride);
          frame_data_u8 = OFFSET (frame_data_u8, frame_byte_stride);
        }
      }
    }
  } else if (depth == SCHRO_FRAME_FORMAT_DEPTH_S16) {
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16) {
      texture_data_u16 = (uint16_t *) texture_data;
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          texture_data_u16[x * texture_components]
              = (uint16_t) ((int32_t) frame_data_s16[x] + 32768);
        }

        texture_data_u16 = OFFSET (texture_data_u16, texture_byte_stride);
        frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
      }
    } else if (_schro_opengl_frame_flags
        & SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32) {
      texture_data_f32 = (float *) texture_data;
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);

      for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
          texture_data_f32[x * texture_components]
              = (float) ((int32_t) frame_data_s16[x] + 32768) / 65535.0;
        }

        texture_data_f32 = OFFSET (texture_data_f32, texture_byte_stride);
        frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
      }
    } else {
      texture_data_s16 = (int16_t *) texture_data;
      frame_data_s16 = SCHRO_FRAME_DATA_GET_LINE (src, y_offset);

      if (texture_components > 1) {
        for (y = 0; y < height; ++y) {
          for (x = 0; x < width; ++x) {
            texture_data_s16[x * texture_components] = frame_data_s16[x];
          }

          texture_data_s16 = OFFSET (texture_data_s16, texture_byte_stride);
          frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
        }
      } else {
        for (y = 0; y < height; ++y) {
          oil_memcpy (texture_data_s16, frame_data_s16,
              width * sizeof (int16_t));

          texture_data_s16 = OFFSET (texture_data_s16, texture_byte_stride);
          frame_data_s16 = OFFSET (frame_data_s16, frame_byte_stride);
        }
      }
    }
  } else {
    SCHRO_ERROR ("unhandled depth");
    SCHRO_ASSERT (0);
  }
}

void
schro_opengl_frame_push (SchroFrame *dest, SchroFrame *src)
{
  int i, k;
  int width, height, depth;
  int components;
  int pixelbuffer_y_offset, pixelbuffer_height;
  SchroOpenGLFrameData *dest_opengl_data = NULL;
  static void *texture_data = NULL; // FIXME
  static int texture_data_length = 0;
  GLuint src_texture = 0;
  void *mapped_data = NULL;
  SchroOpenGLShader *shader;
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
  double start, end;
#endif

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (dest != NULL);
  SCHRO_ASSERT (src != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (dest));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_OPENGL (src));
  SCHRO_ASSERT (dest->format == src->format);

  schro_opengl_lock ();

  if (SCHRO_FRAME_IS_PACKED (src->format)) {
    components = 1;
  } else {
    components = 3;
  }

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    dest_opengl_data = (SchroOpenGLFrameData *) dest->components[i].data;

    SCHRO_ASSERT (dest_opengl_data != NULL);

    width = dest->components[i].width;
    height = dest->components[i].height;
    depth = SCHRO_FRAME_FORMAT_DEPTH (src->format);

    //SCHRO_ASSERT (stride == src->components[i].stride);
    SCHRO_ASSERT (width == src->components[i].width);
    SCHRO_ASSERT (height == src->components[i].height);

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    start = schro_utils_get_time ();
#endif

    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD) {
      glGenTextures (1, &src_texture);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, src_texture);
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0,
          dest_opengl_data->texture.internal_format, width, height, 0,
          dest_opengl_data->texture.pixel_format,
          dest_opengl_data->texture.type, NULL);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
          GL_NEAREST);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
          GL_NEAREST);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

      SCHRO_OPENGL_CHECK_ERROR

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      end = schro_utils_get_time ();
      SCHRO_INFO ("tex %f", end - start);
      start = schro_utils_get_time ();
#endif
    }

    if ((_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER
        && depth == SCHRO_FRAME_FORMAT_DEPTH_U8)
        || (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER
        && depth == SCHRO_FRAME_FORMAT_DEPTH_S16)) {
      pixelbuffer_y_offset = 0;

      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
        start = schro_utils_get_time ();
#endif
        pixelbuffer_height = dest_opengl_data->push.heights[k];

        glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_EXT,
            dest_opengl_data->push.pixelbuffers[k]);
        glBufferDataARB (GL_PIXEL_UNPACK_BUFFER_ARB,
            dest_opengl_data->push.byte_stride * pixelbuffer_height, NULL,
            GL_STREAM_DRAW_ARB);

        mapped_data = glMapBufferARB (GL_PIXEL_UNPACK_BUFFER_EXT,
            GL_WRITE_ONLY_ARB);

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
        end = schro_utils_get_time ();
        SCHRO_INFO ("map %i %f %i", i, end - start, k);
        start = schro_utils_get_time ();
#endif

        schro_opengl_frame_push_convert (dest->components + i,
            src->components + i, mapped_data, pixelbuffer_y_offset,
            pixelbuffer_height);

        glUnmapBufferARB (GL_PIXEL_UNPACK_BUFFER_EXT);

        pixelbuffer_y_offset += pixelbuffer_height;

        SCHRO_OPENGL_CHECK_ERROR
      }

      if (!(_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD)) {
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
            dest_opengl_data->texture.handle);
      }

      pixelbuffer_y_offset = 0;

      if (dest_opengl_data->push.type == GL_SHORT) {
        /* OpenGL maps signed values different to float values than unsigned
           values. for S16 -32768 is mapped to -1.0 and 32767 to 1.0, for U16
           0 is mapped to 0.0 and 65535 to 1.0. after this mapping scale and
           bias are applied and the resulting value is clamped to [0..1].
           with default scale = 1 and default bias = 0 all negative values
           from S16 are clamped to 0.0, changing scale and bias to 0.5 gives
           a unclamped mapping that doesn't discard all negative values for
           S16 */
        glPixelTransferf (GL_RED_SCALE, 0.5);
        glPixelTransferf (GL_RED_BIAS, 0.5);
      }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      start = schro_utils_get_time ();
#endif

      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
        pixelbuffer_height = dest_opengl_data->push.heights[k];

        glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_EXT,
            dest_opengl_data->push.pixelbuffers[k]);

        SCHRO_OPENGL_CHECK_ERROR

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS) {
          glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
              dest_opengl_data->framebuffer);

          glWindowPos2iARB (0, pixelbuffer_y_offset);
          glDrawPixels (width, pixelbuffer_height,
              dest_opengl_data->texture.pixel_format,
              dest_opengl_data->push.type, NULL);

          glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
        } else {
          glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0,
              pixelbuffer_y_offset, width, pixelbuffer_height,
              dest_opengl_data->texture.pixel_format,
              dest_opengl_data->push.type, NULL);
        }

        SCHRO_OPENGL_CHECK_ERROR

        pixelbuffer_y_offset += pixelbuffer_height;
      }

      if (dest_opengl_data->push.type == GL_SHORT) {
        glPixelTransferf (GL_RED_SCALE, 1);
        glPixelTransferf (GL_RED_BIAS, 0);
      }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      end = schro_utils_get_time ();
      SCHRO_INFO ("upl %f", end - start);
      start = schro_utils_get_time ();
#endif

      glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_EXT, 0);
    } else {
      if (texture_data_length != dest_opengl_data->push.byte_stride * height
          || !texture_data) {
        texture_data_length = dest_opengl_data->push.byte_stride * height;

        if (!texture_data) {
          texture_data = schro_malloc (texture_data_length);
        } else {
          texture_data = schro_realloc (texture_data, texture_data_length);
        }
      }

      schro_opengl_frame_push_convert (dest->components + i,
          src->components + i, texture_data, 0, height);

      if (!(_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD)) {
        glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
            dest_opengl_data->texture.handle);
      }

      if (dest_opengl_data->push.type == GL_SHORT) {
        /* OpenGL maps signed values different to float values than unsigned
           values. for S16 -32768 is mapped to -1.0 and 32767 to 1.0, for U16
           0 is mapped to 0.0 and 65535 to 1.0. after this mapping scale and
           bias are applied and the resulting value is clamped to [0..1].
           with default scale = 1 and default bias = 0 all negative values
           from S16 are clamped to 0.0, changing scale and bias to 0.5 gives
           a unclamped mapping that doesn't discard all negative values for
           S16 */
        glPixelTransferf (GL_RED_SCALE, 0.5);
        glPixelTransferf (GL_RED_BIAS, 0.5);
      }

      if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS) {
        glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
            dest_opengl_data->framebuffer);

        glWindowPos2iARB (0, 0);
        glDrawPixels (width, height, dest_opengl_data->texture.pixel_format,
            dest_opengl_data->push.type, texture_data);

        glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
      } else {
        glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, width, height,
            dest_opengl_data->texture.pixel_format,
            dest_opengl_data->push.type, texture_data);
      }

      if (dest_opengl_data->push.type == GL_SHORT) {
        glPixelTransferf (GL_RED_SCALE, 1);
        glPixelTransferf (GL_RED_BIAS, 0);
      }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      end = schro_utils_get_time ();
      SCHRO_INFO ("upl %f", end - start);
      start = schro_utils_get_time ();
#endif
    }

    SCHRO_OPENGL_CHECK_ERROR

    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD) {
      glViewport (0, 0, width, height);

      glLoadIdentity ();
      glOrtho (0, width, 0, height, -1, 1);

      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, dest_opengl_data->framebuffer);

      SCHRO_OPENGL_CHECK_ERROR

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      end = schro_utils_get_time ();
      SCHRO_INFO ("fbo %f", end - start);
      start = schro_utils_get_time ();
#endif

      if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_SHADER) {
        shader = schro_opengl_shader_get (SCHRO_OPENGL_SHADER_IDENTITY);

        glUseProgramObjectARB (shader->program);
        glUniform1iARB (shader->textures[0], 0);
      }

      glBegin (GL_QUADS);
      glTexCoord2f (width, 0);      glVertex3f (width, 0,      0);
      glTexCoord2f (0,     0);      glVertex3f (0,     0,      0);
      glTexCoord2f (0,     height); glVertex3f (0,     height, 0);
      glTexCoord2f (width, height); glVertex3f (width, height, 0);
      glEnd ();

      if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_SHADER) {
        glUseProgramObjectARB (0);
      }

      glFlush ();

      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

      SCHRO_OPENGL_CHECK_ERROR

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
      end = schro_utils_get_time ();
      SCHRO_INFO ("drw %f", end - start);
      start = schro_utils_get_time ();
#endif
    }

    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD) {
      glDeleteTextures (1, &src_texture);
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("fin %f +++", end - start);
#endif
  }

  schro_opengl_unlock ();
}

