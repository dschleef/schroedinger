
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <schroedinger/opengl/schroopenglwavelet.h>
#include <liboil/liboil.h>
#include <stdio.h>

unsigned int _schro_opengl_frame_flags
    = 0
    //| SCHRO_OPENGL_FRAME_STORE_BGRA /* FIXME: currently broken with packed formats in convert */
    //| SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8
    //| SCHRO_OPENGL_FRAME_STORE_U8_AS_F16
    //| SCHRO_OPENGL_FRAME_STORE_U8_AS_F32
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_I16
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_F16 /* FIXME: currently broken */
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_F32 /* FIXME: currently broken */

    //| SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD
    //| SCHRO_OPENGL_FRAME_PUSH_SHADER
    //| SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS /* FIXME: currently broken */
    | SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER
    //| SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32
    //| SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER
    | SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16
    //| SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32

    //| SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER
    //| SCHRO_OPENGL_FRAME_PULL_U8_AS_F32
    | SCHRO_OPENGL_FRAME_PULL_S16_AS_U16
    //| SCHRO_OPENGL_FRAME_PULL_S16_AS_F32
    ;

/* results on a NVIDIA 8800 GT with nvidia-glx-new drivers on Ubuntu Hardy */

/* U8: 259.028421/502.960679 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = 0;*/

/* U8: 382.692291/447.573619 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD
    | SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER;*/

/* U8: 972.809028/962.217704 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8;*/

/* U8: 1890.699986/848.954058 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8
    | SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER;*/

/* U8: 2003.478261/462.976159 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER;*/

/* S16: 22.265474/492.245509 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16
    | SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER
    | SCHRO_OPENGL_FRAME_PULL_S16_AS_U16;*/

/* S16: 85.136173/499.591624 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_PULL_S16_AS_U16;*/

/* S16: 266.568537/490.034023 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16
    | SCHRO_OPENGL_FRAME_PULL_S16_AS_U16;*/

/* S16: 601.249413/914.319981 mbyte/sec *//*
unsigned int _schro_opengl_frame_flags
    = SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
    | SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16
    | SCHRO_OPENGL_FRAME_PULL_S16_AS_U16;*/

void
schro_opengl_frame_check_flags (void)
{
  /* store */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_BGRA)
      && !GLEW_EXT_bgra) {
    SCHRO_ERROR ("missing extension GL_EXT_bgra, disabling BGRA storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_BGRA);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8) ||
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
    if (!GLEW_EXT_texture_integer) {
      SCHRO_ERROR ("missing extension GL_EXT_texture_integer, can't store "
          "U8/S16 as UI8/UI16/I16, disabling U8/S16 as UI8/UI16/I16 storing");
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_UI8);
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_UI16);
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_I16);
    }
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F16) ||
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F32) ||
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16) ||
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
    if (!GLEW_ARB_texture_float && !GLEW_ATI_texture_float) {
      SCHRO_ERROR ("missing extension GL_{ARB|ATI}_texture_float, can't "
          "store U8/S16 as F16/F32, disabling U8/S16 as F16/F32 storing");
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_F16);
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_F32);
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F16);
      SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F32);
    }
  }

  /* store U8 */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F16)) {
    SCHRO_ERROR ("can't store U8 in UI8 and F16, disabling F16 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_F16);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F32)) {
    SCHRO_ERROR ("can't store U8 in UI8 and F32, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_F32);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F32)) {
    SCHRO_ERROR ("can't store U8 in F16 and F32, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_U8_AS_F32);
  }

  /* store S16 */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
    SCHRO_ERROR ("can't store S16 in UI16 and I16, disabling UI16 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_UI16);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16)) {
    SCHRO_ERROR ("can't store S16 in UI16 and F16, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F16);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
    SCHRO_ERROR ("can't store S16 in UI16 and F32, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F32);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16)) {
    SCHRO_ERROR ("can't store S16 in I16 and F16, disabling F16 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F16);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
    SCHRO_ERROR ("can't store S16 in I16 and F32, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F32);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
    SCHRO_ERROR ("can't store S16 in F16 and F32, disabling F32 storing");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (STORE_S16_AS_F32);
  }

  /* push */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_RENDER_QUAD) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_DRAWPIXELS)) {
    SCHRO_ERROR ("can't render quad and drawpixels to push, disabling "
        "drawpixels push");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PUSH_DRAWPIXELS);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_DRAWPIXELS) &&
      !GLEW_ARB_window_pos) {
    SCHRO_ERROR ("missing extension GL_ARB_window_pos, disabling drawpixels "
        "push");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PUSH_DRAWPIXELS);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_U8_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling U8 pixelbuffer push");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PUSH_U8_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling S16 pixelbuffer push");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PUSH_S16_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_U16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_F32)) {
    SCHRO_ERROR ("can't push S16 as U16 and F32, disabling U16 push");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PUSH_S16_AS_U16);
  }

  /* pull */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_PIXELBUFFER) &&
      (!GLEW_ARB_vertex_buffer_object || !GLEW_ARB_pixel_buffer_object)) {
    SCHRO_ERROR ("missing extensions GL_ARB_vertex_buffer_object and/or "
        "GL_ARB_pixel_buffer_object, disabling S16 pixelbuffer pull");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PULL_PIXELBUFFER);
  }

  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_U16) &&
      SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_F32)) {
    SCHRO_ERROR ("can't pull S16 as U16 and F32, disabling U16 pull");
    SCHRO_OPENGL_FRAME_CLEAR_FLAG (PULL_S16_AS_U16);
  }
}

void
schro_opengl_frame_print_flags (const char* indent)
{
  schro_opengl_frame_check_flags ();

  #define PRINT_FLAG(_text, _flag) \
      printf ("%s  "_text"%s\n", indent, \
          SCHRO_OPENGL_FRAME_IS_FLAG_SET (_flag) ? "on" : "off")

  printf ("%sstore flags\n", indent);

  PRINT_FLAG ("BGRA:            ", STORE_BGRA);
  PRINT_FLAG ("U8 as UI8:       ", STORE_U8_AS_UI8);
  PRINT_FLAG ("U8 as F16:       ", STORE_U8_AS_F16);
  PRINT_FLAG ("U8 as F32:       ", STORE_U8_AS_F32);
  PRINT_FLAG ("S16 as UI16:     ", STORE_S16_AS_UI16);
  PRINT_FLAG ("S16 as I16:      ", STORE_S16_AS_I16);
  PRINT_FLAG ("S16 as F16:      ", STORE_S16_AS_F16);
  PRINT_FLAG ("S16 as F32:      ", STORE_S16_AS_F32);

  printf ("%spush flags\n", indent);

  PRINT_FLAG ("render quad:     ", PUSH_RENDER_QUAD);
  PRINT_FLAG ("shader:          ", PUSH_SHADER);
  PRINT_FLAG ("drawpixels:      ", PUSH_DRAWPIXELS);
  PRINT_FLAG ("U8 pixelbuffer:  ", PUSH_U8_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", PUSH_U8_AS_F32);
  PRINT_FLAG ("S16 pixelbuffer: ", PUSH_S16_PIXELBUFFER);
  PRINT_FLAG ("S16 as U16:      ", PUSH_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", PUSH_S16_AS_F32);

  printf ("%spull flags\n", indent);

  PRINT_FLAG ("pixelbuffer:     ", PULL_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", PULL_U8_AS_F32);
  PRINT_FLAG ("S16 as U16:      ", PULL_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", PULL_S16_AS_F32);

  #undef PRINT_FLAG
}

void
schro_opengl_frame_setup (SchroOpenGL *opengl, SchroFrame *frame)
{
  int i, k;
  int width, height;
  int components;
  int create_push_pixelbuffers = FALSE;
  SchroOpenGLFrameData *opengl_data = NULL;
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
  double start, end;
#endif

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;

  schro_opengl_lock (opengl);

  schro_opengl_frame_check_flags ();

  for (i = 0; i < components; ++i) {
    width = frame->components[i].width;
    height = frame->components[i].height;

    // FIXME: hack to store custom data per frame component
    opengl_data = (SchroOpenGLFrameData *) frame->components[i].data;

    SCHRO_ASSERT (opengl_data != NULL);
    SCHRO_ASSERT (opengl_data->opengl == NULL);
    SCHRO_ASSERT (opengl_data->texture.handles[0] == 0);
    SCHRO_ASSERT (opengl_data->texture.handles[1] == 0);
    SCHRO_ASSERT (opengl_data->framebuffers[0] == 0);
    SCHRO_ASSERT (opengl_data->framebuffers[1] == 0);

    opengl_data->opengl = opengl;

    switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
      case SCHRO_FRAME_FORMAT_DEPTH_U8:
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F16)) {
          if (!SCHRO_FRAME_IS_PACKED (frame->format) && GLEW_NV_float_buffer) {
            opengl_data->texture.internal_format = GL_FLOAT_R16_NV;
          } else {
            opengl_data->texture.internal_format = GL_RGBA16F_ARB;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F32)) {
          if (!SCHRO_FRAME_IS_PACKED (frame->format) && GLEW_NV_float_buffer) {
            opengl_data->texture.internal_format = GL_FLOAT_R32_NV;
          } else {
            opengl_data->texture.internal_format = GL_RGBA32F_ARB;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
          if (SCHRO_FRAME_IS_PACKED (frame->format)) {
            opengl_data->texture.internal_format = GL_RGBA8UI_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA8UI_EXT;
          }

          opengl_data->texture.type = GL_UNSIGNED_BYTE;
        } else {
          /* must use RGBA format here, because other formats are in general
             not supported by framebuffers */
          opengl_data->texture.internal_format = GL_RGBA8;
          opengl_data->texture.type = GL_UNSIGNED_BYTE;
        }

        if (SCHRO_FRAME_IS_PACKED (frame->format)) {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_BGRA)) {
            if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
              opengl_data->texture.pixel_format = GL_BGRA_INTEGER_EXT;
            } else {
              opengl_data->texture.pixel_format = GL_BGRA;
            }

            opengl_data->texture.components = 4;
          } else {
            if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
              opengl_data->texture.pixel_format = GL_RGBA_INTEGER_EXT;
            } else {
              opengl_data->texture.pixel_format = GL_RGBA;
            }

            opengl_data->texture.components = 4;
          }
        } else {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
            opengl_data->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RED;
          }

          opengl_data->texture.components = 1;
        }

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_U8_AS_F32)) {
          opengl_data->push.type = GL_FLOAT;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (float);
        } else {
          opengl_data->push.type = GL_UNSIGNED_BYTE;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint8_t);
        }

        opengl_data->push.byte_stride
            = ROUND_UP_4 (width * opengl_data->push.bytes_per_texel);
        opengl_data->push.texel_stride
            = width;/*opengl_data->push.byte_stride
            / opengl_data->push.bytes_per_texel*/;

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_U8_AS_F32)) {
          opengl_data->pull.type = GL_FLOAT;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (float);
        } else {
          opengl_data->pull.type = GL_UNSIGNED_BYTE;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint8_t);
        }

        opengl_data->pull.byte_stride
            = ROUND_UP_4 (width * opengl_data->pull.bytes_per_texel);
        opengl_data->pull.texel_stride
            = width;/*opengl_data->pull.byte_stride
            / opengl_data->pull.bytes_per_texel*/;

        //printf ("byte_stride %i texel_stride %i\n",
        //    opengl_data->pull.byte_stride, opengl_data->pull.texel_stride);

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_U8_PIXELBUFFER)) {
          create_push_pixelbuffers = TRUE;
        }

        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S16:
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16)) {
          if (!SCHRO_FRAME_IS_PACKED (frame->format) && GLEW_NV_float_buffer) {
            opengl_data->texture.internal_format = GL_FLOAT_R16_NV;
          } else {
            opengl_data->texture.internal_format = GL_RGBA16F_ARB;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
          if (!SCHRO_FRAME_IS_PACKED (frame->format) && GLEW_NV_float_buffer) {
            opengl_data->texture.internal_format = GL_FLOAT_R32_NV;
          } else {
            opengl_data->texture.internal_format = GL_RGBA32F_ARB;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16)) {
          if (SCHRO_FRAME_IS_PACKED (frame->format)) {
            opengl_data->texture.internal_format = GL_RGBA16UI_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA16UI_EXT;
          }

          opengl_data->texture.type = GL_UNSIGNED_SHORT;
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
          if (SCHRO_FRAME_IS_PACKED (frame->format)) {
            opengl_data->texture.internal_format = GL_RGBA16I_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA16I_EXT;
          }

          opengl_data->texture.type = GL_SHORT;
        } else {
          /* must use RGBA format here, because other formats are in general
             not supported by framebuffers */
          opengl_data->texture.internal_format = GL_RGBA16;
          opengl_data->texture.type = GL_SHORT;
        }

        if (SCHRO_FRAME_IS_PACKED (frame->format)) {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_BGRA)) {
            if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
                SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
              opengl_data->texture.pixel_format = GL_BGRA_INTEGER_EXT;
            } else {
              opengl_data->texture.pixel_format = GL_BGRA;
            }

            opengl_data->texture.components = 4;
          } else {
            if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
                SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
              opengl_data->texture.pixel_format = GL_RGBA_INTEGER_EXT;
            } else {
              opengl_data->texture.pixel_format = GL_RGBA;
            }

            opengl_data->texture.components = 4;
          }
        } else {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
              SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
            opengl_data->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RED;
          }

          opengl_data->texture.components = 1;
        }

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_U16)) {
          opengl_data->push.type = GL_UNSIGNED_SHORT;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint16_t);
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_F32)) {
          opengl_data->push.type = GL_FLOAT;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (float);
        } else {
          opengl_data->push.type = GL_SHORT;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (int16_t);
        }

        opengl_data->push.byte_stride
            = ROUND_UP_4 (width * opengl_data->push.bytes_per_texel);
        opengl_data->push.texel_stride
            = width;/*opengl_data->push.byte_stride
            / opengl_data->push.bytes_per_texel*/;

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_U16)) {
          /* must pull S16 as GL_UNSIGNED_SHORT instead of GL_SHORT because
             the OpenGL mapping form internal float represenation into S16
             values with GL_SHORT maps 0.0 to 0 and 1.0 to 32767 clamping all
             negative values to 0, see glReadPixel documentation. so the pull
             is done with GL_UNSIGNED_SHORT and the resulting U16 values are
             manually shifted to S16 */
          opengl_data->pull.type = GL_UNSIGNED_SHORT;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint16_t);
        } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_F32)) {
          opengl_data->pull.type = GL_FLOAT;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (float);
        } else {
          // FIXME: pulling S16 as GL_SHORT doesn't work in general, maybe
          // it's the right mode if the internal format is an integer format
          // but for some reason storing as I16 doesn't work either and only
          // gives garbage pull results
          opengl_data->pull.type = GL_SHORT;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (int16_t);
        }

        opengl_data->pull.byte_stride
            = ROUND_UP_4 (width * opengl_data->pull.bytes_per_texel);
        opengl_data->pull.texel_stride
            = width;/*opengl_data->pull.byte_stride
            / opengl_data->pull.bytes_per_texel*/;

        //printf ("byte_stride %i texel_stride %i\n",
        //    opengl_data->pull.byte_stride, opengl_data->pull.texel_stride);

        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_PIXELBUFFER)) {
          create_push_pixelbuffers = TRUE;
        }

        break;
      default:
        SCHRO_ASSERT (0);
        break;
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    start = schro_utils_get_time ();
#endif

    /* textures */
    for (k = 0; k < 2; ++k) {
      glGenTextures (1, &opengl_data->texture.handles[k]);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[k]);
      glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0,
          opengl_data->texture.internal_format, width, height, 0,
          opengl_data->texture.pixel_format, opengl_data->texture.type, NULL);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
          GL_NEAREST);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
          GL_NEAREST);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri (GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexEnvi (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("tex %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* framebuffers */
    for (k = 0; k < 2; ++k) {
      glGenFramebuffersEXT (1, &opengl_data->framebuffers[k]);
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffers[k]);
      glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
          GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handles[k], 0);
      glDrawBuffer (GL_COLOR_ATTACHMENT0_EXT);
      glReadBuffer (GL_COLOR_ATTACHMENT0_EXT);

      SCHRO_OPENGL_CHECK_ERROR
      // FIXME: checking framebuffer status is an expensive operation
      SCHRO_OPENGL_CHECK_FRAMEBUFFER
    }

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("fbo %f", end - start);
    start = schro_utils_get_time ();
#endif

    SCHRO_ASSERT (height >= SCHRO_OPENGL_FRAME_PIXELBUFFERS);

    /* push pixelbuffers */
    if (create_push_pixelbuffers) {
      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
        SCHRO_ASSERT (opengl_data->push.pixelbuffers[k] == 0);

        if (k == SCHRO_OPENGL_FRAME_PIXELBUFFERS - 1) {
          opengl_data->push.heights[k]
              = height - (height / SCHRO_OPENGL_FRAME_PIXELBUFFERS) * k;
        } else {
          opengl_data->push.heights[k]
              = height / SCHRO_OPENGL_FRAME_PIXELBUFFERS;
        }

        glGenBuffersARB (1, &opengl_data->push.pixelbuffers[k]);
        glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB,
            opengl_data->push.pixelbuffers[k]);
        glBufferDataARB (GL_PIXEL_UNPACK_BUFFER_ARB,
            opengl_data->push.byte_stride * opengl_data->push.heights[k],
            NULL, GL_STREAM_DRAW_ARB);

        SCHRO_OPENGL_CHECK_ERROR
      }

      glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    }

    /* pull pixelbuffers */
    if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_PIXELBUFFER)) {
      for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
        SCHRO_ASSERT (opengl_data->pull.pixelbuffers[k] == 0);

        if (k == SCHRO_OPENGL_FRAME_PIXELBUFFERS - 1) {
          opengl_data->pull.heights[k]
              = height - (height / SCHRO_OPENGL_FRAME_PIXELBUFFERS) * k;
        } else {
          opengl_data->pull.heights[k]
              = height / SCHRO_OPENGL_FRAME_PIXELBUFFERS;
        }

        glGenBuffersARB (1, &opengl_data->pull.pixelbuffers[k]);
        glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB,
            opengl_data->pull.pixelbuffers[k]);
        glBufferDataARB (GL_PIXEL_PACK_BUFFER_ARB,
            opengl_data->pull.byte_stride * opengl_data->pull.heights[k], NULL,
            GL_STATIC_READ_ARB);

        SCHRO_OPENGL_CHECK_ERROR
      }

      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, 0);
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("pbo %f", end - start);
#endif
  }

  schro_opengl_unlock (opengl);
}

void
schro_opengl_frame_cleanup (SchroFrame *frame)
{
  int i, k;
  int components;
  SchroOpenGL *opengl;
  SchroOpenGLFrameData *opengl_data;
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
  double start, end;
#endif

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;
  opengl_data = (SchroOpenGLFrameData *) frame->components[0].data;

  SCHRO_ASSERT (opengl_data != NULL);

  opengl = opengl_data->opengl;

  schro_opengl_lock (opengl);

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    opengl_data = (SchroOpenGLFrameData *) frame->components[i].data;

    SCHRO_ASSERT (opengl_data != NULL);
    SCHRO_ASSERT (opengl_data->opengl == opengl);

    opengl_data->opengl = NULL;

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    start = schro_utils_get_time ();
#endif

    /* textures */
    for (k = 0; k < 2; ++k) {
      if (opengl_data->texture.handles[k]) {
        glDeleteTextures (1, &opengl_data->texture.handles[k]);

        opengl_data->texture.handles[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("tex %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* framebuffers */
    for (k = 0; k < 2; ++k) {
      if (opengl_data->framebuffers[k]) {
        glDeleteFramebuffersEXT (1, &opengl_data->framebuffers[k]);

        opengl_data->framebuffers[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("fbo %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* pixelbuffers */
    for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
      if (opengl_data->push.pixelbuffers[k]) {
        glDeleteBuffersARB (1, &opengl_data->push.pixelbuffers[k]);

        opengl_data->push.pixelbuffers[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }

      if (opengl_data->pull.pixelbuffers[k]) {
        glDeleteBuffersARB (1, &opengl_data->pull.pixelbuffers[k]);

        opengl_data->pull.pixelbuffers[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("pbo %f", end - start);
#endif
  }

  schro_opengl_unlock (opengl);
}

SchroFrame *
schro_opengl_frame_new (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrameFormat format, int width,
    int height)
{
  SchroFrame *opengl_frame;

  SCHRO_ASSERT (opengl_domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL);

  opengl_frame = schro_frame_new_and_alloc (opengl_domain, format, width,
      height);

  schro_opengl_frame_setup (opengl, opengl_frame);

  return opengl_frame;
}

SchroFrame *
schro_opengl_frame_clone (SchroFrame *opengl_frame)
{
  SchroOpenGL *opengl;
  SchroOpenGLFrameData *opengl_data;

  SCHRO_ASSERT (opengl_frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (opengl_frame));

  opengl_data = (SchroOpenGLFrameData *) opengl_frame->components[0].data;

  SCHRO_ASSERT (opengl_data != NULL);

  opengl = opengl_data->opengl;

  return schro_opengl_frame_new (opengl, opengl_frame->domain,
      opengl_frame->format, opengl_frame->width, opengl_frame->height);
}

SchroFrame *
schro_opengl_frame_clone_and_push (SchroOpenGL *opengl,
    SchroMemoryDomain *opengl_domain, SchroFrame *cpu_frame)
{
  SchroFrame *opengl_frame;

  SCHRO_ASSERT (opengl_domain->flags & SCHRO_MEMORY_DOMAIN_OPENGL);
  SCHRO_ASSERT (!SCHRO_FRAME_IS_OPENGL (cpu_frame));

  opengl_frame = schro_frame_clone (opengl_domain, cpu_frame);

  schro_opengl_frame_setup (opengl, opengl_frame);
  schro_opengl_frame_push (opengl_frame, cpu_frame);

  return opengl_frame;
}

void
schro_opengl_frame_inverse_iwt_transform (SchroFrame *frame,
    SchroParams *params)
{
  int i;
  int width, height;
  int level;
  SchroOpenGL *opengl;
  SchroOpenGLFrameData *opengl_data;

  opengl_data = (SchroOpenGLFrameData *) frame->components[0].data;

  SCHRO_ASSERT (opengl_data != NULL);

  opengl = opengl_data->opengl;

  schro_opengl_lock (opengl);

  for (i = 0; i < 3; ++i) {
    opengl_data = (SchroOpenGLFrameData *) frame->components[i].data;

    SCHRO_ASSERT (opengl_data->opengl == opengl);

    if (i == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }

    /* FIXME: vertical deinterleave subbands here until there is an option to
              get non-interleaved subbands, or the filtering is changed to work
              together with vertical interleaved subbands */
    for (level = 0; level < params->transform_depth; ++level) {
      SchroFrameData frame_data;

      frame_data.format = frame->format;
      frame_data.data = frame->components[i].data;
      frame_data.width = width >> level;
      frame_data.height = height >> level;
      frame_data.stride = frame->components[i].stride << level;

      schro_opengl_wavelet_vertical_deinterleave (&frame_data);
    }

    for (level = params->transform_depth - 1; level >= 0; --level) {
      SchroFrameData frame_data;

      frame_data.format = frame->format;
      frame_data.data = frame->components[i].data;
      frame_data.width = width >> level;
      frame_data.height = height >> level;
      frame_data.stride = frame->components[i].stride << level;

      schro_opengl_wavelet_inverse_transform (&frame_data,
          params->wavelet_filter_index);
    }
  }

  schro_opengl_unlock (opengl);
}

static void
schro_opengl_upsampled_frame_render_quad (SchroOpenGLShader *shader, int x,
    int y, int quad_width, int quad_height, int total_width, int total_height)
{
  int x_inverse, y_inverse;
  int four_x = 0, four_y = 0, three_x = 0, three_y = 0, two_x = 0, two_y = 0,
      one_x = 0, one_y = 0;

  x_inverse = total_width - x - quad_width;
  y_inverse = total_height - y - quad_height;

  if (quad_width == total_width && quad_height < total_height) {
    four_y = 4;
    three_y = 3;
    two_y = 2;
    one_y = 1;
  } else if (quad_width < total_width && quad_height == total_height) {
    four_x = 4;
    three_x = 3;
    two_x = 2;
    one_x = 1;
  } else {
    SCHRO_ERROR ("invalid quad to total relation");
    SCHRO_ASSERT (0);
  }

  SCHRO_ASSERT (x_inverse >= 0);
  SCHRO_ASSERT (y_inverse >= 0);

  #define UNIFORM(_number, _operation, __x, __y) \
      do { \
        if (shader->_number##_##_operation != -1) { \
          glUniform2fARB (shader->_number##_##_operation, \
              __x < _number##_x ? __x : _number##_x, \
              __y < _number##_y ? __y : _number##_y); \
        } \
      } while (0)

  UNIFORM (four, decrease, x, y);
  UNIFORM (three, decrease, x, y);
  UNIFORM (two, decrease, x, y);
  UNIFORM (one, decrease, x, y);
  UNIFORM (one, increase, x_inverse, y_inverse);
  UNIFORM (two, increase, x_inverse, y_inverse);
  UNIFORM (three, increase, x_inverse, y_inverse);
  UNIFORM (four, increase, x_inverse, y_inverse);

  #undef UNIFORM

  schro_opengl_render_quad (x, y, quad_width, quad_height);
}

void
schro_opengl_upsampled_frame_upsample (SchroUpsampledFrame *upsampled_frame)
{
  int i;
  int width, height;
  SchroOpenGLFrameData *opengl_data[4];
  SchroOpenGL *opengl;
  SchroOpenGLShader *shader = NULL;

  SCHRO_ASSERT (upsampled_frame->frames[0] != NULL);
  SCHRO_ASSERT (upsampled_frame->frames[1] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[2] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[3] == NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (upsampled_frame->frames[0]));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_PACKED (upsampled_frame->frames[0]->format));

  opengl_data[0] = (SchroOpenGLFrameData *) upsampled_frame->frames[0]->components[0].data;

  SCHRO_ASSERT (opengl_data != NULL);

  opengl = opengl_data[0]->opengl;

  schro_opengl_lock (opengl);

  upsampled_frame->frames[1] = schro_opengl_frame_clone (upsampled_frame->frames[0]);
  upsampled_frame->frames[2] = schro_opengl_frame_clone (upsampled_frame->frames[0]);
  upsampled_frame->frames[3] = schro_opengl_frame_clone (upsampled_frame->frames[0]);

  shader = schro_opengl_shader_get (opengl, SCHRO_OPENGL_SHADER_UPSAMPLE_U8);

  SCHRO_ASSERT (shader != NULL);

  glUseProgramObjectARB (shader->program);
  glUniform1iARB (shader->textures[0], 0);

  SCHRO_OPENGL_CHECK_ERROR

  for (i = 0; i < 3; ++i) {
    // FIXME: hack to store custom data per frame component
    opengl_data[0] = (SchroOpenGLFrameData *) upsampled_frame->frames[0]->components[i].data;
    opengl_data[1] = (SchroOpenGLFrameData *) upsampled_frame->frames[1]->components[i].data;
    opengl_data[2] = (SchroOpenGLFrameData *) upsampled_frame->frames[2]->components[i].data;
    opengl_data[3] = (SchroOpenGLFrameData *) upsampled_frame->frames[3]->components[i].data;

    width = upsampled_frame->frames[0]->components[i].width;
    height = upsampled_frame->frames[0]->components[i].height;

    SCHRO_ASSERT (width >= 2);
    SCHRO_ASSERT (height >= 2);
    SCHRO_ASSERT (width % 2 == 0);
    SCHRO_ASSERT (height % 2 == 0);

    schro_opengl_setup_viewport (width, height);

    /* horizontal filter 0 -> 1 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data[1]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data[0]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_HORIZONTAL(_x, _quad_width) \
        schro_opengl_upsampled_frame_render_quad (shader, _x, 0,  _quad_width,\
            height, width, height)

    RENDER_QUAD_HORIZONTAL (0, 1);

    if (width > 2) {
      RENDER_QUAD_HORIZONTAL (1, 1);

      if (width > 4) {
        RENDER_QUAD_HORIZONTAL (2, 1);

        if (width > 6) {
          RENDER_QUAD_HORIZONTAL (3, 1);

           if (width > 8) {
             RENDER_QUAD_HORIZONTAL (4, width - 8);
           }

           RENDER_QUAD_HORIZONTAL (width - 4, 1);
        }

        RENDER_QUAD_HORIZONTAL (width - 3, 1);
      }

      RENDER_QUAD_HORIZONTAL (width - 2, 1);
    }

    RENDER_QUAD_HORIZONTAL (width - 1, 1);

    #undef RENDER_QUAD_HORIZONTAL

    /* vertical filter 0 -> 2 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data[2]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data[0]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_VERTICAL(_y, _quad_height) \
        schro_opengl_upsampled_frame_render_quad (shader, 0, _y,  width,\
            _quad_height, width, height)

    RENDER_QUAD_VERTICAL (0, 1);

    if (height > 2) {
      RENDER_QUAD_VERTICAL (1, 1);

      if (height > 4) {
        RENDER_QUAD_VERTICAL (2, 1);

        if (height > 6) {
          RENDER_QUAD_VERTICAL (3, 1);

           if (height > 8) {
             RENDER_QUAD_VERTICAL (4, height - 8);
           }

           RENDER_QUAD_VERTICAL (height - 4, 1);
        }

        RENDER_QUAD_VERTICAL (height - 3, 1);
      }

      RENDER_QUAD_VERTICAL (height - 2, 1);
    }

    RENDER_QUAD_VERTICAL (height - 1, 1);

    #undef RENDER_QUAD_VERTICAL

    /* horizontal filter 2 -> 3 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data[3]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data[2]->texture.handles[0]);

    SCHRO_OPENGL_CHECK_ERROR

    #define RENDER_QUAD_HORIZONTAL(_x, _quad_width) \
        schro_opengl_upsampled_frame_render_quad (shader, _x, 0,  _quad_width,\
            height, width, height)

    RENDER_QUAD_HORIZONTAL (0, 1);

    if (width > 2) {
      RENDER_QUAD_HORIZONTAL (1, 1);

      if (width > 4) {
        RENDER_QUAD_HORIZONTAL (2, 1);

        if (width > 6) {
          RENDER_QUAD_HORIZONTAL (3, 1);

           if (width > 8) {
             RENDER_QUAD_HORIZONTAL (4, width - 8);
           }

           RENDER_QUAD_HORIZONTAL (width - 4, 1);
        }

        RENDER_QUAD_HORIZONTAL (width - 3, 1);
      }

      RENDER_QUAD_HORIZONTAL (width - 2, 1);
    }

    RENDER_QUAD_HORIZONTAL (width - 1, 1);

    #undef RENDER_QUAD_HORIZONTAL
  }

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
  glUseProgramObjectARB (0);

  schro_opengl_unlock (opengl);
}

void
schro_frame_print (SchroFrame *frame, const char* name)
{
  printf ("schro_frame_print: %s\n", name);

  switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      printf ("  depth:  U8\n");
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      printf ("  depth:  S16\n");
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      printf ("  depth:  S32\n");
      break;
    default:
      printf ("  depth:  unknown\n");
      break;
  }

  printf ("  packed: %s\n", SCHRO_FRAME_IS_PACKED (frame->format) ? "yes": "no");
  printf ("  width:  %i\n", frame->width);
  printf ("  height: %i\n", frame->height);
  printf ("  opengl: %s\n", SCHRO_FRAME_IS_OPENGL (frame) ? "yes": "no");
}

