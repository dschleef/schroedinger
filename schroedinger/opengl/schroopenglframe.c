
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglextensions.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <liboil/liboil.h>
#include <stdio.h>

unsigned int _schro_opengl_frame_flags
    = 0
    //| SCHRO_OPENGL_FRAME_STORE_RGBA
    //| SCHRO_OPENGL_FRAME_STORE_BGRA
    //| SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8
    //| SCHRO_OPENGL_FRAME_STORE_U8_AS_F32
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_I16
    //| SCHRO_OPENGL_FRAME_STORE_S16_AS_F32

    //| SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD
    //| SCHRO_OPENGL_FRAME_PUSH_SHADER
    //| SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS /* currently broken */
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
  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA
      && !(_schro_opengl_extensions & SCHRO_OPENGL_EXTENSION_BGRA)) {
    SCHRO_ERROR ("no BGRA extension, disabling BGRA storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_BGRA;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
    SCHRO_ERROR ("can't store in RGBA and BGRA, disabling BGRA storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_BGRA;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_F32
      || _schro_opengl_frame_flags
      & SCHRO_OPENGL_FRAME_STORE_S16_AS_F32) {
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
        || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
      if (!(_schro_opengl_extensions & SCHRO_OPENGL_EXTENSION_TEXTURE_FLOAT)) {
        SCHRO_ERROR ("no texturefloat extension, can't store U8/S16 as F32 in "
            "RGBA/BRGA mode, disabling U8/S16 as F32 storing");
        _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_U8_AS_F32;
        _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_F32;
      }
    } else {
      if (!(_schro_opengl_extensions
          & SCHRO_OPENGL_EXTENSION_NVIDIA_FLOAT_BUFFER)) {
        SCHRO_ERROR ("no NVIDIA floatbuffer extension, can't store U8/S16 as "
            "F32, disabling U8/S16 as F32 storing");
        _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_U8_AS_F32;
        _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_F32;
      }
    }
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8
      || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
      || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
    if (!(_schro_opengl_extensions
        & SCHRO_OPENGL_EXTENSION_TEXTURE_INTEGER)) {
      SCHRO_ERROR ("no texture integer extension, can't store U8/S16 as "
          "UI8/UI16/I16, disabling U8/S16 as UI8/UI16/I16 storing");
      _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8;
      _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16;
      _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_I16;
    }
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_F32
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8) {
    SCHRO_ERROR ("can't store U8 in F32 and UI8, disabling F32 storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_U8_AS_F32;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_F32
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16) {
    SCHRO_ERROR ("can't store S16 in F32 and UI16, disabling F32 storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_F32;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_F32
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
    SCHRO_ERROR ("can't store S16 in F32 and I16, disabling F32 storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_F32;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
    SCHRO_ERROR ("can't store S16 in UI16 and I16, disabling UI16 storing");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16;
  }

  /* push */
  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS) {
    SCHRO_ERROR ("can't render quad and drawpixels, disabling drawpixels");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS
      && !(_schro_opengl_extensions
      & SCHRO_OPENGL_EXTENSION_WINDOW_POSITION)) {
    SCHRO_ERROR ("no window position extension, disabling drawpixels");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER
      && !(_schro_opengl_extensions & SCHRO_OPENGL_EXTENSION_PIXELBUFFER)) {
    SCHRO_ERROR ("no pixelbuffer extension, disabling U8 pixelbuffer push");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER
      && !(_schro_opengl_extensions & SCHRO_OPENGL_EXTENSION_PIXELBUFFER)) {
    SCHRO_ERROR ("no pixelbuffer extension, disabling S16 pixelbuffer push");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32) {
    SCHRO_ERROR ("can't push S16 as U16 and F32, disabling U16 push");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16;
  }

  /* pull */
  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER
      && !(_schro_opengl_extensions & SCHRO_OPENGL_EXTENSION_PIXELBUFFER)) {
    SCHRO_ERROR ("no pixelbuffer extension, disabling pixelbuffer pull");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER;
  }

  if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_S16_AS_U16
      && _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_S16_AS_F32) {
    SCHRO_ERROR ("can't pull S16 as U16 and F32, disabling U16 pull");
    _schro_opengl_frame_flags &= ~SCHRO_OPENGL_FRAME_PULL_S16_AS_U16;
  }
}

void
schro_opengl_frame_print_flags (const char* indent)
{
  schro_opengl_frame_check_flags ();

  #define PRINT_FLAG(text, flag) \
      printf ("%s  "text"%s\n", indent, \
          _schro_opengl_frame_flags & (flag) ? "on" : "off")

  printf ("%sstore flags\n", indent);

  PRINT_FLAG ("RGBA:            ", SCHRO_OPENGL_FRAME_STORE_RGBA);
  PRINT_FLAG ("BGRA:            ", SCHRO_OPENGL_FRAME_STORE_BGRA);
  PRINT_FLAG ("U8 as UI8:       ", SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8);
  PRINT_FLAG ("U8 as F32:       ", SCHRO_OPENGL_FRAME_STORE_U8_AS_F32);
  PRINT_FLAG ("S16 as UI16:     ", SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16);
  PRINT_FLAG ("S16 as I16:      ", SCHRO_OPENGL_FRAME_STORE_S16_AS_I16);
  PRINT_FLAG ("S16 as F32:      ", SCHRO_OPENGL_FRAME_STORE_S16_AS_F32);

  printf ("%spush flags\n", indent);

  PRINT_FLAG ("render quad:     ", SCHRO_OPENGL_FRAME_PUSH_RENDER_QUAD);
  PRINT_FLAG ("shader:          ", SCHRO_OPENGL_FRAME_PUSH_SHADER);
  PRINT_FLAG ("drawpixels:      ", SCHRO_OPENGL_FRAME_PUSH_DRAWPIXELS);
  PRINT_FLAG ("U8 pixelbuffer:  ", SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32);
  PRINT_FLAG ("S16 pixelbuffer: ", SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER);
  PRINT_FLAG ("S16 as U16:      ", SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32);

  printf ("%spull flags\n", indent);

  PRINT_FLAG ("pixelbuffer:     ", SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER);
  PRINT_FLAG ("U8 as F32:       ", SCHRO_OPENGL_FRAME_PULL_U8_AS_F32);
  PRINT_FLAG ("S16 as U16:      ", SCHRO_OPENGL_FRAME_PULL_S16_AS_U16);
  PRINT_FLAG ("S16 as F32:      ", SCHRO_OPENGL_FRAME_PULL_S16_AS_F32);

  #undef PRINT_FLAG
}

void
schro_opengl_frame_setup (SchroFrame *frame)
{
  int i, k;
  int width, height;
  int create_push_pixelbuffers = FALSE;
  SchroOpenGLFrameData *opengl_data = NULL;
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
  double start, end;
#endif

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_PACKED (frame->format)); // FIXME: unimplemented

  schro_opengl_lock ();

  schro_opengl_frame_check_flags ();

  for (i = 0; i < 3; ++i) {
    width = frame->components[i].width;
    height = frame->components[i].height;

    // FIXME: hack to store custom data per frame component
    opengl_data = (SchroOpenGLFrameData *) frame->components[i].data;

    SCHRO_ASSERT (opengl_data != NULL);
    SCHRO_ASSERT (opengl_data->texture.handle == 0);
    SCHRO_ASSERT (opengl_data->framebuffer == 0);

    switch (SCHRO_FRAME_FORMAT_DEPTH (frame->format)) {
      case SCHRO_FRAME_FORMAT_DEPTH_U8:
        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_F32) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
              || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
            opengl_data->texture.internal_format = GL_RGBA32F_ARB;
          } else {
            opengl_data->texture.internal_format = GL_FLOAT_R32_NV;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
              || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
            opengl_data->texture.internal_format = GL_RGBA8UI_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA8UI_EXT;
          }

          opengl_data->texture.type = GL_UNSIGNED_BYTE;
        } else {
          opengl_data->texture.internal_format = GL_RGBA8;
          opengl_data->texture.type = GL_UNSIGNED_BYTE;
        }

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8) {
            opengl_data->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RGBA;
          }

          opengl_data->texture.components = 4;
        } else if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8) {
            opengl_data->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_BGRA;
          }

          opengl_data->texture.components = 4;
        } else {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_U8_AS_UI8) {
            opengl_data->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RED;
          }

          opengl_data->texture.components = 1;
        }

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_U8_AS_F32) {
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

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_U8_AS_F32) {
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

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_U8_PIXELBUFFER)
          create_push_pixelbuffers = TRUE;

        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S16:
        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_F32) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
              || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
            opengl_data->texture.internal_format = GL_RGBA32F_ARB;
          } else {
            opengl_data->texture.internal_format = GL_FLOAT_R32_NV;
          }

          opengl_data->texture.type = GL_FLOAT;
        } else if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
              || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
            opengl_data->texture.internal_format = GL_RGBA16UI_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA16UI_EXT;
          }

          opengl_data->texture.type = GL_UNSIGNED_SHORT;
        } else if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA
              || _schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
            opengl_data->texture.internal_format = GL_RGBA16I_EXT;
          } else {
            opengl_data->texture.internal_format = GL_ALPHA16I_EXT;
          }

          opengl_data->texture.type = GL_SHORT;
        } else {
          opengl_data->texture.internal_format = GL_RGBA16;
          opengl_data->texture.type = GL_SHORT;
        }

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_RGBA) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
              || _schro_opengl_frame_flags
              & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
            opengl_data->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RGBA;
          }

          opengl_data->texture.components = 4;
        } else if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_BGRA) {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
              || _schro_opengl_frame_flags
              & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
            opengl_data->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_BGRA;
          }

          opengl_data->texture.components = 4;
        } else {
          if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_STORE_S16_AS_UI16
              || _schro_opengl_frame_flags
              & SCHRO_OPENGL_FRAME_STORE_S16_AS_I16) {
            opengl_data->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
          } else {
            opengl_data->texture.pixel_format = GL_RED;
          }

          opengl_data->texture.components = 1;
        }

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PUSH_S16_AS_U16) {
          opengl_data->push.type = GL_UNSIGNED_SHORT;
          opengl_data->push.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint16_t);
        } else if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_PUSH_S16_AS_F32) {
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

        if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_S16_AS_U16) {
          /* must pull S16 as GL_UNSIGNED_SHORT instead of GL_SHORT because
             the OpenGL mapping form internal float represenation into S16
             values with GL_SHORT maps 0.0 to 0 and 1.0 to 32767 clamping all
             negative values to 0, see glReadPixel documentation. so the pull
             is done with GL_UNSIGNED_SHORT and the resulting U16 values are
             manually shifted to S16 */
          opengl_data->pull.type = GL_UNSIGNED_SHORT;
          opengl_data->pull.bytes_per_texel
              = opengl_data->texture.components * sizeof (uint16_t);
        } else if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_PULL_S16_AS_F32) {
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

        if (_schro_opengl_frame_flags
            & SCHRO_OPENGL_FRAME_PUSH_S16_PIXELBUFFER)
          create_push_pixelbuffers = TRUE;

        break;
      default:
        SCHRO_ASSERT (0);
        break;
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    start = schro_utils_get_time ();
#endif

    /* texture */
    glGenTextures (1, &opengl_data->texture.handle);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handle);
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
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);

    SCHRO_OPENGL_CHECK_ERROR

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("tex %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* framebuffer */
    glGenFramebuffersEXT (1, &opengl_data->framebuffer);
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, opengl_data->framebuffer);
    glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,
        GL_TEXTURE_RECTANGLE_ARB, opengl_data->texture.handle, 0);
    glDrawBuffer (GL_COLOR_ATTACHMENT1_EXT);
    glReadBuffer (GL_COLOR_ATTACHMENT1_EXT);

    SCHRO_OPENGL_CHECK_ERROR
    // FIXME: checking framebuffer status is an expensive operation
    SCHRO_OPENGL_CHECK_FRAMEBUFFER

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
    if (_schro_opengl_frame_flags & SCHRO_OPENGL_FRAME_PULL_PIXELBUFFER) {
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
            opengl_data->pull.byte_stride * opengl_data->pull.heights[k],
            NULL, GL_STATIC_READ_ARB);

        SCHRO_OPENGL_CHECK_ERROR
      }

      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, 0);
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("pbo %f", end - start);
#endif
  }

  schro_opengl_unlock ();
}

void
schro_opengl_frame_cleanup (SchroFrame *frame)
{
  int i, k;
  SchroOpenGLFrameData *opengl_data;
#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
  double start, end;
#endif

  //SCHRO_ASSERT (schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_OPENGL);
  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  schro_opengl_lock ();

  for (i = 0; i < 3; ++i) {
    // FIXME: hack to store custom data per frame component
    opengl_data = (SchroOpenGLFrameData *) frame->components[i].data;

    SCHRO_ASSERT (opengl_data != NULL);

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    start = schro_utils_get_time ();
#endif

    /* texture */
    if (opengl_data->texture.handle) {
      glDeleteTextures (1, &opengl_data->texture.handle);

      opengl_data->texture.handle = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("tex %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* framebuffer */
    if (opengl_data->framebuffer) {
      glDeleteFramebuffersEXT (1, &opengl_data->framebuffer);

      opengl_data->framebuffer = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("fbo %f", end - start);
    start = schro_utils_get_time ();
#endif

    /* pixelbuffers */
    for (k = 0; k < SCHRO_OPENGL_FRAME_PIXELBUFFERS; ++k) {
      if (opengl_data->push.pixelbuffers[k]) {
        glDeleteBuffersARB(1, &opengl_data->push.pixelbuffers[k]);

        opengl_data->push.pixelbuffers[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }

      if (opengl_data->pull.pixelbuffers[k]) {
        glDeleteBuffersARB(1, &opengl_data->pull.pixelbuffers[k]);

        opengl_data->pull.pixelbuffers[k] = 0;

        SCHRO_OPENGL_CHECK_ERROR
      }
    }

#ifdef OPENGL_INTERNAL_TIME_MEASUREMENT
    end = schro_utils_get_time ();
    SCHRO_INFO ("pbo %f", end - start);
#endif
  }

  schro_opengl_unlock ();
}

