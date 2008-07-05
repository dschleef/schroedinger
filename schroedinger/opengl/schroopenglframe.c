
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
  int i;
  int components;
  int width, height;
  SchroFrameFormat format;
  SchroOpenGLCanvasPool *canvas_pool;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;

  schro_opengl_lock (opengl);

  canvas_pool = schro_opengl_get_canvas_pool (opengl);

  schro_opengl_frame_check_flags ();

  for (i = 0; i < components; ++i) {
    format = frame->components[i].format;
    width = frame->components[i].width;
    height = frame->components[i].height;

    SCHRO_ASSERT (frame->format == format);

    if (!schro_opengl_canvas_pool_is_empty (canvas_pool)) {
      canvas = schro_opengl_canvas_pool_pull (canvas_pool, format, width,
          height);
    } else {
      canvas = NULL;
    }

    if (!canvas) {
      canvas = schro_opengl_canvas_new (opengl, format, width, height);
    }

    // FIXME: hack to store custom data per frame component
    *((SchroOpenGLCanvas **) frame->components[i].data) = canvas;
  }

  schro_opengl_unlock (opengl);
}

void
schro_opengl_frame_cleanup (SchroFrame *frame)
{
  int i;
  int components;
  SchroOpenGL *opengl;
  SchroOpenGLCanvasPool *canvas_pool;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (frame));

  components = SCHRO_FRAME_IS_PACKED (frame->format) ? 1 : 3;
  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  canvas_pool = schro_opengl_get_canvas_pool (opengl);

  for (i = 0; i < components; ++i) {
    // FIXME: hack to store custom data per frame component
    canvas = *((SchroOpenGLCanvas **) frame->components[i].data);

    SCHRO_ASSERT (canvas != NULL);
    SCHRO_ASSERT (canvas->opengl == opengl);

    if (!schro_opengl_canvas_pool_is_full (canvas_pool)) {
      schro_opengl_canvas_pool_push (canvas_pool, canvas);
    } else {
      schro_opengl_canvas_free (canvas);
    }
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
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (opengl_frame != NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (opengl_frame));

  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) opengl_frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  return schro_opengl_frame_new (canvas->opengl, opengl_frame->domain,
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
  SchroOpenGLCanvas *canvas;

  // FIXME: hack to store custom data per frame component
  canvas = *((SchroOpenGLCanvas **) frame->components[0].data);

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;

  schro_opengl_lock (opengl);

  for (i = 0; i < 3; ++i) {
    // FIXME: hack to store custom data per frame component
    canvas = *((SchroOpenGLCanvas **) frame->components[i].data);

    SCHRO_ASSERT (canvas != NULL);
    SCHRO_ASSERT (canvas->opengl == opengl);

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
  SchroOpenGLCanvas *canvases[4];
  SchroOpenGL *opengl;
  SchroOpenGLShader *shader = NULL;

  SCHRO_ASSERT (upsampled_frame->frames[0] != NULL);
  SCHRO_ASSERT (upsampled_frame->frames[1] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[2] == NULL);
  SCHRO_ASSERT (upsampled_frame->frames[3] == NULL);
  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (upsampled_frame->frames[0]));
  SCHRO_ASSERT (!SCHRO_FRAME_IS_PACKED (upsampled_frame->frames[0]->format));

  // FIXME: hack to store custom data per frame component
  canvases[0] = *((SchroOpenGLCanvas **) upsampled_frame->frames[0]->components[0].data);

  SCHRO_ASSERT (canvases[0] != NULL);

  opengl = canvases[0]->opengl;

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
    canvases[0] = *((SchroOpenGLCanvas **) upsampled_frame->frames[0]->components[i].data);
    canvases[1] = *((SchroOpenGLCanvas **) upsampled_frame->frames[1]->components[i].data);
    canvases[2] = *((SchroOpenGLCanvas **) upsampled_frame->frames[2]->components[i].data);
    canvases[3] = *((SchroOpenGLCanvas **) upsampled_frame->frames[3]->components[i].data);

    SCHRO_ASSERT (canvases[0] != NULL);
    SCHRO_ASSERT (canvases[1] != NULL);
    SCHRO_ASSERT (canvases[2] != NULL);
    SCHRO_ASSERT (canvases[3] != NULL);
    SCHRO_ASSERT (canvases[0]->opengl == opengl);
    SCHRO_ASSERT (canvases[1]->opengl == opengl);
    SCHRO_ASSERT (canvases[2]->opengl == opengl);
    SCHRO_ASSERT (canvases[3]->opengl == opengl);

    width = upsampled_frame->frames[0]->components[i].width;
    height = upsampled_frame->frames[0]->components[i].height;

    SCHRO_ASSERT (width >= 2);
    SCHRO_ASSERT (height >= 2);
    SCHRO_ASSERT (width % 2 == 0);
    SCHRO_ASSERT (height % 2 == 0);

    schro_opengl_setup_viewport (width, height);

    /* horizontal filter 0 -> 1 */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[1]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[0]->texture.handles[0]);

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
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[2]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[0]->texture.handles[0]);

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
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvases[3]->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvases[2]->texture.handles[0]);

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

SchroOpenGLCanvas *
schro_opengl_canvas_new (SchroOpenGL *opengl, SchroFrameFormat format,
    int width, int height)
{
  int i;
  int create_push_pixelbuffers = FALSE;
  SchroOpenGLCanvas *canvas = schro_malloc0 (sizeof (SchroOpenGLCanvas));

  schro_opengl_frame_check_flags (); // FIXME

  schro_opengl_lock (opengl);

  canvas->opengl = opengl;
  canvas->format = format;
  canvas->width = width;
  canvas->height = height;

  SCHRO_ERROR ("+++++++++++++++++++++++++++");

  switch (SCHRO_FRAME_FORMAT_DEPTH (format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F16)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R16_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA16F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_F32)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R32_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA32F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA8UI_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA8UI_EXT;
        }

        canvas->texture.type = GL_UNSIGNED_BYTE;
      } else {
        /* must use RGBA format here, because other formats are in general
           not supported by framebuffers */
        canvas->texture.internal_format = GL_RGBA8;
        canvas->texture.type = GL_UNSIGNED_BYTE;
      }

      if (SCHRO_FRAME_IS_PACKED (format)) {
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_BGRA)) {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
            canvas->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_BGRA;
          }

          canvas->texture.channels = 4;
        } else {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
            canvas->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_RGBA;
          }

          canvas->texture.channels = 4;
        }
      } else {
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_U8_AS_UI8)) {
          canvas->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
        } else {
          canvas->texture.pixel_format = GL_RED;
        }

        canvas->texture.channels = 1;
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_U8_PIXELBUFFER)) {
        create_push_pixelbuffers = TRUE;
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_U8_AS_F32)) {
        canvas->push.type = GL_FLOAT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->push.type = GL_UNSIGNED_BYTE;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint8_t));
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_U8_AS_F32)) {
        canvas->pull.type = GL_FLOAT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->pull.type = GL_UNSIGNED_BYTE;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint8_t));
      }

      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F16)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R16_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA16F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_F32)) {
        if (!SCHRO_FRAME_IS_PACKED (format) && GLEW_NV_float_buffer) {
          canvas->texture.internal_format = GL_FLOAT_R32_NV;
        } else {
          canvas->texture.internal_format = GL_RGBA32F_ARB;
        }

        canvas->texture.type = GL_FLOAT;
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA16UI_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA16UI_EXT;
        }

        canvas->texture.type = GL_UNSIGNED_SHORT;
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
        if (SCHRO_FRAME_IS_PACKED (format)) {
          canvas->texture.internal_format = GL_RGBA16I_EXT;
        } else {
          canvas->texture.internal_format = GL_ALPHA16I_EXT;
        }

        canvas->texture.type = GL_SHORT;
      } else {
        /* must use RGBA format here, because other formats are in general
           not supported by framebuffers */
        canvas->texture.internal_format = GL_RGBA16;
        canvas->texture.type = GL_SHORT;
      }

      if (SCHRO_FRAME_IS_PACKED (format)) {
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_BGRA)) {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
              SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
            canvas->texture.pixel_format = GL_BGRA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_BGRA;
          }

          canvas->texture.channels = 4;
        } else {
          if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
              SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
            canvas->texture.pixel_format = GL_RGBA_INTEGER_EXT;
          } else {
            canvas->texture.pixel_format = GL_RGBA;
          }

          canvas->texture.channels = 4;
        }
      } else {
        if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_UI16) ||
            SCHRO_OPENGL_FRAME_IS_FLAG_SET (STORE_S16_AS_I16)) {
          canvas->texture.pixel_format = GL_ALPHA_INTEGER_EXT;
        } else {
          canvas->texture.pixel_format = GL_RED;
        }

        canvas->texture.channels = 1;
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_U16)) {
        canvas->push.type = GL_UNSIGNED_SHORT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint16_t));
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_AS_F32)) {
        canvas->push.type = GL_FLOAT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        canvas->push.type = GL_SHORT;
        canvas->push.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (int16_t));
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PUSH_S16_PIXELBUFFER)) {
        create_push_pixelbuffers = TRUE;
      }

      if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_U16)) {
        /* must pull S16 as GL_UNSIGNED_SHORT instead of GL_SHORT because
           the OpenGL mapping form internal float represenation into S16
           values with GL_SHORT maps 0.0 to 0 and 1.0 to 32767 clamping all
           negative values to 0, see glReadPixel documentation. so the pull
           is done with GL_UNSIGNED_SHORT and the resulting U16 values are
           manually shifted to S16 */
        canvas->pull.type = GL_UNSIGNED_SHORT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (uint16_t));
      } else if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_S16_AS_F32)) {
        canvas->pull.type = GL_FLOAT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (float));
      } else {
        // FIXME: pulling S16 as GL_SHORT doesn't work in general, maybe
        // it's the right mode if the internal format is an integer format
        // but for some reason storing as I16 doesn't work either and only
        // gives garbage pull results
        canvas->pull.type = GL_SHORT;
        canvas->pull.stride = ROUND_UP_4 (width * canvas->texture.channels
            * sizeof (int16_t));
      }

      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }

  /* textures */
  for (i = 0; i < 2; ++i) {
    glGenTextures (1, &canvas->texture.handles[i]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, canvas->texture.handles[i]);
    glTexImage2D (GL_TEXTURE_RECTANGLE_ARB, 0,
        canvas->texture.internal_format, width, height, 0,
        canvas->texture.pixel_format, canvas->texture.type, NULL);
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

  /* framebuffers */
  for (i = 0; i < 2; ++i) {
    glGenFramebuffersEXT (1, &canvas->framebuffers[i]);
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, canvas->framebuffers[i]);
    glFramebufferTexture2DEXT (GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
        GL_TEXTURE_RECTANGLE_ARB, canvas->texture.handles[i], 0);
    glDrawBuffer (GL_COLOR_ATTACHMENT0_EXT);
    glReadBuffer (GL_COLOR_ATTACHMENT0_EXT);

    SCHRO_OPENGL_CHECK_ERROR
    // FIXME: checking framebuffer status is an expensive operation
    SCHRO_OPENGL_CHECK_FRAMEBUFFER
  }

  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  SCHRO_ASSERT (height >= SCHRO_OPENGL_TRANSFER_PIXELBUFFERS);

  /* push pixelbuffers */
  if (create_push_pixelbuffers) {
    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      SCHRO_ASSERT (canvas->push.pixelbuffers[i] == 0);

      if (i == SCHRO_OPENGL_TRANSFER_PIXELBUFFERS - 1) {
        canvas->push.heights[i]
            = height - (height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS) * i;
      } else {
        canvas->push.heights[i] = height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS;
      }

      glGenBuffersARB (1, &canvas->push.pixelbuffers[i]);
      glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB,
          canvas->push.pixelbuffers[i]);
      glBufferDataARB (GL_PIXEL_UNPACK_BUFFER_ARB,
          canvas->push.stride * canvas->push.heights[i], NULL,
          GL_STREAM_DRAW_ARB);

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindBufferARB (GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  }

  /* pull pixelbuffers */
  if (SCHRO_OPENGL_FRAME_IS_FLAG_SET (PULL_PIXELBUFFER)) {
    for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
      SCHRO_ASSERT (canvas->pull.pixelbuffers[i] == 0);

      if (i == SCHRO_OPENGL_TRANSFER_PIXELBUFFERS - 1) {
        canvas->pull.heights[i]
            = height - (height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS) * i;
      } else {
        canvas->pull.heights[i] = height / SCHRO_OPENGL_TRANSFER_PIXELBUFFERS;
      }

      glGenBuffersARB (1, &canvas->pull.pixelbuffers[i]);
      glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, canvas->pull.pixelbuffers[i]);
      glBufferDataARB (GL_PIXEL_PACK_BUFFER_ARB,
          canvas->pull.stride * canvas->pull.heights[i], NULL,
          GL_STATIC_READ_ARB);

      SCHRO_OPENGL_CHECK_ERROR
    }

    glBindBufferARB (GL_PIXEL_PACK_BUFFER_ARB, 0);
  }

  schro_opengl_unlock (opengl);

  return canvas;
}

void
schro_opengl_canvas_free (SchroOpenGLCanvas *canvas)
{
  int i;
  SchroOpenGL *opengl;

  SCHRO_ASSERT (canvas != NULL);

  opengl = canvas->opengl;
  canvas->opengl = NULL;

  schro_opengl_lock (opengl);

  SCHRO_ERROR ("-----------------------------");

  /* textures */
  for (i = 0; i < 2; ++i) {
    if (canvas->texture.handles[i]) {
      glDeleteTextures (1, &canvas->texture.handles[i]);

      canvas->texture.handles[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  /* framebuffers */
  for (i = 0; i < 2; ++i) {
    if (canvas->framebuffers[i]) {
      glDeleteFramebuffersEXT (1, &canvas->framebuffers[i]);

      canvas->framebuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  /* pixelbuffers */
  for (i = 0; i < SCHRO_OPENGL_TRANSFER_PIXELBUFFERS; ++i) {
    if (canvas->push.pixelbuffers[i]) {
      glDeleteBuffersARB (1, &canvas->push.pixelbuffers[i]);

      canvas->push.pixelbuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }

    if (canvas->pull.pixelbuffers[i]) {
      glDeleteBuffersARB (1, &canvas->pull.pixelbuffers[i]);

      canvas->pull.pixelbuffers[i] = 0;

      SCHRO_OPENGL_CHECK_ERROR
    }
  }

  schro_opengl_unlock (opengl);

  schro_free (canvas);
}

SchroOpenGLCanvasPool *schro_opengl_canvas_pool_new (void)
{
  SchroOpenGLCanvasPool *canvas_pool;

  canvas_pool = schro_malloc0 (sizeof (SchroOpenGLCanvasPool));

  canvas_pool->size = 0;

  return canvas_pool;
}

void schro_opengl_canvas_pool_free (SchroOpenGLCanvasPool* canvas_pool)
{
  int i;

  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  for (i = 0; i < canvas_pool->size; ++i) {
    schro_opengl_canvas_free (canvas_pool->canvases[i]);
  }

  schro_free (canvas_pool);
}

int
schro_opengl_canvas_pool_is_empty (SchroOpenGLCanvasPool* canvas_pool)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  return canvas_pool->size == 0;
}

int
schro_opengl_canvas_pool_is_full (SchroOpenGLCanvasPool* canvas_pool)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  return canvas_pool->size == SCHRO_OPENGL_CANVAS_POOL_SIZE;
}

SchroOpenGLCanvas *
schro_opengl_canvas_pool_pull (SchroOpenGLCanvasPool* canvas_pool,
    SchroFrameFormat format, int width, int height)
{
  int i;
  SchroOpenGLCanvas *canvas;

  SCHRO_ASSERT (canvas_pool->size >= 1);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE);

  for (i = 0; i < canvas_pool->size; ++i) {
    canvas = canvas_pool->canvases[i];

    if (canvas->format == format && canvas->width == width &&
        canvas->height == height) {
      --canvas_pool->size;

      /* move the last canvas in the pool to the slot of the pulled one to
         maintain the pool continuous in memory */
      canvas_pool->canvases[i] = canvas_pool->canvases[canvas_pool->size];

      return canvas;
    }
  }

  SCHRO_ERROR ("NOT FOUND");

  return NULL;
}

void
schro_opengl_canvas_pool_push (SchroOpenGLCanvasPool* canvas_pool,
    SchroOpenGLCanvas *canvas)
{
  SCHRO_ASSERT (canvas_pool->size >= 0);
  SCHRO_ASSERT (canvas_pool->size <= SCHRO_OPENGL_CANVAS_POOL_SIZE - 1);

  canvas_pool->canvases[canvas_pool->size] = canvas;

  ++canvas_pool->size;
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

