 
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglshader.h>
#include <stdio.h>

typedef struct _SchroOpenGLMotion SchroOpenGLMotion;

struct _SchroOpenGLMotion {
  SchroMotion *motion;
  SchroOpenGLFrameData *src_opengl_data[2][4];
  SchroOpenGLShader *shader_dc;
  SchroOpenGLShader *shader_ref_prec0;
  SchroOpenGLShader *shader_ref_prec0_weight;
  SchroOpenGLShader *shader_ref_prec1;
  GLuint previous_texture;
  GLuint obmc_weight_texture;
};
/*
static*/ void
schro_opengl_motion_render_dc_block (SchroOpenGLMotion *opengl_motion, int i,
    int x, int y, int u, int v)
{
  int xblen, yblen;
  SchroMotion *motion;
  SchroMotionVectorDC *motion_vector_dc;
  uint8_t dc;

  motion = opengl_motion->motion;
  motion_vector_dc = (SchroMotionVectorDC *)
      &motion->motion_vectors[v * motion->params->x_num_blocks + u];
  dc = (int) motion_vector_dc->dc[i] + 128;

  glUseProgramObjectARB (opengl_motion->shader_dc->program);

  glActiveTextureARB (GL_TEXTURE0_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_motion->previous_texture);
  glUniform1iARB (opengl_motion->shader_dc->textures[0], 0);

  glActiveTextureARB (GL_TEXTURE1_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_motion->obmc_weight_texture);
  glUniform1iARB (opengl_motion->shader_dc->textures[1], 1);

  glActiveTextureARB (GL_TEXTURE0_ARB);

  glUniform2fARB (opengl_motion->shader_dc->origin, x, y);
  glUniform1fARB (opengl_motion->shader_dc->dc, dc);

  if (x < 0) {
    xblen = motion->xblen + x;
    x = 0;
  } else {
    xblen = motion->xblen;
  }

  if (y < 0) {
    yblen = motion->yblen + y;
    y = 0;
  } else {
    yblen = motion->yblen;
  }

  schro_opengl_render_quad (x, y, xblen, yblen);
}
/*
static*/ void
schro_opengl_motion_render_ref_block (SchroOpenGLMotion *opengl_motion,
    int i, int x, int y, int u, int v, int ref)
{
  int s, dx, dy, px, py, hx, hy, rx, ry;
  int weight, shift, addend, divisor;
  SchroMotion *motion;
  SchroMotionVector *motion_vector;
  SchroChromaFormat chroma_format;

  motion = opengl_motion->motion;
  motion_vector = &motion->motion_vectors[v * motion->params->x_num_blocks + u];
  chroma_format = motion->params->video_format->chroma_format;

  SCHRO_ASSERT (motion_vector->using_global == FALSE);

  dx = motion_vector->dx[ref];
  dy = motion_vector->dy[ref];

  if (i > 0) {
    dx >>= SCHRO_CHROMA_FORMAT_H_SHIFT (chroma_format);
    dy >>= SCHRO_CHROMA_FORMAT_V_SHIFT (chroma_format);
  }

  px = (x << motion->mv_precision) + dx;
  py = (y << motion->mv_precision) + dy;

  switch (motion->mv_precision) {
    case 0:
      weight = motion->ref1_weight + motion->ref2_weight;
      shift = motion->ref_weight_precision;
      addend = 1 << (shift - 1);
      divisor = 1 << shift;

      if (weight != divisor) {
        glUseProgramObjectARB (opengl_motion->shader_ref_prec0_weight->program);
      } else {
        glUseProgramObjectARB (opengl_motion->shader_ref_prec0->program);
      }

      glActiveTextureARB (GL_TEXTURE0_ARB);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_motion->previous_texture);
      glUniform1iARB (opengl_motion->shader_ref_prec0->textures[0], 0);

      glActiveTextureARB (GL_TEXTURE1_ARB);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_motion->obmc_weight_texture);
      glUniform1iARB (opengl_motion->shader_ref_prec0->textures[1], 1);

      glActiveTextureARB (GL_TEXTURE2_ARB);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
          opengl_motion->src_opengl_data[ref][0]->texture.handles[0]);
      glUniform1iARB (opengl_motion->shader_ref_prec0->textures[2], 2);

      glActiveTextureARB (GL_TEXTURE0_ARB);

      glUniform2fARB (opengl_motion->shader_ref_prec0->offset, px - x, py - y);
      glUniform2fARB (opengl_motion->shader_ref_prec0->origin, x, y);

      if (weight != divisor) {
        glUniform1fARB (opengl_motion->shader_ref_prec0->weight, weight);
        glUniform1fARB (opengl_motion->shader_ref_prec0->addend, addend);
        glUniform1fARB (opengl_motion->shader_ref_prec0->divisor, divisor);
      }

      schro_opengl_render_quad (x, y, motion->xblen, motion->yblen);
      break;
    case 1:
      s = ((py & 1) << 1) | (px & 1);

  SCHRO_ERROR ("1");

      glUseProgramObjectARB (opengl_motion->shader_ref_prec1->program);

      glActiveTextureARB (GL_TEXTURE0_ARB);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
          opengl_motion->obmc_weight_texture);
      glUniform1iARB (opengl_motion->shader_ref_prec1->textures[0], 0);

      glActiveTextureARB (GL_TEXTURE1_ARB);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
          opengl_motion->src_opengl_data[ref][s]->texture.handles[0]);
      glUniform1iARB (opengl_motion->shader_ref_prec1->textures[1], 0);

      glActiveTextureARB (GL_TEXTURE0_ARB);

      glUniform2fARB (opengl_motion->shader_ref_prec1->offset, (px >> 1) - x, (py >> 1) - y);
      glUniform2fARB (opengl_motion->shader_ref_prec1->origin, x, y);

      schro_opengl_render_quad (x, y, motion->xblen, motion->yblen);
      break;
    case 2:
      px <<= 1;
      py <<= 1;
      /* fall through */
    case 3:
      hx = px >> 2;
      hy = py >> 2;
      rx = px & 0x3;
      ry = py & 0x3;

  SCHRO_ERROR ("2/3");

      switch ((ry << 2) | rx) {
        case 0:
          s = ((hy & 1) << 1) | (hx & 1);

          glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
              opengl_motion->src_opengl_data[ref][s]->texture.handles[0]);

          glUseProgramObjectARB (opengl_motion->shader_ref_prec1->program);
          glUniform1iARB (opengl_motion->shader_ref_prec1->textures[0], 0);
          glUniform2fARB (opengl_motion->shader_ref_prec1->offset, (hx >> 1) - x, (hy >> 1) - y);
          break;
        case 2:
        case 8:


        /*
      __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy, &fd00);
      if (rx == 0) {
        __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx, hy + 1, &fd10);
      } else {
        __schro_upsampled_frame_get_subdata_prec1 (upframe, k, hx + 1, hy, &fd10);
      }

      switch (fd->width) {
        case 8:
          oil_avg2_8xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        case 12:
          oil_avg2_12xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        case 16:
          oil_avg2_16xn_u8 (fd->data, fd->stride,
              fd00.data, fd00.stride, fd10.data, fd10.stride, fd->height);
          break;
        default:
          for(j=0;j<fd->height;j++) {
            uint8_t *data = SCHRO_FRAME_DATA_GET_LINE (fd, j);
            uint8_t *d00 = SCHRO_FRAME_DATA_GET_LINE (&fd00, j);
            uint8_t *d10 = SCHRO_FRAME_DATA_GET_LINE (&fd10, j);

            for(i=0;i<fd->width;i++) {
              data[i] = (1 + d00[i] + d10[i]) >> 1;
            }
          }
          break;
      }
      break;

      */

          break;
      }

      //schro_upsampled_frame_get_block_fast_prec3 (motion->src1, i, px, py, fd);
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }





  /*int weight = motion->ref1_weight + motion->ref2_weight;
  int shift = motion->ref_weight_precision;

  if (weight == (1<<shift)) {
    for(jj=0;jj<motion->yblen;jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->tmp_block_ref[0], jj);
      memcpy(d,s,motion->xblen);
    }
  } else {
    for(jj=0;jj<motion->yblen;jj++) {
      uint8_t *d = SCHRO_FRAME_DATA_GET_LINE (&motion->block, jj);
      uint8_t *s = SCHRO_FRAME_DATA_GET_LINE (&motion->tmp_block_ref[0], jj);
      for(ii=0;ii<motion->xblen;ii++) {
        d[ii] = ROUND_SHIFT(s[ii] * weight, shift);
      }
    }
  }*/



}
/*
static*/ void
schro_opengl_motion_render_block (SchroOpenGLMotion *opengl_motion, int i,
    int x, int y, int u, int v)
{
  SchroMotion *motion;
  SchroMotionVector *motion_vector;

  motion = opengl_motion->motion;
  motion_vector = &motion->motion_vectors[v * motion->params->x_num_blocks + u];

  switch (motion_vector->pred_mode) {
    case 0:
      schro_opengl_motion_render_dc_block (opengl_motion, i, x, y, u, v);
      break;
    case 1:
      schro_opengl_motion_render_ref_block (opengl_motion, i, x, y, u, v, 0);
      break;
    case 2:
      schro_opengl_motion_render_ref_block (opengl_motion, i, x, y, u, v, 1);
      break;
    case 3:
      //schro_opengl_motion_render_biref_block (opengl_motion, i, x, y, u, v);
      break;
    default:
      SCHRO_ASSERT (0);
      break;
  }
}

void
schro_opengl_motion_render (SchroMotion *motion, SchroFrame *dest)
{
  int i, k, u, v;
  int x, y;
  SchroParams *params = motion->params;
  SchroOpenGLFrameData *dest_opengl_data;
  SchroOpenGL *opengl;
  SchroChromaFormat chroma_format;
  SchroOpenGLShader *shader_clear;
  SchroOpenGLShader *shader_shift;
  SchroOpenGLMotion opengl_motion;

  SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (dest));
  SCHRO_ASSERT (SCHRO_FRAME_FORMAT_DEPTH (dest->format)
      == SCHRO_FRAME_FORMAT_DEPTH_S16);

  if (params->num_refs == 1) {
    SCHRO_ASSERT(params->picture_weight_2 == 1);
  }

  dest_opengl_data = (SchroOpenGLFrameData *) dest->components[0].data;

  SCHRO_ASSERT (dest_opengl_data != NULL);

  opengl = dest_opengl_data->opengl;

  schro_opengl_lock (opengl);

  shader_clear = schro_opengl_shader_get (opengl, SCHRO_OPENGL_SHADER_MC_CLEAR);
  shader_shift = schro_opengl_shader_get (opengl, SCHRO_OPENGL_SHADER_MC_SHIFT);

  SCHRO_ASSERT (shader_clear != NULL);
  SCHRO_ASSERT (shader_shift != NULL);

  opengl_motion.motion = motion;
  opengl_motion.shader_dc = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_MC_RENDER_DC);
  opengl_motion.shader_ref_prec0 = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_0);
  opengl_motion.shader_ref_prec0_weight = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_0_WEIGHT);
  opengl_motion.shader_ref_prec1 = schro_opengl_shader_get (opengl,
      SCHRO_OPENGL_SHADER_MC_RENDER_REF_PREC_1);

  SCHRO_ASSERT (opengl_motion.shader_dc != NULL);
  SCHRO_ASSERT (opengl_motion.shader_ref_prec0 != NULL);
  SCHRO_ASSERT (opengl_motion.shader_ref_prec0_weight != NULL);
  SCHRO_ASSERT (opengl_motion.shader_ref_prec1 != NULL);

  motion->ref_weight_precision = params->picture_weight_bits;
  motion->ref1_weight = params->picture_weight_1;
  motion->ref2_weight = params->picture_weight_2;
  motion->mv_precision = params->mv_precision;

  chroma_format = params->video_format->chroma_format;

  for (i = 0; i < 3; ++i) {
    dest_opengl_data = (SchroOpenGLFrameData *) dest->components[i].data;

    SCHRO_ASSERT (dest_opengl_data != NULL);
    SCHRO_ASSERT (dest_opengl_data->opengl == opengl);

    for (k = 0; k < 4; ++k) {
      if (motion->src1->frames[k]) {
        SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (motion->src1->frames[k]));

        opengl_motion.src_opengl_data[0][k]
            = (SchroOpenGLFrameData *) motion->src1->frames[k]->components[i].data;

        SCHRO_ASSERT (opengl_motion.src_opengl_data[0][k] != NULL);
        SCHRO_ASSERT (opengl_motion.src_opengl_data[0][k]->opengl == opengl);
      } else {
        opengl_motion.src_opengl_data[0][k] = NULL;
      }
    }

    if (params->num_refs > 1) {
      for (k = 0; k < 4; ++k) {
        if (motion->src2->frames[k]) {
          SCHRO_ASSERT (SCHRO_FRAME_IS_OPENGL (motion->src2->frames[k]));

          opengl_motion.src_opengl_data[1][k]
              = (SchroOpenGLFrameData *) motion->src2->frames[k]->components[i].data;

          SCHRO_ASSERT (opengl_motion.src_opengl_data[1][k] != NULL);
          SCHRO_ASSERT (opengl_motion.src_opengl_data[1][k]->opengl == opengl);
        } else {
          opengl_motion.src_opengl_data[1][k] = NULL;
        }
      }
    }

    if (i == 0) {
      motion->xbsep = params->xbsep_luma;
      motion->ybsep = params->ybsep_luma;
      motion->xblen = params->xblen_luma;
      motion->yblen = params->yblen_luma;
    } else {
      motion->xbsep = params->xbsep_luma
          >> SCHRO_CHROMA_FORMAT_H_SHIFT (chroma_format);
      motion->ybsep = params->ybsep_luma
          >> SCHRO_CHROMA_FORMAT_V_SHIFT (chroma_format);
      motion->xblen = params->xblen_luma
          >> SCHRO_CHROMA_FORMAT_H_SHIFT (chroma_format);
      motion->yblen = params->yblen_luma
          >> SCHRO_CHROMA_FORMAT_V_SHIFT (chroma_format);
    }

    motion->width = dest->components[i].width;
    motion->height = dest->components[i].height;
    motion->xoffset = (motion->xblen - motion->xbsep) / 2;
    motion->yoffset = (motion->yblen - motion->ybsep) / 2;
    motion->max_fast_x = (motion->width - motion->xblen) << motion->mv_precision;
    motion->max_fast_y = (motion->height - motion->yblen) << motion->mv_precision;
    motion->obmc_weight.data = schro_malloc (motion->xblen * motion->yblen * sizeof (int16_t));
    motion->obmc_weight.stride = motion->xblen * sizeof (int16_t);
    motion->obmc_weight.width = motion->xblen;
    motion->obmc_weight.height = motion->yblen;

    schro_motion_init_obmc_weight (motion);

    /* push obmc weight to texture */
    opengl_motion.obmc_weight_texture
         = schro_opengl_get_obmc_weight_texture (opengl,
         motion->obmc_weight.width, motion->obmc_weight.height);

    uint16_t *obmc_weight_u16 = schro_malloc (motion->obmc_weight.width
        * motion->obmc_weight.height * sizeof (uint16_t));
    uint16_t *obmc_weight_line_u16 = obmc_weight_u16;

    glBindTexture (GL_TEXTURE_RECTANGLE_ARB, opengl_motion.obmc_weight_texture);

    SCHRO_OPENGL_CHECK_ERROR

    int16_t *obmc_weight_line_s16 = (int16_t *) motion->obmc_weight.data;

    for (y = 0; y < motion->obmc_weight.height; ++y) {
      for (x = 0; x < motion->obmc_weight.width; ++x) {
        obmc_weight_line_u16[x] = (uint16_t) ((int32_t) obmc_weight_line_s16[x] + 32768);
      }

      obmc_weight_line_u16 = OFFSET (obmc_weight_line_u16, motion->obmc_weight.stride);
      obmc_weight_line_s16 = OFFSET (obmc_weight_line_s16, motion->obmc_weight.stride);
    }

    glTexSubImage2D (GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0,
        motion->obmc_weight.width, motion->obmc_weight.height, GL_RED,
        GL_UNSIGNED_SHORT, obmc_weight_u16);

    SCHRO_OPENGL_CHECK_ERROR

    schro_free (obmc_weight_u16);

    /* clear */
    schro_opengl_setup_viewport (motion->width, motion->height);

    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
        dest_opengl_data->framebuffers[1]);

    glUseProgramObjectARB (shader_clear->program);

    schro_opengl_render_quad (0, 0, motion->width, motion->height);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush();

    /* render blocks */
    int passes[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };

    for (k = 0; k < 4; ++k) {
      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
          dest_opengl_data->framebuffers[0]);
      glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
          dest_opengl_data->texture.handles[1]);

      glUseProgramObjectARB (0);

      schro_opengl_render_quad (0, 0, motion->width, motion->height);

      SCHRO_OPENGL_CHECK_ERROR

      glFlush();

      glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
          dest_opengl_data->framebuffers[1]);

      opengl_motion.previous_texture = dest_opengl_data->texture.handles[0];

      for (v = passes[k][0]; v < params->y_num_blocks; v += 2) {
        y = motion->ybsep * v - motion->yoffset;

        for (u = passes[k][1]; u < params->x_num_blocks; u += 2) {
          x = motion->xbsep * u - motion->xoffset;

          schro_opengl_motion_render_block (&opengl_motion, i, x, y, u, v);
        }
      }

      SCHRO_OPENGL_CHECK_ERROR

      glFlush();
    }

    /* shift */
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT,
        dest_opengl_data->framebuffers[0]);
    glBindTexture (GL_TEXTURE_RECTANGLE_ARB,
        dest_opengl_data->texture.handles[1]);

    glUseProgramObjectARB (shader_shift->program);
    glUniform1iARB (shader_shift->textures[0], 0);

    schro_opengl_render_quad (0, 0, motion->width, motion->height);

    SCHRO_OPENGL_CHECK_ERROR

    glFlush();

    schro_free (motion->obmc_weight.data);
  }

  glUseProgramObjectARB (0);
  glActiveTextureARB (GL_TEXTURE0_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
  glActiveTextureARB (GL_TEXTURE1_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
  glActiveTextureARB (GL_TEXTURE2_ARB);
  glBindTexture (GL_TEXTURE_RECTANGLE_ARB, 0);
  glActiveTextureARB (GL_TEXTURE0_ARB);
  glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, 0);

  schro_opengl_unlock (opengl);
}

