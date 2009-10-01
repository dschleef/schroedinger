
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <string.h>
#include <stdlib.h>
#include "../common.h"
#include "opengl_util.h"

int _benchmark = FALSE;
int _failed = FALSE;
int _generators = 0;
int _abort_on_failure = FALSE;
SchroMemoryDomain *_cpu_domain = NULL;
SchroMemoryDomain *_opengl_domain = NULL;
SchroOpenGL *_opengl = NULL;


void
opengl_test_failed (void)
{
  _failed = TRUE;

  if (_abort_on_failure) {
    abort ();
  }
}

int
opengl_format_name (SchroFrameFormat format, char *format_name, int size)
{
  switch (format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      strncpy (format_name, "U8 444", size);
      break;
    case SCHRO_FRAME_FORMAT_U8_422:
      strncpy (format_name, "U8 422", size);
      break;
    case SCHRO_FRAME_FORMAT_U8_420:
      strncpy (format_name, "U8 420", size);
      break;
    case SCHRO_FRAME_FORMAT_S16_444:
      strncpy (format_name, "S16 444", size);
      break;
    case SCHRO_FRAME_FORMAT_S16_422:
      strncpy (format_name, "S16 422", size);
      break;
    case SCHRO_FRAME_FORMAT_S16_420:
      strncpy (format_name, "S16 420", size);
      break;
    case SCHRO_FRAME_FORMAT_S32_444:
      strncpy (format_name, "S32 444", size);
      break;
    case SCHRO_FRAME_FORMAT_S32_422:
      strncpy (format_name, "S32 422", size);
      break;
    case SCHRO_FRAME_FORMAT_S32_420:
      strncpy (format_name, "S32 420", size);
      break;
    case SCHRO_FRAME_FORMAT_YUYV:
      strncpy (format_name, "YUYV", size);
      break;
    case SCHRO_FRAME_FORMAT_UYVY:
      strncpy (format_name, "UYVY", size);
      break;
    case SCHRO_FRAME_FORMAT_AYUV:
      strncpy (format_name, "AYUV", size);
      break;
    case SCHRO_FRAME_FORMAT_ARGB:
      strncpy (format_name, "ARGB", size);
      break;
    default:
      strncpy (format_name, "unknown", size);
      return FALSE;
  }

  return TRUE;
}

int
opengl_filter_name (int filter, char *filter_name, int size)
{
  switch (filter) {
    case SCHRO_WAVELET_DESLAURIES_DUBUC_9_7:
      strncpy (filter_name, "Deslauriers-Debuc (9,7)", size);
      break;
    case SCHRO_WAVELET_LE_GALL_5_3:
      strncpy (filter_name, "LeGall (5,3)", size);
      break;
    case SCHRO_WAVELET_DESLAURIES_DUBUC_13_7:
      strncpy (filter_name, "Deslauriers-Debuc (13,7)", size);
      break;
    case SCHRO_WAVELET_HAAR_0:
      strncpy (filter_name, "Haar 0", size);
      break;
    case SCHRO_WAVELET_HAAR_1:
      strncpy (filter_name, "Haar 1", size);
      break;
    case SCHRO_WAVELET_FIDELITY:
      strncpy (filter_name, "Fidelity", size);
      break;
    case SCHRO_WAVELET_DAUBECHIES_9_7:
      strncpy (filter_name, "Daubechies (9,7)", size);
      break;
    default:
      strncpy (filter_name, "unknown", size);
      return FALSE;
  }

  return TRUE;
}

static SchroFrameFormat
opengl_format_as_u8 (SchroFrameFormat format)
{
  switch (format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      return SCHRO_FRAME_FORMAT_U8_444;
    case SCHRO_FRAME_FORMAT_U8_422:
      return SCHRO_FRAME_FORMAT_U8_422;
    case SCHRO_FRAME_FORMAT_U8_420:
      return SCHRO_FRAME_FORMAT_U8_420;
    case SCHRO_FRAME_FORMAT_S16_444:
      return SCHRO_FRAME_FORMAT_U8_444;
    case SCHRO_FRAME_FORMAT_S16_422:
      return SCHRO_FRAME_FORMAT_U8_422;
    case SCHRO_FRAME_FORMAT_S16_420:
      return SCHRO_FRAME_FORMAT_U8_420;
    case SCHRO_FRAME_FORMAT_S32_444:
      return SCHRO_FRAME_FORMAT_U8_444;
    case SCHRO_FRAME_FORMAT_S32_422:
      return SCHRO_FRAME_FORMAT_U8_422;
    case SCHRO_FRAME_FORMAT_S32_420:
      return SCHRO_FRAME_FORMAT_U8_420;
    case SCHRO_FRAME_FORMAT_YUYV:
      return SCHRO_FRAME_FORMAT_U8_422;
    case SCHRO_FRAME_FORMAT_UYVY:
      return SCHRO_FRAME_FORMAT_U8_422;
    case SCHRO_FRAME_FORMAT_AYUV:
      return SCHRO_FRAME_FORMAT_U8_444;
    case SCHRO_FRAME_FORMAT_ARGB:
      return SCHRO_FRAME_FORMAT_U8_444;
  }

  return SCHRO_FRAME_FORMAT_U8_444;
}

#define CUSTOM_PATTERN_CONST(_block_u8, _block_s16) { \
    int i, j; \
    uint8_t *data_u8; \
    int16_t *data_s16; \
    if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format) \
        == SCHRO_FRAME_FORMAT_DEPTH_U8) { \
      for (j = 0; j < frame_data->height; ++j) { \
        data_u8 = SCHRO_FRAME_DATA_GET_LINE(frame_data, j); \
        for (i = 0; i < frame_data->width; ++i) { \
          _block_u8 \
        } \
     } \
    } else { \
      for (j = 0; j < frame_data->height; ++j) { \
        data_s16 = SCHRO_FRAME_DATA_GET_LINE(frame_data, j); \
        for (i = 0; i < frame_data->width; ++i) { \
          _block_s16 \
        } \
      } \
    } \
  }

static void
opengl_custom_pattern_random (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = orc_rand_u8();
  },{
    data_s16[i] = orc_rand_s16();

    /* FIXME: can't use full S16 range here, schro_frame_convert_u8_s16 doesn't
              support it, but I need to use schro_frame_convert_u8_s16 to get a
              ref frame to test the opengl convert against */
    if (data_s16[i] > 32767 - 128)
      data_s16[i] = 32767 - 128;
  })
}

static void
opengl_custom_pattern_random_u8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = orc_rand_u8();
  },{
    data_s16[i] = orc_rand_u8();
  })
}

static void
opengl_custom_pattern_random_s8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = orc_rand_u8();
  },{
    data_s16[i] = orc_rand_s8();
  })
}

static void
opengl_custom_pattern_const_1 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 1;
  },{
    data_s16[i] = 1;
  })
}

static void
opengl_custom_pattern_const_16 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 16;
  },{
    data_s16[i] = 16;
  })
}

static void
opengl_custom_pattern_const_min (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 0;
  },{
    data_s16[i] = -32768;
  })
}

static void
opengl_custom_pattern_const_min_u8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 0;
  },{
    data_s16[i] = 0;
  })
}

static void
opengl_custom_pattern_const_min_s8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = -128;
  },{
    data_s16[i] = -128;
  })
}

static void
opengl_custom_pattern_const_middle (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 127;
  },{
    data_s16[i] = 0;
  })
}

static void
opengl_custom_pattern_const_middle_u8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 127;
  },{
    data_s16[i] = 127;
  })
}

static void
opengl_custom_pattern_const_max (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 255;
  },{
    /* FIXME: can't use full S16 range here, schro_frame_convert_u8_s16 doesn't
              support it, but I need to use schro_frame_convert_u8_s16 to get a
              ref frame to test the opengl convert against */
    data_s16[i] = 32767 - 128;
  })
}

static void
opengl_custom_pattern_const_max_u8 (SchroFrameData *frame_data)
{
  CUSTOM_PATTERN_CONST({
    data_u8[i] = 255;
  },{
    data_s16[i] = 255;
  })
}

void
opengl_custom_pattern_generate (SchroFrame *cpu_frame,
    int pattern_type, int pattern_index, char* pattern_name)
{
  SchroFrameFormat format_u8;
  SchroFrame* cpu_frame_u8 = NULL;

  SCHRO_ASSERT(cpu_frame != NULL);

  switch (pattern_type) {
    default:
    case OPENGL_CUSTOM_PATTERN_NONE:
      format_u8 = opengl_format_as_u8 (cpu_frame->format);
      cpu_frame_u8 = schro_frame_new_and_alloc (_cpu_domain, format_u8,
          cpu_frame->width, cpu_frame->height);

      test_pattern_generate (cpu_frame_u8->components + 0, pattern_name,
          pattern_index % _generators);
      test_pattern_generate (cpu_frame_u8->components + 1, pattern_name,
          pattern_index % _generators);
      test_pattern_generate (cpu_frame_u8->components + 2, pattern_name,
          pattern_index % _generators);

      schro_frame_convert (cpu_frame, cpu_frame_u8);
  
      schro_frame_unref (cpu_frame_u8);
      break;
    case OPENGL_CUSTOM_PATTERN_RANDOM:
      strcpy (pattern_name, "custom random");

      opengl_custom_pattern_random (cpu_frame->components + 0);
      opengl_custom_pattern_random (cpu_frame->components + 1);
      opengl_custom_pattern_random (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_RANDOM_U8:
      strcpy (pattern_name, "custom random U8");

      opengl_custom_pattern_random_u8 (cpu_frame->components + 0);
      opengl_custom_pattern_random_u8 (cpu_frame->components + 1);
      opengl_custom_pattern_random_u8 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_RANDOM_S8:
      strcpy (pattern_name, "custom random S8");

      opengl_custom_pattern_random_s8 (cpu_frame->components + 0);
      opengl_custom_pattern_random_s8 (cpu_frame->components + 1);
      opengl_custom_pattern_random_s8 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_1:
      strcpy (pattern_name, "custom const 1");

      opengl_custom_pattern_const_1 (cpu_frame->components + 0);
      opengl_custom_pattern_const_1 (cpu_frame->components + 1);
      opengl_custom_pattern_const_1 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_16:
      strcpy (pattern_name, "custom const 16");

      opengl_custom_pattern_const_16 (cpu_frame->components + 0);
      opengl_custom_pattern_const_16 (cpu_frame->components + 1);
      opengl_custom_pattern_const_16 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIN:
      strcpy (pattern_name, "custom const min");

      opengl_custom_pattern_const_min (cpu_frame->components + 0);
      opengl_custom_pattern_const_min (cpu_frame->components + 1);
      opengl_custom_pattern_const_min (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIN_U8:
      strcpy (pattern_name, "custom const min U8");

      opengl_custom_pattern_const_min_u8 (cpu_frame->components + 0);
      opengl_custom_pattern_const_min_u8 (cpu_frame->components + 1);
      opengl_custom_pattern_const_min_u8 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIN_S8:
      strcpy (pattern_name, "custom const min S8");

      opengl_custom_pattern_const_min_s8 (cpu_frame->components + 0);
      opengl_custom_pattern_const_min_s8 (cpu_frame->components + 1);
      opengl_custom_pattern_const_min_s8 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIDDLE:
      strcpy (pattern_name, "custom const middle");

      opengl_custom_pattern_const_middle (cpu_frame->components + 0);
      opengl_custom_pattern_const_middle (cpu_frame->components + 1);
      opengl_custom_pattern_const_middle (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIDDLE_U8:
      strcpy (pattern_name, "custom const middle U8");

      opengl_custom_pattern_const_middle_u8 (cpu_frame->components + 0);
      opengl_custom_pattern_const_middle_u8 (cpu_frame->components + 1);
      opengl_custom_pattern_const_middle_u8 (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MAX:
      strcpy (pattern_name, "custom const max");

      opengl_custom_pattern_const_max (cpu_frame->components + 0);
      opengl_custom_pattern_const_max (cpu_frame->components + 1);
      opengl_custom_pattern_const_max (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MAX_U8:
      strcpy (pattern_name, "custom const max U8");

      opengl_custom_pattern_const_max_u8 (cpu_frame->components + 0);
      opengl_custom_pattern_const_max_u8 (cpu_frame->components + 1);
      opengl_custom_pattern_const_max_u8 (cpu_frame->components + 2);
      break;
  }
}

