
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>
#include <string.h>
#include "../common.h"
#include "opengl_util.h"

int _benchmark = FALSE;
int _failed = FALSE;
int _generators = 0;
SchroMemoryDomain *_cpu_domain = NULL;
SchroMemoryDomain *_opengl_domain = NULL;

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

static void
opengl_custom_pattern_random (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = oil_rand_u8();
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = oil_rand_s16();

        // FIXME: can't use full S16 range here, schro_frame_convert_u8_s16
        // doesn't support it, but I need to use schro_frame_convert_u8_s16
        // to get a ref frame to test the opengl convert against
        if (data[i] > 32767 - 128)
          data[i] = 32767 - 128;
      }
    }
  }
}
static void
opengl_custom_pattern_random_u8 (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = oil_rand_u8();
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = oil_rand_u8();
      }
    }
  }
}

static void
opengl_custom_pattern_random_s8 (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    opengl_custom_pattern_random_u8 (frame_data);
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = oil_rand_s8();
      }
    }
  }
}

static void
opengl_custom_pattern_const_min (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 0;
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = -32768;
      }
    }
  }
}

static void
opengl_custom_pattern_const_middle (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 127;
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 0;
      }
    }
  }
}

static void
opengl_custom_pattern_const_max (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 255;
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        // FIXME: can't use full S16 range here, schro_frame_convert_u8_s16
        // doesn't support it, but I need to use schro_frame_convert_u8_s16
        // to get a ref frame to test the opengl convert against
        data[i] = 32767 - 128;
      }
    }
  }
}

static void
opengl_custom_pattern_const_max_u8 (SchroFrameData *frame_data)
{
  int i, j;

  if (SCHRO_FRAME_FORMAT_DEPTH(frame_data->format)
      == SCHRO_FRAME_FORMAT_DEPTH_U8) {
    uint8_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 255;
      }
    }
  } else {
    int16_t *data;

    for (j = 0; j < frame_data->height; ++j) {
      data = SCHRO_FRAME_DATA_GET_LINE(frame_data, j);

      for (i = 0; i < frame_data->width; ++i) {
        data[i] = 255;
      }
    }
  }
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
    case OPENGL_CUSTOM_PATTERN_CONST_MIN:
      strcpy (pattern_name, "custom const min");

      opengl_custom_pattern_const_min (cpu_frame->components + 0);
      opengl_custom_pattern_const_min (cpu_frame->components + 1);
      opengl_custom_pattern_const_min (cpu_frame->components + 2);
      break;
    case OPENGL_CUSTOM_PATTERN_CONST_MIDDLE:
      strcpy (pattern_name, "custom const middle");

      opengl_custom_pattern_const_middle (cpu_frame->components + 0);
      opengl_custom_pattern_const_middle (cpu_frame->components + 1);
      opengl_custom_pattern_const_middle (cpu_frame->components + 2);
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

