
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboil.h>
#include <liboil/liboilrandom.h>
#include <string.h>
#include "../common.h"

int _benchmark = FALSE;
int _failed = FALSE;

#define OPENGL_CUSTOM_TEST_PATTERN_NONE         1
#define OPENGL_CUSTOM_TEST_PATTERN_RANDOM       2
#define OPENGL_CUSTOM_TEST_PATTERN_CONST_MIN    3
#define OPENGL_CUSTOM_TEST_PATTERN_CONST_MIDDLE 4
#define OPENGL_CUSTOM_TEST_PATTERN_CONST_MAX    5

void
opengl_test_push_pull (SchroFrameFormat format, int width, int height,
    int todo)
{
  SchroFrameFormat format_u8;
  char format_name[64];
  SchroMemoryDomain *cpu_domain;
  SchroMemoryDomain *opengl_domain;
  SchroFrame *cpu_ref_frame_u8;
  SchroFrame *cpu_ref_frame;
  SchroFrame *cpu_test_frame;
  SchroFrame *opengl_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i;
  int ok;
  int frames = 0;
  int generators = test_pattern_get_n_generators();
  int total_length;
  double start_push;
  double start_pull;
  double elapsed_push = 0;
  double elapsed_pull = 0;

  printf ("==========================================================\n");

  switch (format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      format_u8 = format;
      strcpy (format_name, "U8 444");
      break;
    case SCHRO_FRAME_FORMAT_U8_422:
      format_u8 = format;
      strcpy (format_name, "U8 422");
      break;
    case SCHRO_FRAME_FORMAT_U8_420:
      format_u8 = format;
      strcpy (format_name, "U8 420");
      break;
    case SCHRO_FRAME_FORMAT_S16_444:
      format_u8 = SCHRO_FRAME_FORMAT_U8_444;
      strcpy (format_name, "S16 444");
      break;
    case SCHRO_FRAME_FORMAT_S16_422:
      format_u8 = SCHRO_FRAME_FORMAT_U8_422;
      strcpy (format_name, "S16 422");
      break;
    case SCHRO_FRAME_FORMAT_S16_420:
      format_u8 = SCHRO_FRAME_FORMAT_U8_420;
      strcpy (format_name, "S16 420");
      break;
    default:
      printf ("opengl_test_push_pull: %ix%i\n", width, height);
      printf ("  unhandled format 0x%x", format);
      printf ("==========================================================\n");
      return;
  }

  printf ("opengl_test_push_pull: %ix%i %s\n", width, height, format_name);
  schro_opengl_frame_print_flags ("  ");

  cpu_domain = schro_memory_domain_new_local ();
  opengl_domain = schro_memory_domain_new_opengl ();
  cpu_ref_frame_u8 = schro_frame_new_and_alloc (cpu_domain, format_u8,
      width, height);
  cpu_ref_frame = schro_frame_new_and_alloc (cpu_domain, format, width,
      height);
  cpu_test_frame = schro_frame_new_and_alloc (cpu_domain, format, width,
      height);
  opengl_frame = schro_frame_new_and_alloc (opengl_domain, format, width,
      height);

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    test_pattern_generate (cpu_ref_frame_u8->components + 0, pattern_name,
        i % generators);
    test_pattern_generate (cpu_ref_frame_u8->components + 1, pattern_name,
        i % generators);
    test_pattern_generate (cpu_ref_frame_u8->components + 2, pattern_name,
        i % generators);

    schro_frame_convert (cpu_ref_frame, cpu_ref_frame_u8);

    /*test_pattern_generate (cpu_ref_frame->components + 0, pattern_name,
        i % generators);
    test_pattern_generate (cpu_ref_frame->components + 1, pattern_name,
        i % generators);
    test_pattern_generate (cpu_ref_frame->components + 2, pattern_name,
        i % generators);*/

    schro_opengl_lock ();

    schro_opengl_frame_setup (opengl_frame);

    start_push = schro_utils_get_time ();

    schro_opengl_frame_push (opengl_frame, cpu_ref_frame);

    start_pull = schro_utils_get_time ();
    elapsed_push += start_pull - start_push;

    schro_opengl_frame_pull (cpu_test_frame, opengl_frame);

    elapsed_pull += schro_utils_get_time () - start_pull;

    schro_opengl_frame_cleanup (opengl_frame);

    schro_opengl_unlock ();

    ++frames;

    ok = frame_compare (cpu_ref_frame, cpu_test_frame);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      _failed = TRUE;

      if (width <= 24 && height <= 24) {
        frame_dump (cpu_ref_frame, cpu_ref_frame);
        frame_dump (cpu_test_frame, cpu_ref_frame);
      }
    }
  }

  if (_benchmark) {
    total_length = (cpu_ref_frame->components[0].length
        + cpu_ref_frame->components[1].length
        + cpu_ref_frame->components[2].length) * frames;

    printf ("  results\n");
    printf ("    %i frames pushed/pulled: %.2f mbyte each\n", frames,
        (float)total_length / (1024 * 1024));
    printf ("    total %f/%f sec, %.2f/%.2f mbyte/sec\n", elapsed_push,
        elapsed_pull, total_length / elapsed_push / (1024 * 1024),
        total_length / elapsed_pull / (1024 * 1024));
    printf ("    avg   %f/%f sec, %f sec\n", elapsed_push / frames,
        elapsed_pull / frames, elapsed_push / frames + elapsed_pull / frames);
  }

  schro_frame_unref (cpu_ref_frame_u8);
  schro_frame_unref (cpu_ref_frame);
  schro_frame_unref (cpu_test_frame);
  schro_frame_unref (opengl_frame);
  schro_memory_domain_free (cpu_domain);
  schro_memory_domain_free (opengl_domain);

  printf ("==========================================================\n");
}

static void
opengl_custom_random (SchroFrameData *frame_data)
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
opengl_custom_const_min (SchroFrameData *frame_data)
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
opengl_custom_const_middle (SchroFrameData *frame_data)
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
opengl_custom_const_max (SchroFrameData *frame_data)
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

void
opengl_test_convert (SchroFrameFormat dest_format, SchroFrameFormat src_format,
    int dest_width, int dest_height, int src_width, int src_height, int todo,
    int custom_pattern)
{
  SchroFrameFormat src_format_u8;
  char dest_format_name[64];
  char src_format_name[64];
  SchroMemoryDomain *cpu_domain;
  SchroMemoryDomain *opengl_domain;
  SchroFrame *cpu_src_ref_frame_u8;
  SchroFrame *cpu_src_ref_frame;
  SchroFrame *cpu_dest_ref_frame;
  SchroFrame *cpu_dest_test_frame;
  SchroFrame *opengl_dest_frame;
  SchroFrame *opengl_src_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i, r;
  int ok;
  int frames = 0;
  int repeats = _benchmark ? 64 : 1;
  int generators = test_pattern_get_n_generators();
  int total_length;
  double start_cpu_convert, start_opengl_convert;
  double elapsed_cpu_convert = 0, elapsed_opengl_convert = 0;

  printf ("==========================================================\n");

  switch (dest_format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      strcpy (dest_format_name, "U8 444");
      break;
    case SCHRO_FRAME_FORMAT_U8_422:
      strcpy (dest_format_name, "U8 422");
      break;
    case SCHRO_FRAME_FORMAT_U8_420:
      strcpy (dest_format_name, "U8 420");
      break;
    case SCHRO_FRAME_FORMAT_S16_444:
      strcpy (dest_format_name, "S16 444");
      break;
    case SCHRO_FRAME_FORMAT_S16_422:
      strcpy (dest_format_name, "S16 422");
      break;
    case SCHRO_FRAME_FORMAT_S16_420:
      strcpy (dest_format_name, "S16 420");
      break;
    default:
      printf ("opengl_test_convert: %ix%i -> %ix%i\n", src_width, src_height,
          dest_width, dest_height);
      printf ("  unhandled dest_format 0x%x", dest_format);
      printf ("==========================================================\n");
      return;
  }

  switch (src_format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      src_format_u8 = src_format;
      strcpy (src_format_name, "U8 444");
      break;
    case SCHRO_FRAME_FORMAT_U8_422:
      src_format_u8 = src_format;
      strcpy (src_format_name, "U8 422");
      break;
    case SCHRO_FRAME_FORMAT_U8_420:
      src_format_u8 = src_format;
      strcpy (src_format_name, "U8 420");
      break;
    case SCHRO_FRAME_FORMAT_S16_444:
      src_format_u8 = SCHRO_FRAME_FORMAT_U8_444;
      strcpy (src_format_name, "S16 444");
      break;
    case SCHRO_FRAME_FORMAT_S16_422:
      src_format_u8 = SCHRO_FRAME_FORMAT_U8_422;
      strcpy (src_format_name, "S16 422");
      break;
    case SCHRO_FRAME_FORMAT_S16_420:
      src_format_u8 = SCHRO_FRAME_FORMAT_U8_420;
      strcpy (src_format_name, "S16 420");
      break;
    default:
      printf ("opengl_test_convert: %ix%i -> %ix%i\n", src_width, src_height,
          dest_width, dest_height);
      printf ("  unhandled src_format 0x%x", src_format);
      printf ("==========================================================\n");
      return;
  }

  printf ("opengl_test_convert: %ix%i -> %ix%i (%s -> %s)\n", src_width,
      src_height, dest_width, dest_height, src_format_name, dest_format_name);
  schro_opengl_frame_print_flags ("  ");

  cpu_domain = schro_memory_domain_new_local ();
  opengl_domain = schro_memory_domain_new_opengl ();
  cpu_src_ref_frame_u8 = schro_frame_new_and_alloc (cpu_domain, src_format_u8,
      src_width, src_height);
  cpu_src_ref_frame = schro_frame_new_and_alloc (cpu_domain, src_format,
      src_width, src_height);
  cpu_dest_ref_frame = schro_frame_new_and_alloc (cpu_domain, dest_format,
      dest_width, dest_height);
  cpu_dest_test_frame = schro_frame_new_and_alloc (cpu_domain, dest_format,
      dest_width, dest_height);
  opengl_dest_frame = schro_frame_new_and_alloc (opengl_domain, dest_format,
      dest_width, dest_height);
  opengl_src_frame = schro_frame_new_and_alloc (opengl_domain, src_format,
      src_width, src_height);

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    switch (custom_pattern) {
      default:
      case OPENGL_CUSTOM_TEST_PATTERN_NONE:
        test_pattern_generate (cpu_src_ref_frame_u8->components + 0,
            pattern_name, i % generators);
        test_pattern_generate (cpu_src_ref_frame_u8->components + 1,
            pattern_name, i % generators);
        test_pattern_generate (cpu_src_ref_frame_u8->components + 2,
            pattern_name, i % generators);

        schro_frame_convert (cpu_src_ref_frame, cpu_src_ref_frame_u8);
        break;
      case OPENGL_CUSTOM_TEST_PATTERN_RANDOM:
        strcpy (pattern_name, "custom random");

        opengl_custom_random (cpu_src_ref_frame->components + 0);
        opengl_custom_random (cpu_src_ref_frame->components + 1);
        opengl_custom_random (cpu_src_ref_frame->components + 2);
        break;
      case OPENGL_CUSTOM_TEST_PATTERN_CONST_MIN:
        strcpy (pattern_name, "custom const min");

        opengl_custom_const_min (cpu_src_ref_frame->components + 0);
        opengl_custom_const_min (cpu_src_ref_frame->components + 1);
        opengl_custom_const_min (cpu_src_ref_frame->components + 2);
        break;
      case OPENGL_CUSTOM_TEST_PATTERN_CONST_MIDDLE:
        strcpy (pattern_name, "custom const middle");

        opengl_custom_const_middle (cpu_src_ref_frame->components + 0);
        opengl_custom_const_middle (cpu_src_ref_frame->components + 1);
        opengl_custom_const_middle (cpu_src_ref_frame->components + 2);
        break;
      case OPENGL_CUSTOM_TEST_PATTERN_CONST_MAX:
        strcpy (pattern_name, "custom const max");

        opengl_custom_const_max (cpu_src_ref_frame->components + 0);
        opengl_custom_const_max (cpu_src_ref_frame->components + 1);
        opengl_custom_const_max (cpu_src_ref_frame->components + 2);
        break;
    }

    start_cpu_convert = schro_utils_get_time ();

    for (r = 0; r < repeats; ++r)
      schro_frame_convert (cpu_dest_ref_frame, cpu_src_ref_frame);

    elapsed_cpu_convert += schro_utils_get_time () - start_cpu_convert;

    schro_opengl_lock ();

    schro_opengl_frame_setup (opengl_dest_frame);
    schro_opengl_frame_setup (opengl_src_frame);

    schro_opengl_frame_push (opengl_src_frame, cpu_src_ref_frame);

    start_opengl_convert = schro_utils_get_time ();

    for (r = 0; r < repeats; ++r)
      schro_opengl_frame_convert (opengl_dest_frame, opengl_src_frame);

    elapsed_opengl_convert += schro_utils_get_time () - start_opengl_convert;

    schro_opengl_frame_pull (cpu_dest_test_frame, opengl_dest_frame);

    schro_opengl_frame_cleanup (opengl_dest_frame);
    schro_opengl_frame_cleanup (opengl_src_frame);

    schro_opengl_unlock ();

    ++frames;

    ok = frame_compare (cpu_dest_ref_frame, cpu_dest_test_frame);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      _failed = TRUE;

      if (dest_width <= 24 && dest_height <= 24 && src_width <= 24
          && src_height <= 24) {
        printf ("    src ref frame\n");
        frame_dump (cpu_src_ref_frame, cpu_src_ref_frame);

        printf ("    dest ref frame\n");
        frame_dump (cpu_dest_ref_frame, cpu_dest_ref_frame);

        printf ("    dest test frame <-> dest ref frame\n");
        frame_dump (cpu_dest_test_frame, cpu_dest_ref_frame);
      }
    }
  }

  if (_benchmark) {
    total_length = (cpu_src_ref_frame->components[0].length
        + cpu_src_ref_frame->components[1].length
        + cpu_src_ref_frame->components[2].length) * frames;

    printf ("  results\n");
    printf ("    %i frames converted via cpu/opengl with %i repeats: %.2f "
        "mbyte each\n", frames, repeats,
        ((double)total_length * repeats) / (1024 * 1024));
    printf ("    total %f/%f sec, %.2f/%.2f mbyte/sec\n", elapsed_cpu_convert,
        elapsed_opengl_convert,
        ((double)total_length * repeats) / elapsed_cpu_convert / (1024 * 1024),
        ((double)total_length * repeats) / elapsed_opengl_convert
        / (1024 * 1024));
    printf ("    avg   %.8f/%.8f sec\n",
        elapsed_cpu_convert / repeats / frames,
        elapsed_opengl_convert / repeats / frames);
  }

  schro_frame_unref (cpu_src_ref_frame_u8);
  schro_frame_unref (cpu_src_ref_frame);
  schro_frame_unref (cpu_dest_ref_frame);
  schro_frame_unref (cpu_dest_test_frame);
  schro_frame_unref (opengl_dest_frame);
  schro_frame_unref (opengl_src_frame);
  schro_memory_domain_free (cpu_domain);
  schro_memory_domain_free (opengl_domain);

  printf ("==========================================================\n");
}

struct ConvertTest {
  SchroFrameFormat dest_format;
  SchroFrameFormat src_format;
  int dest_width, dest_height;
  int src_width, src_height;
  int todo;
  int custom_pattern;
};

struct ConvertTest opengl_test_convert_list[] = {
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1280,  720, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  //{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 1280,  720, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 19, 19, 21, 21, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 21, 21, 19, 19, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 19, 21, 21, 19, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 1280,  720, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 1920, 1080, 1280,  720, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  //{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 1920, 1080, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 16, 16, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 19, 19, 21, 21, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 21, 21, 19, 19, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 19, 21, 21, 19, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },
  //{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 32, 32, 16, 16, -1, OPENGL_CUSTOM_TEST_PATTERN_NONE },

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_U8_444, 1280,  720, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1280,  720, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  //{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_U8_420, 1920, 1080, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 1280,  720, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, 1920, 1080, 1280,  720, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
  //{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, 1920, 1080, 1920, 1080, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM },
};

int
main (int argc, char *argv[])
{
  int i;
  int generators;

  schro_init ();

  if (argc >= 2 && (!strcmp(argv[1], "-b") || !strcmp(argv[1], "--benchmark")))
    _benchmark = TRUE;

  generators = test_pattern_get_n_generators();

  if (_benchmark) {
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 100);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 1920, 1080, 100);
    opengl_test_convert (SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_TEST_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_TEST_PATTERN_RANDOM);
    //opengl_test_convert (SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
    //opengl_test_convert (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
    //    16, 16, 16, 16, 1, OPENGL_CUSTOM_TEST_PATTERN_RANDOM);
  } else {
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 16, 16, generators);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 16, 16, generators);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 23, 16, generators);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 16, 23, generators);

    for (i = 0; i < ARRAY_SIZE (opengl_test_convert_list); ++i) {
      opengl_test_convert (opengl_test_convert_list[i].dest_format,
          opengl_test_convert_list[i].src_format,
          opengl_test_convert_list[i].dest_width,
          opengl_test_convert_list[i].dest_height,
          opengl_test_convert_list[i].src_width,
          opengl_test_convert_list[i].src_height,
          opengl_test_convert_list[i].todo < 1 ? generators
          : opengl_test_convert_list[i].todo,
          opengl_test_convert_list[i].custom_pattern);
    }
  }

  if (_failed) {
    printf ("FAILED\n");
    return 1;
  }

  printf ("SUCCESS\n");
  return 0;
}

