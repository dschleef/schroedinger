
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <string.h>
#include "../common.h"

static void
opengl_test_push_pull (SchroFrameFormat format, int width, int height,
    int todo)
{
  SchroFrameFormat format_u8;
  char format_name[64];
  SchroMemoryDomain *cpu_domain;
  SchroMemoryDomain *opengl_domain;
  SchroFrame *cpu_frame_u8_ref;
  SchroFrame *cpu_frame_ref;
  SchroFrame *cpu_frame_test;
  SchroFrame *opengl_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i;
  int ok;
  int frames = 0;
  double start_push;
  double start_pull;
  double elapsed_push = 0;
  double elapsed_pull = 0;

  switch (format) {
    case SCHRO_FRAME_FORMAT_U8_444:
      format_u8 = format;
      strcpy(format_name, "U8 444");
      break;
    case SCHRO_FRAME_FORMAT_U8_422:
      format_u8 = format;
      strcpy(format_name, "U8 422");
      break;
    case SCHRO_FRAME_FORMAT_U8_420:
      format_u8 = format;
      strcpy(format_name, "U8 420");
      break;
    case SCHRO_FRAME_FORMAT_S16_444:
      format_u8 = SCHRO_FRAME_FORMAT_U8_444;
      strcpy(format_name, "S16 444");
      break;
    case SCHRO_FRAME_FORMAT_S16_422:
      format_u8 = SCHRO_FRAME_FORMAT_U8_422;
      strcpy(format_name, "S16 422");
      break;
    case SCHRO_FRAME_FORMAT_S16_420:
      format_u8 = SCHRO_FRAME_FORMAT_U8_420;
      strcpy(format_name, "S16 420");
      break;
    default:
      printf("opengl_test_push_pull: %ix%i\n", width, height);
      printf("  unhandled format 0x%x", format);
      return;
  }

  printf("opengl_test_push_pull: %ix%i %s\n", width, height, format_name);
  schro_opengl_frame_print_flags ("  ");

  cpu_domain = schro_memory_domain_new_local ();
  opengl_domain = schro_memory_domain_new_opengl ();
  cpu_frame_u8_ref = schro_frame_new_and_alloc (cpu_domain,
      format_u8, width, height);
  cpu_frame_ref = schro_frame_new_and_alloc (cpu_domain, format, width,
      height);
  cpu_frame_test = schro_frame_new_and_alloc (cpu_domain, format, width,
      height);
  opengl_frame = schro_frame_new_and_alloc (opengl_domain, format, width,
      height);

  printf("  patterns\n");

  for (i = 0; i < todo; ++i) {
    test_pattern_generate (cpu_frame_u8_ref->components + 0, pattern_name,
        i % test_pattern_get_n_generators());
    test_pattern_generate (cpu_frame_u8_ref->components + 1, pattern_name,
        i % test_pattern_get_n_generators());
    test_pattern_generate (cpu_frame_u8_ref->components + 2, pattern_name,
        i % test_pattern_get_n_generators());

    schro_frame_convert (cpu_frame_ref, cpu_frame_u8_ref);

    schro_opengl_lock ();

    schro_opengl_frame_setup (opengl_frame);

    start_push = schro_utils_get_time ();

    schro_opengl_frame_push (opengl_frame, cpu_frame_ref);

    start_pull = schro_utils_get_time ();
    elapsed_push += start_pull - start_push;

    schro_opengl_frame_pull (cpu_frame_test, opengl_frame);

    elapsed_pull += schro_utils_get_time () - start_pull;

    schro_opengl_frame_cleanup (opengl_frame);

    schro_opengl_unlock ();

    ++frames;

    ok = frame_compare (cpu_frame_ref, cpu_frame_test);

    printf("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok && width <= 24 && height <= 24) {
      frame_dump (cpu_frame_ref, cpu_frame_ref);
      frame_dump (cpu_frame_test, cpu_frame_ref);
    }
  }

  int total_length = (cpu_frame_ref->components[0].length
      + cpu_frame_ref->components[1].length
      + cpu_frame_ref->components[2].length) * frames;

  printf("  results\n");
  printf("    %i frames push/pull %f mbyte each\n", frames,
      (float)total_length / (1024 * 1024));
  printf("    total %f/%f sec, %f/%f mbyte/sec\n", elapsed_push,
      elapsed_pull, total_length / elapsed_push / (1024 * 1024),
      total_length / elapsed_pull / (1024 * 1024));
  printf("    avg   %f/%f sec, %f sec\n", elapsed_push / frames,
      elapsed_pull / frames, elapsed_push / frames + elapsed_pull / frames);

  schro_frame_unref (cpu_frame_u8_ref);
  schro_frame_unref (cpu_frame_ref);
  schro_frame_unref (cpu_frame_test);
  schro_frame_unref (opengl_frame);
  schro_memory_domain_free (cpu_domain);
  schro_memory_domain_free (opengl_domain);
}

int
main (int argc, char *argv[])
{
  schro_init ();

  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 16, 16, 1);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 1);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 99, 17, 1);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 99, 17, 1);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 32, 32, 1);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 640, 480, 1);
  opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 100);
  opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 1920, 1080, 100);
  //opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_422, 1917, 1080, 1);

  return 0;
}

