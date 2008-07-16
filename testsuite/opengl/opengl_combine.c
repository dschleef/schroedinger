
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <string.h>
#include "../common.h"
#include "opengl_util.h"

void
opengl_test_add (SchroFrameFormat dest_format, SchroFrameFormat src_format,
    int dest_width, int dest_height, int src_width, int src_height, int todo,
    int custom_dest_pattern, int custom_src_pattern, int dest_pattern_drift)
{
  char dest_format_name[64];
  char src_format_name[64];
  SchroFrame *cpu_dest_preref_frame;
  SchroFrame *cpu_dest_postref_frame;
  SchroFrame *cpu_src_ref_frame;
  SchroFrame *cpu_dest_test_frame;
  SchroFrame *opengl_dest_frame;
  SchroFrame *opengl_src_frame;
  char dest_pattern_name[TEST_PATTERN_NAME_SIZE];
  char src_pattern_name[TEST_PATTERN_NAME_SIZE];
  int i/*, r*/;
  int ok;
  int frames = 0;
  int repeats = 1;//_benchmark ? 64 : 1;
  int total_length;
  double start_cpu, start_opengl;
  double elapsed_cpu = 0, elapsed_opengl = 0;

  printf ("==========================================================\n");

  if (!opengl_format_name (dest_format, dest_format_name, 64)) {
    printf ("opengl_test_add: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled dest_format 0x%x\n", dest_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  if (!opengl_format_name (src_format, src_format_name, 64)) {
    printf ("opengl_test_add: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled src_format 0x%x\n", src_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  printf ("opengl_test_add: %ix%i -> %ix%i (%s -> %s)\n", src_width,
      src_height, dest_width, dest_height, src_format_name, dest_format_name);

  cpu_dest_preref_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  cpu_dest_postref_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  cpu_src_ref_frame = schro_frame_new_and_alloc (_cpu_domain, src_format,
      src_width, src_height);
  cpu_dest_test_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  opengl_dest_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      dest_format, dest_width, dest_height);
  opengl_src_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      src_format, src_width, src_height);

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    opengl_custom_pattern_generate (cpu_dest_preref_frame, custom_dest_pattern,
        i + dest_pattern_drift, dest_pattern_name);
    opengl_custom_pattern_generate (cpu_src_ref_frame, custom_src_pattern, i,
        src_pattern_name);

    schro_frame_convert (cpu_dest_postref_frame, cpu_dest_preref_frame);

    schro_opengl_lock (_opengl);

    schro_opengl_frame_push (opengl_dest_frame, cpu_dest_preref_frame);
    schro_opengl_frame_push (opengl_src_frame, cpu_src_ref_frame);

    schro_opengl_unlock (_opengl);

    start_cpu = schro_utils_get_time ();

    //for (r = 0; r < repeats; ++r)
      schro_frame_add (cpu_dest_postref_frame, cpu_src_ref_frame);

    elapsed_cpu += schro_utils_get_time () - start_cpu;

    schro_opengl_lock (_opengl);

    start_opengl = schro_utils_get_time ();

    //for (r = 0; r < repeats; ++r)
      schro_opengl_frame_add (opengl_dest_frame, opengl_src_frame);

    elapsed_opengl += schro_utils_get_time () - start_opengl;

    schro_opengl_frame_pull (cpu_dest_test_frame, opengl_dest_frame);

    schro_opengl_unlock (_opengl);

    ++frames;

    ok = frame_compare (cpu_dest_postref_frame, cpu_dest_test_frame);

    printf ("    %s -> %s: %s\n", src_pattern_name, dest_pattern_name,
        ok ? "OK" : "broken");

    if (!ok) {
      if (dest_width <= 32 && dest_height <= 32 && src_width <= 32
          && src_height <= 32) {
        printf ("dest preref frame\n");
        frame_dump (cpu_dest_preref_frame, cpu_dest_preref_frame);

        printf ("src ref frame\n");
        frame_dump (cpu_src_ref_frame, cpu_src_ref_frame);

        printf ("dest postref frame = dest preref frame + src ref frame\n");
        frame_dump (cpu_dest_postref_frame, cpu_dest_postref_frame);

        printf ("dest test frame <-> dest postref frame\n");
        frame_dump (cpu_dest_test_frame, cpu_dest_postref_frame);
      }

      opengl_test_failed ();
    }
  }

  if (_benchmark) {
    total_length = (cpu_src_ref_frame->components[0].length
        + cpu_src_ref_frame->components[1].length
        + cpu_src_ref_frame->components[2].length) * frames;

    printf ("  results\n");
    printf ("    %i frames added via cpu/opengl with %i repeats: %.2f "
        "mbyte each\n", frames, repeats,
        ((double)total_length * repeats) / (1024 * 1024));
    printf ("    total %f/%f sec, %.2f/%.2f mbyte/sec\n", elapsed_cpu,
        elapsed_opengl,
        ((double)total_length * repeats) / elapsed_cpu / (1024 * 1024),
        ((double)total_length * repeats) / elapsed_opengl / (1024 * 1024));
    printf ("    avg   %.8f/%.8f sec\n", elapsed_cpu / repeats / frames,
        elapsed_opengl / repeats / frames);
  }

  schro_frame_unref (cpu_dest_preref_frame);
  schro_frame_unref (cpu_dest_postref_frame);
  schro_frame_unref (cpu_src_ref_frame);
  schro_frame_unref (cpu_dest_test_frame);
  schro_frame_unref (opengl_dest_frame);
  schro_frame_unref (opengl_src_frame);

  printf ("==========================================================\n");
}

void
opengl_test_subtract (SchroFrameFormat dest_format,
    SchroFrameFormat src_format, int dest_width, int dest_height,
    int src_width, int src_height, int todo, int custom_dest_pattern,
    int custom_src_pattern, int dest_pattern_drift)
{
  char dest_format_name[64];
  char src_format_name[64];
  SchroFrame *cpu_dest_preref_frame;
  SchroFrame *cpu_dest_postref_frame;
  SchroFrame *cpu_src_ref_frame;
  SchroFrame *cpu_dest_test_frame;
  SchroFrame *opengl_dest_frame;
  SchroFrame *opengl_src_frame;
  char dest_pattern_name[TEST_PATTERN_NAME_SIZE];
  char src_pattern_name[TEST_PATTERN_NAME_SIZE];
  int i/*, r*/;
  int ok;
  int frames = 0;
  int repeats = 1;//_benchmark ? 64 : 1;
  int total_length;
  double start_cpu, start_opengl;
  double elapsed_cpu = 0, elapsed_opengl = 0;

  printf ("==========================================================\n");

  if (!opengl_format_name (dest_format, dest_format_name, 64)) {
    printf ("opengl_test_add: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled dest_format 0x%x\n", dest_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  if (!opengl_format_name (src_format, src_format_name, 64)) {
    printf ("opengl_test_add: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled src_format 0x%x\n", src_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  printf ("opengl_test_subtract: %ix%i -> %ix%i (%s -> %s)\n", src_width,
      src_height, dest_width, dest_height, src_format_name, dest_format_name);

  cpu_dest_preref_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  cpu_dest_postref_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  cpu_src_ref_frame = schro_frame_new_and_alloc (_cpu_domain, src_format,
      src_width, src_height);
  cpu_dest_test_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
      dest_width, dest_height);
  opengl_dest_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      dest_format, dest_width, dest_height);
  opengl_src_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      src_format, src_width, src_height);

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    opengl_custom_pattern_generate (cpu_dest_preref_frame, custom_dest_pattern,
        i + dest_pattern_drift, dest_pattern_name);
    opengl_custom_pattern_generate (cpu_src_ref_frame, custom_src_pattern, i,
        src_pattern_name);

    schro_frame_convert (cpu_dest_postref_frame, cpu_dest_preref_frame);

    schro_opengl_lock (_opengl);

    schro_opengl_frame_push (opengl_dest_frame, cpu_dest_preref_frame);
    schro_opengl_frame_push (opengl_src_frame, cpu_src_ref_frame);

    schro_opengl_unlock (_opengl);

    start_cpu = schro_utils_get_time ();

    //for (r = 0; r < repeats; ++r)
      schro_frame_subtract (cpu_dest_postref_frame, cpu_src_ref_frame);

    elapsed_cpu += schro_utils_get_time () - start_cpu;

    schro_opengl_lock (_opengl);

    start_opengl = schro_utils_get_time ();

    //for (r = 0; r < repeats; ++r)
      schro_opengl_frame_subtract (opengl_dest_frame, opengl_src_frame);

    elapsed_opengl += schro_utils_get_time () - start_opengl;

    schro_opengl_frame_pull (cpu_dest_test_frame, opengl_dest_frame);

    schro_opengl_unlock (_opengl);

    ++frames;

    ok = frame_compare (cpu_dest_postref_frame, cpu_dest_test_frame);

    printf ("    %s -> %s: %s\n", src_pattern_name, dest_pattern_name,
        ok ? "OK" : "broken");

    if (!ok) {
      if (dest_width <= 32 && dest_height <= 32 && src_width <= 32
          && src_height <= 32) {
        printf ("dest preref frame\n");
        frame_dump (cpu_dest_preref_frame, cpu_dest_preref_frame);

        printf ("src ref frame\n");
        frame_dump (cpu_src_ref_frame, cpu_src_ref_frame);

        printf ("dest postref frame = dest preref frame + src ref frame\n");
        frame_dump (cpu_dest_postref_frame, cpu_dest_postref_frame);

        printf ("dest test frame <-> dest postref frame\n");
        frame_dump (cpu_dest_test_frame, cpu_dest_postref_frame);
      }

      opengl_test_failed ();
    }
  }

  if (_benchmark) {
    total_length = (cpu_src_ref_frame->components[0].length
        + cpu_src_ref_frame->components[1].length
        + cpu_src_ref_frame->components[2].length) * frames;

    printf ("  results\n");
    printf ("    %i frames subtracted via cpu/opengl with %i repeats: %.2f "
        "mbyte each\n", frames, repeats,
        ((double)total_length * repeats) / (1024 * 1024));
    printf ("    total %f/%f sec, %.2f/%.2f mbyte/sec\n", elapsed_cpu,
        elapsed_opengl,
        ((double)total_length * repeats) / elapsed_cpu / (1024 * 1024),
        ((double)total_length * repeats) / elapsed_opengl / (1024 * 1024));
    printf ("    avg   %.8f/%.8f sec\n", elapsed_cpu / repeats / frames,
        elapsed_opengl / repeats / frames);
  }

  schro_frame_unref (cpu_dest_preref_frame);
  schro_frame_unref (cpu_dest_postref_frame);
  schro_frame_unref (cpu_src_ref_frame);
  schro_frame_unref (cpu_dest_test_frame);
  schro_frame_unref (opengl_dest_frame);
  schro_frame_unref (opengl_src_frame);

  printf ("==========================================================\n");
}

struct CombineTest {
  SchroFrameFormat dest_format;
  SchroFrameFormat src_format;
  int dest_width, dest_height;
  int src_width, src_height;
  int todo;
  int custom_dest_pattern;
  int custom_src_pattern;
  int dest_pattern_drift;
};

struct CombineTest opengl_test_add_list[] = {
  /* U8 -> S16 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIDDLE, OPENGL_CUSTOM_PATTERN_RANDOM, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIN, OPENGL_CUSTOM_PATTERN_CONST_MAX, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, -1,
    OPENGL_CUSTOM_PATTERN_NONE, OPENGL_CUSTOM_PATTERN_NONE, 2 },

  /* S16 -> S16 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIN, OPENGL_CUSTOM_PATTERN_CONST_MAX, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIN, OPENGL_CUSTOM_PATTERN_RANDOM_U8, 0 }
};

struct CombineTest opengl_test_subtract_list[] = {
  /* U8 -> S16 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIDDLE, OPENGL_CUSTOM_PATTERN_RANDOM, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MIDDLE, OPENGL_CUSTOM_PATTERN_CONST_MAX, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, -1,
    OPENGL_CUSTOM_PATTERN_NONE, OPENGL_CUSTOM_PATTERN_NONE, 2 },

  /* S16 -> S16 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MAX, OPENGL_CUSTOM_PATTERN_CONST_MAX_U8, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MAX, OPENGL_CUSTOM_PATTERN_CONST_MAX, 0 },
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, 1,
    OPENGL_CUSTOM_PATTERN_CONST_MAX, OPENGL_CUSTOM_PATTERN_RANDOM_U8, 0 }
};

void
opengl_test_combine_run ()
{
  int i;

  /* add */
  for (i = 0; i < ARRAY_SIZE (opengl_test_add_list); ++i) {
    opengl_test_add (opengl_test_add_list[i].dest_format,
        opengl_test_add_list[i].src_format,
        opengl_test_add_list[i].dest_width,
        opengl_test_add_list[i].dest_height,
        opengl_test_add_list[i].src_width,
        opengl_test_add_list[i].src_height,
        opengl_test_add_list[i].todo < 1 ? _generators
        : opengl_test_add_list[i].todo,
        opengl_test_add_list[i].custom_dest_pattern,
        opengl_test_add_list[i].custom_src_pattern,
        opengl_test_add_list[i].dest_pattern_drift);
  }

  /* subtract */
  for (i = 0; i < ARRAY_SIZE (opengl_test_subtract_list); ++i) {
    opengl_test_subtract (opengl_test_subtract_list[i].dest_format,
        opengl_test_subtract_list[i].src_format,
        opengl_test_subtract_list[i].dest_width,
        opengl_test_subtract_list[i].dest_height,
        opengl_test_subtract_list[i].src_width,
        opengl_test_subtract_list[i].src_height,
        opengl_test_subtract_list[i].todo < 1 ? _generators
        : opengl_test_subtract_list[i].todo,
        opengl_test_subtract_list[i].custom_dest_pattern,
        opengl_test_subtract_list[i].custom_src_pattern,
        opengl_test_subtract_list[i].dest_pattern_drift);
  }
}

