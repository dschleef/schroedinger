
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <string.h>
#include "../common.h"
#include "opengl_util.h"

void
opengl_test_push_pull (SchroFrameFormat format, int width, int height,
    int todo, int custom_pattern)
{
  char format_name[64];
  SchroFrame *cpu_ref_frame;
  SchroFrame *cpu_test_frame;
  SchroFrame *opengl_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i;
  int ok;
  int frames = 0;
  int total_length;
  double start_push;
  double start_pull;
  double elapsed_push = 0;
  double elapsed_pull = 0;

  printf ("==========================================================\n");

  if (!opengl_format_name(format, format_name, 64)) {
    printf ("opengl_test_push_pull: %ix%i\n", width, height);
    printf ("  unhandled format 0x%x", format);
    printf ("==========================================================\n");
    return;
  }

  printf ("opengl_test_push_pull: %ix%i %s\n", width, height, format_name);

  if (_benchmark) {
    schro_opengl_canvas_print_flags ("  ");
  }

  cpu_ref_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  cpu_test_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  opengl_frame = schro_opengl_frame_new (_opengl, _opengl_domain, format, width,
      height);

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    opengl_custom_pattern_generate (cpu_ref_frame, custom_pattern, i,
        pattern_name);

    schro_opengl_lock (_opengl);

    start_push = schro_utils_get_time ();

    schro_opengl_frame_push (opengl_frame, cpu_ref_frame);

    start_pull = schro_utils_get_time ();
    elapsed_push += start_pull - start_push;

    schro_opengl_frame_pull (cpu_test_frame, opengl_frame);

    elapsed_pull += schro_utils_get_time () - start_pull;

    schro_opengl_unlock (_opengl);

    ++frames;

    ok = frame_compare (cpu_ref_frame, cpu_test_frame);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      if (width <= 32 && height <= 32) {
        frame_dump (cpu_ref_frame, cpu_ref_frame);
        frame_dump (cpu_test_frame, cpu_ref_frame);
      }

      opengl_test_failed ();
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

  schro_frame_unref (cpu_ref_frame);
  schro_frame_unref (cpu_test_frame);
  schro_frame_unref (opengl_frame);

  printf ("==========================================================\n");
}

struct PushPullTest {
  SchroFrameFormat format;
  int width, height;
  int todo;
  int custom_pattern;
};

static struct PushPullTest opengl_test_push_pull_list[] = {
  { SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_444, 16, 16, -1, OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_444, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_444, 16, 16, -1, OPENGL_CUSTOM_PATTERN_NONE  },
  { SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_444, 23, 16, -1, OPENGL_CUSTOM_PATTERN_NONE  },
  { SCHRO_FRAME_FORMAT_S16_444, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_444, 16, 23, -1, OPENGL_CUSTOM_PATTERN_NONE  },
  { SCHRO_FRAME_FORMAT_YUYV, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_YUYV, 16, 16, -1, OPENGL_CUSTOM_PATTERN_NONE  },
  { SCHRO_FRAME_FORMAT_UYVY, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_UYVY, 16, 16, -1, OPENGL_CUSTOM_PATTERN_NONE  },
  { SCHRO_FRAME_FORMAT_AYUV, 1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_AYUV, 16, 16, -1, OPENGL_CUSTOM_PATTERN_NONE  }
};

void opengl_test_push_pull_run ()
{
  int i;

  for (i = 0; i < ARRAY_SIZE (opengl_test_push_pull_list); ++i) {
    opengl_test_push_pull (opengl_test_push_pull_list[i].format,
        opengl_test_push_pull_list[i].width,
        opengl_test_push_pull_list[i].height,
        opengl_test_push_pull_list[i].todo < 1 ? _generators
        : opengl_test_push_pull_list[i].todo,
        opengl_test_push_pull_list[i].custom_pattern);
  }
}
