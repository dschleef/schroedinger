
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>
#include <schroedinger/opengl/schroopenglcanvas.h>
#include <schroedinger/opengl/schroopenglframe.h>
#include <schroedinger/opengl/schroopenglmotion.h>
#include <schroedinger/opengl/schroopenglwavelet.h>
#include <string.h>
#include "../common.h"
#include "opengl_util.h"

void
opengl_test_convert (SchroFrameFormat dest_format, SchroFrameFormat src_format,
    int dest_width, int dest_height, int src_width, int src_height, int todo,
    int custom_pattern)
{
  char dest_format_name[64];
  char src_format_name[64];
  SchroFrame *cpu_dest_ref_frame;
  SchroFrame *cpu_src_ref_frame;
  SchroFrame *cpu_dest_test_frame;
  SchroFrame *opengl_dest_frame;
  SchroFrame *opengl_src_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i, r;
  int ok;
  int frames = 0;
  int repeats = _benchmark ? 64 : 1;
  int total_length;
  double start_cpu_convert, start_opengl_convert;
  double elapsed_cpu_convert = 0, elapsed_opengl_convert = 0;

  printf ("==========================================================\n");

  if (!opengl_format_name (dest_format, dest_format_name, 64)) {
    printf ("opengl_test_convert: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled dest_format 0x%x\n", dest_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  if (!opengl_format_name (src_format, src_format_name, 64)) {
    printf ("opengl_test_convert: %ix%i -> %ix%i\n", src_width, src_height,
        dest_width, dest_height);
    printf ("  unhandled src_format 0x%x\n", src_format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  printf ("opengl_test_convert: %ix%i -> %ix%i (%s -> %s)\n", src_width,
      src_height, dest_width, dest_height, src_format_name, dest_format_name);

  cpu_dest_ref_frame = schro_frame_new_and_alloc (_cpu_domain, dest_format,
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
    opengl_custom_pattern_generate (cpu_src_ref_frame, custom_pattern, i,
        pattern_name);

    start_cpu_convert = schro_utils_get_time ();

    for (r = 0; r < repeats; ++r) {
      schro_frame_convert (cpu_dest_ref_frame, cpu_src_ref_frame);
    }

    elapsed_cpu_convert += schro_utils_get_time () - start_cpu_convert;

    schro_opengl_lock (_opengl);

    schro_opengl_frame_push (opengl_src_frame, cpu_src_ref_frame);

    start_opengl_convert = schro_utils_get_time ();

    for (r = 0; r < repeats; ++r) {
      schro_opengl_frame_convert (opengl_dest_frame, opengl_src_frame);
    }

    elapsed_opengl_convert += schro_utils_get_time () - start_opengl_convert;

    schro_opengl_frame_pull (cpu_dest_test_frame, opengl_dest_frame);

    schro_opengl_unlock (_opengl);

    ++frames;

    ok = frame_compare (cpu_dest_ref_frame, cpu_dest_test_frame);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      if (dest_width <= 32 && dest_height <= 32 && src_width <= 32
          && src_height <= 32) {
        printf ("dest ref frame\n");
        frame_dump (cpu_dest_ref_frame, cpu_dest_ref_frame);

        printf ("src ref frame\n");
        frame_dump (cpu_src_ref_frame, cpu_src_ref_frame);

        printf ("dest test frame <-> dest ref frame\n");
        frame_dump (cpu_dest_test_frame, cpu_dest_ref_frame);
      }

      opengl_test_failed ();
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

  schro_frame_unref (cpu_dest_ref_frame);
  schro_frame_unref (cpu_src_ref_frame);
  schro_frame_unref (cpu_dest_test_frame);
  schro_frame_unref (opengl_dest_frame);
  schro_frame_unref (opengl_src_frame);

  printf ("==========================================================\n");
}

void
opengl_test_wavelet_inverse (SchroFrameFormat format, int width, int height,
    int todo, int custom_pattern, int filter, int transform_depth)
{
  char format_name[64];
  char filter_name[64];
  SchroFrame *cpu_preref_frame;
  SchroFrame *cpu_postref_frame;
  SchroFrame *cpu_test_frame;
  SchroFrame *opengl_frame;
  SchroFrameData *cpu_preref_frame_data;
  SchroFrameData *cpu_postref_frame_data;
  SchroFrameData *cpu_test_frame_data;
  SchroFrameData *opengl_frame_data;
  int16_t *tmp;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i/*, r*/;
  int ok;
  int frames = 0;
  int repeats = 1;//_benchmark ? 64 : 1;
  int level;
  int total_length;
  double start_cpu;
  double start_opengl;
  double elapsed_cpu = 0;
  double elapsed_opengl = 0;

  printf ("==========================================================\n");

  if (!opengl_format_name(format, format_name, 64)) {
    printf ("opengl_test_wavelet_inverse: %ix%i\n", width, height);
    printf ("  unhandled format 0x%x", format);
    printf ("==========================================================\n");
    return;
  }

  if (!opengl_filter_name(filter, filter_name, 64)) {
    printf ("opengl_test_wavelet_inverse: %ix%i %s\n", width, height,
        format_name);
    printf ("  unhandled filter 0x%x", format);
    printf ("==========================================================\n");
    return;
  }

  printf ("opengl_test_wavelet_inverse: %ix%i %s, %s, %i level%c\n", width,
      height, format_name, filter_name, transform_depth,
      transform_depth > 1 ? 's' : ' ');

  if (_benchmark) {
    schro_opengl_canvas_print_flags ("  ");
  }

  cpu_preref_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  cpu_postref_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  cpu_test_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  opengl_frame = schro_opengl_frame_new (_opengl, _opengl_domain, format, width,
      height);
  tmp = schro_malloc (2 * width * sizeof(int16_t));

  cpu_preref_frame_data = cpu_preref_frame->components + 0;
  cpu_postref_frame_data = cpu_postref_frame->components + 0;
  cpu_test_frame_data = cpu_test_frame->components + 0;
  opengl_frame_data = opengl_frame->components + 0;

  printf ("  patterns\n");

  for (i = 0; i < todo; ++i) {
    opengl_custom_pattern_generate (cpu_preref_frame, custom_pattern, i,
        pattern_name);

    /* FIXME: use schro_frame_iwt_transform */
    /* cpu forward transform */
    for (level = 0; level < transform_depth; ++level) {
      SchroFrameData frame_data;

      frame_data.format = cpu_preref_frame->format;
      frame_data.data = cpu_preref_frame->components[0].data;
      frame_data.width = cpu_preref_frame->components[0].width >> level;
      frame_data.height = cpu_preref_frame->components[0].height >> level;
      frame_data.stride = cpu_preref_frame->components[0].stride << level;

      schro_wavelet_transform_2d (&frame_data, filter, tmp);
    }

    schro_frame_convert (cpu_postref_frame, cpu_preref_frame);

    schro_opengl_lock (_opengl);

    schro_opengl_frame_push (opengl_frame, cpu_postref_frame);

    start_cpu = schro_utils_get_time ();

    /* FIXME: use schro_frame_inverse_iwt_transform */
    /* cpu inverse transform */
    for (level = transform_depth - 1; level >= 0; --level) {
      SchroFrameData frame_data;

      frame_data.format = cpu_postref_frame->format;
      frame_data.data = cpu_postref_frame->components[0].data;
      frame_data.width = cpu_postref_frame->components[0].width >> level;
      frame_data.height = cpu_postref_frame->components[0].height >> level;
      frame_data.stride = cpu_postref_frame->components[0].stride << level;

      schro_wavelet_inverse_transform_2d (&frame_data, filter, tmp);
    }

    start_opengl = schro_utils_get_time ();
    elapsed_cpu += start_opengl - start_cpu;

    /* FIXME: use schro_opengl_frame_inverse_iwt_transform */
    /* opengl vertical deinterleave */
    for (level = 0; level < transform_depth; ++level) {
      SchroFrameData frame_data;

      frame_data.format = opengl_frame->format;
      frame_data.data = opengl_frame->components[0].data;
      frame_data.width = opengl_frame->components[0].width >> level;
      frame_data.height = opengl_frame->components[0].height >> level;
      frame_data.stride = opengl_frame->components[0].stride << level;

      schro_opengl_wavelet_vertical_deinterleave (&frame_data);
    }

    /* opengl inverse transform */
    for (level = transform_depth - 1; level >= 0; --level) {
      SchroFrameData frame_data;

      frame_data.format = opengl_frame->format;
      frame_data.data = opengl_frame->components[0].data;
      frame_data.width = opengl_frame->components[0].width >> level;
      frame_data.height = opengl_frame->components[0].height >> level;
      frame_data.stride = opengl_frame->components[0].stride << level;

      schro_opengl_wavelet_inverse_transform (&frame_data, filter);
    }

    elapsed_opengl += schro_utils_get_time () - start_opengl;

    schro_opengl_frame_pull (cpu_test_frame, opengl_frame);

    schro_opengl_unlock (_opengl);

    ++frames;

    ok = frame_data_compare (cpu_postref_frame_data, cpu_test_frame_data);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      if (width <= 32 && height <= 32) {
        printf ("preref frame\n");
        frame_data_dump (cpu_preref_frame_data, cpu_preref_frame_data);

        printf ("postref frame\n");
        frame_data_dump (cpu_postref_frame_data, cpu_postref_frame_data);

        printf ("test frame <-> postref frame\n");
        frame_data_dump (cpu_test_frame_data, cpu_postref_frame_data);
      }

      opengl_test_failed ();
    }
  }

  if (_benchmark) {
    total_length = cpu_preref_frame_data->length * frames;

    printf ("  results\n");
    printf ("    %i frames iiwt'ed via cpu/opengl with %i repeats: %.2f "
        "mbyte each\n", frames, repeats,
        ((double)total_length * repeats) / (1024 * 1024));
    printf ("    total %f/%f sec, %.2f/%.2f mbyte/sec\n", elapsed_cpu,
        elapsed_opengl,
        ((double)total_length * repeats) / elapsed_cpu / (1024 * 1024),
        ((double)total_length * repeats) / elapsed_opengl / (1024 * 1024));
    printf ("    avg   %.8f/%.8f sec\n", elapsed_cpu / repeats / frames,
        elapsed_opengl / repeats / frames);
  }

  schro_frame_unref (cpu_preref_frame);
  schro_frame_unref (cpu_postref_frame);
  schro_frame_unref (cpu_test_frame);
  schro_frame_unref (opengl_frame);
  schro_free (tmp);

  printf ("==========================================================\n");
}

void
opengl_test_upsample (SchroFrameFormat format, int width, int height,
    int custom_pattern)
{
  char format_name[64];
  SchroFrame *cpu_ref_frame;
  SchroFrame *cpu_test_frame;
  SchroFrame *opengl_frame;
  SchroUpsampledFrame *upsampled_cpu_ref_frame;
  SchroUpsampledFrame *upsampled_opengl_frame;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i, k;
  int ok;

  printf ("==========================================================\n");

  if (!opengl_format_name (format, format_name, 64)) {
    printf ("opengl_test_upsample: %ix%i\n", width, height);
    printf ("  unhandled format 0x%x\n", format);
    printf ("==========================================================\n");

    opengl_test_failed ();
    return;
  }

  printf ("opengl_test_upsample: %ix%i (%s)\n", width, height, format_name);

  cpu_ref_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  cpu_test_frame = schro_frame_new_and_alloc (_cpu_domain, format, width,
      height);
  opengl_frame = schro_opengl_frame_new (_opengl, _opengl_domain, format, width,
      height);

  printf ("  patterns\n");

  for (i = 0; i < 1; ++i) {
    opengl_custom_pattern_generate (cpu_ref_frame, custom_pattern, i,
        pattern_name);

    schro_opengl_frame_push (opengl_frame, cpu_ref_frame);

    upsampled_cpu_ref_frame = schro_upsampled_frame_new (schro_frame_ref (cpu_ref_frame));
    upsampled_opengl_frame = schro_upsampled_frame_new (schro_frame_ref (opengl_frame));

    schro_upsampled_frame_upsample (upsampled_cpu_ref_frame);
    schro_opengl_upsampled_frame_upsample (upsampled_opengl_frame);

    for (k = 1; k < 4; ++k) {
      schro_opengl_frame_pull (cpu_test_frame, upsampled_opengl_frame->frames[k]);

      ok = frame_compare (upsampled_cpu_ref_frame->frames[k], cpu_test_frame);

      printf ("    %s: %s (%i)\n", pattern_name, ok ? "OK" : "broken", k);

      if (!ok) {
        if (width <= 32 && height <= 32) {
          printf ("test frame %i <-> ref frame %i\n", k ,k);
          frame_dump (cpu_test_frame, upsampled_cpu_ref_frame->frames[1]);
        }

        opengl_test_failed ();
      }
    }

    schro_upsampled_frame_free (upsampled_cpu_ref_frame);
    schro_upsampled_frame_free (upsampled_opengl_frame);
  }

  schro_frame_unref (cpu_ref_frame);
  schro_frame_unref (cpu_test_frame);
  schro_frame_unref (opengl_frame);

  printf ("==========================================================\n");
}

void
opengl_test_motion (int width, int height,
    int custom_pattern)
{
  SchroVideoFormat video_format;
  SchroParams params;
  SchroFrame *cpu_src_frame;
  SchroFrame *opengl_src_frame;
  SchroFrame *cpu_dest_ref_frame;
  SchroFrame *cpu_dest_test_frame;
  SchroFrame *opengl_dest_frame;
  SchroUpsampledFrame *upsampled_cpu_frame;
  SchroUpsampledFrame *upsampled_opengl_frame;
  SchroMotionVector *motion_vectors_dc;
  SchroMotion motion_cpu;
  SchroMotion motion_opengl;
  char pattern_name[TEST_PATTERN_NAME_SIZE];
  int i, u, v;
  int ok;

 /* video_format.width = width;
  video_format.height = height;
  video_format.chroma_format = SCHRO_CHROMA_444;*/

  schro_video_format_set_std_video_format (&video_format, SCHRO_VIDEO_FORMAT_QSIF);

  schro_video_format_validate (&video_format);

  video_format.width = 32;
  video_format.height = 32;

  printf ("==========================================================\n");
  printf ("opengl_test_motion: %ix%i\n", video_format.width, video_format.height);

  params.video_format = &video_format;
  params.xbsep_luma = 4;
  params.ybsep_luma = 4;
  params.xblen_luma = 6;
  params.yblen_luma = 6;
  params.num_refs = 1;
  params.picture_weight_2 = 1;

  schro_params_calculate_mc_sizes (&params);

  cpu_src_frame = schro_frame_new_and_alloc (_cpu_domain,
      SCHRO_FRAME_FORMAT_U8_420, video_format.width, video_format.height);
  opengl_src_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      SCHRO_FRAME_FORMAT_U8_420, video_format.width, video_format.height);
  cpu_dest_ref_frame = schro_frame_new_and_alloc (_cpu_domain,
      SCHRO_FRAME_FORMAT_S16_420, video_format.width, video_format.height);
  cpu_dest_test_frame = schro_frame_new_and_alloc (_cpu_domain,
      SCHRO_FRAME_FORMAT_S16_420,video_format.width, video_format.height);
  opengl_dest_frame = schro_opengl_frame_new (_opengl, _opengl_domain,
      SCHRO_FRAME_FORMAT_S16_420, video_format.width, video_format.height);

  printf ("  patterns\n");

  for (i = 0; i < 1; ++i) {
    opengl_custom_pattern_generate (cpu_src_frame, custom_pattern, i,
        pattern_name);

    schro_opengl_frame_push (opengl_src_frame, cpu_src_frame);

    upsampled_cpu_frame = schro_upsampled_frame_new (schro_frame_ref (cpu_src_frame));
    upsampled_opengl_frame = schro_upsampled_frame_new (schro_frame_ref (opengl_src_frame));

    schro_upsampled_frame_upsample (upsampled_cpu_frame);
    schro_opengl_upsampled_frame_upsample (upsampled_opengl_frame);


    motion_vectors_dc = schro_malloc0 (sizeof (SchroMotionVector) *
        params.x_num_blocks * params.y_num_blocks);

    for (v = 0; v < params.y_num_blocks; ++v) {
      for (u = 0; u < params.x_num_blocks; ++u) {
        if (u == 1 && (v == 1/* || v == 2*/)) {
          motion_vectors_dc[v * params.x_num_blocks + u].pred_mode = 0;
          motion_vectors_dc[v * params.x_num_blocks + u].using_global = FALSE;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[0] = 64;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[1] = 0;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[2] = 0;
        } else {
          motion_vectors_dc[v * params.x_num_blocks + u].pred_mode = 0;
          motion_vectors_dc[v * params.x_num_blocks + u].using_global = FALSE;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[0] = 0;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[1] = 0;
          motion_vectors_dc[v * params.x_num_blocks + u].u.dc.dc[2] = 0;
        }
      }
    }

    memset (&motion_cpu, 0, sizeof (SchroMotion));
    memset (&motion_opengl, 0, sizeof (SchroMotion));

    motion_cpu.src1 = upsampled_cpu_frame;
    motion_cpu.src2 = NULL;
    motion_cpu.motion_vectors = (SchroMotionVector *) motion_vectors_dc;
    motion_cpu.params = &params;
    motion_cpu.mv_precision = 0;

    motion_opengl.src1 = upsampled_opengl_frame;
    motion_opengl.src2 = NULL;
    motion_opengl.motion_vectors = (SchroMotionVector *) motion_vectors_dc;
    motion_opengl.params = &params;
    motion_opengl.mv_precision = 0;


    schro_motion_render (&motion_cpu, cpu_dest_ref_frame);
    schro_opengl_motion_render (&motion_opengl, opengl_dest_frame);

    schro_free (motion_vectors_dc);

    schro_opengl_frame_pull (cpu_dest_test_frame, opengl_dest_frame);

    ok = frame_compare (cpu_dest_ref_frame, cpu_dest_test_frame);

    printf ("    %s: %s\n", pattern_name, ok ? "OK" : "broken");

    if (!ok) {
      /*if (width <= 32 && height <= 32)*/ {
        printf ("ref frame\n");
        frame_dump (cpu_dest_ref_frame, cpu_dest_ref_frame);
        printf ("test frame <-> ref frame\n");
        frame_dump (cpu_dest_test_frame, cpu_dest_ref_frame);
      }

      opengl_test_failed ();
    }

    schro_upsampled_frame_free (upsampled_cpu_frame);
    schro_upsampled_frame_free (upsampled_opengl_frame);
  }

  schro_frame_unref (cpu_src_frame);
  schro_frame_unref (opengl_src_frame);
  schro_frame_unref (cpu_dest_ref_frame);
  schro_frame_unref (cpu_dest_test_frame);
  schro_frame_unref (opengl_dest_frame);

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
  /* S16 -> U8 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1280,
      720, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  /*{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 1280, 720, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },*/

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 16, 16, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 16, 16, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 16, 16, 16, 16,
      -1, OPENGL_CUSTOM_ATTERN_NONE },*/

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 19, 19, 21, 21, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 21, 21, 19, 19, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 19, 21, 21, 19,
     -1, OPENGL_CUSTOM_PATTERN_NONE },*/

  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444, 32, 32, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_U8_422, 32, 32, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_U8_420, 32, 32, 16, 16,
      -1, OPENGL_CUSTOM_PATTERN_NONE },*/

  /* U8 -> S16 */
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 1280, 720,
      1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 1920, 1080,
      1280, 720, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  /*{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 1920, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },*/

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 16, 16, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 16, 16, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 16, 16, 16, 16,
      -1, OPENGL_CUSTOM_PATTERN_NONE },*/

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 19, 19, 21, 21, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 21, 21, 19, 19, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 19, 21, 21, 19,
      -1, OPENGL_CUSTOM_PATTERN_NONE },*/

  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444, 32, 32, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_S16_422, 32, 32, 16, 16, -1,
      OPENGL_CUSTOM_PATTERN_NONE },
  /*{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_S16_420, 32, 32, 16, 16,
      -1, OPENGL_CUSTOM_PATTERN_NONE },*/

  /* U8 -> U8 */
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_U8_444, 1280,  720, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1280,
      720, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  /*{ SCHRO_FRAME_FORMAT_U8_420, SCHRO_FRAME_FORMAT_U8_420, 1920, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },*/

  /* S16 -> S16 */
  { SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444, 1280,  720, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  { SCHRO_FRAME_FORMAT_S16_422, SCHRO_FRAME_FORMAT_S16_422, 1920, 1080, 1280,
      720, 1, OPENGL_CUSTOM_PATTERN_RANDOM },
  /*{ SCHRO_FRAME_FORMAT_S16_420, SCHRO_FRAME_FORMAT_S16_420, 1920, 1080,
      1920, 1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },*/

  /* YUYV -> U8 422 */
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV, 1920 / 2, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },

  /* UYVY -> U8 422 */
  { SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_UYVY, 1920 / 2, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },

  /* AYUV -> U8 444 */
  { SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_AYUV, 1920 / 4, 1080, 1920,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },

  /* U8 422 -> YUYV */
  { SCHRO_FRAME_FORMAT_YUYV, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1920 / 2,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },

  /* U8 422 -> UYVY */
  { SCHRO_FRAME_FORMAT_UYVY, SCHRO_FRAME_FORMAT_U8_422, 1920, 1080, 1920 / 2,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM },

  /* U8 444 -> AYUV */
  { SCHRO_FRAME_FORMAT_AYUV, SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 1920 / 4,
      1080, 1, OPENGL_CUSTOM_PATTERN_RANDOM }
};

void opengl_test_push_pull (SchroFrameFormat format, int width, int height,
    int todo, int custom_pattern);
void opengl_test_push_pull_run (void);

void opengl_test_add (SchroFrameFormat dest_format, SchroFrameFormat src_format,
    int dest_width, int dest_height, int src_width, int src_height, int todo,
    int custom_dest_pattern, int custom_src_pattern, int dest_pattern_drift);
void opengl_test_subtract (SchroFrameFormat dest_format,
    SchroFrameFormat src_format, int dest_width, int dest_height,
    int src_width, int src_height, int todo, int custom_dest_pattern,
    int custom_src_pattern, int dest_pattern_drift);
void opengl_test_combine_run (void);

int
main (int argc, char *argv[])
{
  int i;
  int special = FALSE;
  SchroOpenGL *local_opengl;

  for (i = 1; i < argc; ++i) {
    if (!strcmp (argv[i], "-b") || !strcmp (argv[i], "--benchmark")) {
      _benchmark = TRUE;
    }

    if (!strcmp (argv[i], "-s") || !strcmp (argv[i], "--special")) {
      special = TRUE;
    }
  }

  _abort_on_failure = TRUE;

  schro_init ();

  _generators = test_pattern_get_n_generators ();
  _cpu_domain = schro_memory_domain_new_local ();
  _opengl_domain = schro_memory_domain_new_opengl ();

  local_opengl = schro_opengl_new ();
  _opengl = local_opengl;

  //opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 4, 1,
  //      OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_LE_GALL_5_3, 1);

  _opengl = schro_opengl_new ();

  if (_benchmark && !special) {
    /* push/pull */
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_U8_444, 1920, 1080, 100,
        OPENGL_CUSTOM_PATTERN_NONE);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 1920, 1080, 100,
        OPENGL_CUSTOM_PATTERN_NONE);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_YUYV, 1920, 1080, 100,
        OPENGL_CUSTOM_PATTERN_NONE);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_UYVY, 1920, 1080, 100,
        OPENGL_CUSTOM_PATTERN_NONE);
    opengl_test_push_pull (SCHRO_FRAME_FORMAT_AYUV, 1920, 1080, 100,
        OPENGL_CUSTOM_PATTERN_NONE);

    /* convert */
    opengl_test_convert (SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);

    opengl_test_convert (SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV,
        1920 / 2, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_UYVY,
        1920 / 2, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_AYUV,
        1920 / 4, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);

    opengl_test_convert (SCHRO_FRAME_FORMAT_YUYV, SCHRO_FRAME_FORMAT_U8_422,
        1920, 1080, 1920 / 2, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_UYVY, SCHRO_FRAME_FORMAT_U8_422,
        1920, 1080, 1920 / 2, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_convert (SCHRO_FRAME_FORMAT_AYUV, SCHRO_FRAME_FORMAT_U8_444,
        1920, 1080, 1920 / 4, 1080, 50, OPENGL_CUSTOM_PATTERN_RANDOM);

    /* add */
    opengl_test_add (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_CONST_MIDDLE,
        OPENGL_CUSTOM_PATTERN_RANDOM_S8, 0);
    opengl_test_add (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_CONST_MIDDLE,
        OPENGL_CUSTOM_PATTERN_RANDOM, 0);

    /* subtract */
    opengl_test_subtract (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_CONST_MAX,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, 0);
    opengl_test_subtract (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
        1920, 1080, 1920, 1080, 50, OPENGL_CUSTOM_PATTERN_CONST_MIDDLE,
        OPENGL_CUSTOM_PATTERN_RANDOM, 0);
  } else if (special) {
    /*opengl_test_push_pull (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 1,
        OPENGL_CUSTOM_PATTERN_RANDOM);*/
    /*opengl_test_convert (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_U8_444,
        16, 16, 16, 16, 1, OPENGL_CUSTOM_PATTERN_RANDOM);*/
    /*opengl_test_convert (SCHRO_FRAME_FORMAT_U8_444, SCHRO_FRAME_FORMAT_S16_444,
        16, 16, 16, 16, 1, OPENGL_CUSTOM_PATTERN_RANDOM);*/
    /*opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 16, 16,
        OPENGL_CUSTOM_PATTERN_RANDOM);*/
    opengl_test_motion (720, 480, OPENGL_CUSTOM_PATTERN_RANDOM);

    /*opengl_test_convert (SCHRO_FRAME_FORMAT_S16_444, SCHRO_FRAME_FORMAT_S16_444,
        16, 16, 16, 16, 1, OPENGL_CUSTOM_PATTERN_RANDOM);*/

    /*opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 1,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_LE_GALL_5_3, 1);*/

    /*opengl_test_convert (SCHRO_FRAME_FORMAT_U8_422, SCHRO_FRAME_FORMAT_YUYV,
        32 / 2, 16, 32, 16, 1, OPENGL_CUSTOM_PATTERN_RANDOM);*/
  } else {
    opengl_test_push_pull_run ();

    /* convert */
    for (i = 0; i < ARRAY_SIZE (opengl_test_convert_list); ++i) {
      opengl_test_convert (opengl_test_convert_list[i].dest_format,
          opengl_test_convert_list[i].src_format,
          opengl_test_convert_list[i].dest_width,
          opengl_test_convert_list[i].dest_height,
          opengl_test_convert_list[i].src_width,
          opengl_test_convert_list[i].src_height,
          opengl_test_convert_list[i].todo < 1 ? _generators
          : opengl_test_convert_list[i].todo,
          opengl_test_convert_list[i].custom_pattern);
    }

    opengl_test_combine_run ();

    /* inverse wavelet */
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 64, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_DESLAURIES_DUBUC_9_7, 4);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 64, 64, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_LE_GALL_5_3, 6);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 1024, 1024, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_LE_GALL_5_3, 3);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_LE_GALL_5_3, 4);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_DESLAURIES_DUBUC_13_7, 1);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_HAAR_0, 2);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 1024, 1024, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_HAAR_0, 10);
    opengl_test_wavelet_inverse (SCHRO_FRAME_FORMAT_S16_444, 16, 16, 3,
        OPENGL_CUSTOM_PATTERN_RANDOM_U8, SCHRO_WAVELET_HAAR_1, 4);

    /* upsample */
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 320, 240,
        OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 1920, 1080,
        OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 16, 16,
        OPENGL_CUSTOM_PATTERN_RANDOM);
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 16, 16,
        OPENGL_CUSTOM_PATTERN_CONST_MIN);
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 16, 16,
        OPENGL_CUSTOM_PATTERN_CONST_MIDDLE);
    opengl_test_upsample (SCHRO_FRAME_FORMAT_U8_444, 16, 16,
        OPENGL_CUSTOM_PATTERN_CONST_MAX);
  }

  schro_opengl_free (local_opengl);

  schro_opengl_free (_opengl);
  schro_memory_domain_free (_cpu_domain);
  schro_memory_domain_free (_opengl_domain);

  if (_failed) {
    printf ("FAILED\n");
    return 1;
  }

  printf ("SUCCESS\n");
  return 0;
}

