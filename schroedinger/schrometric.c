
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include "schroorc.h"


int
schro_metric_absdiff_u8 (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  uint32_t metric = 0;

  if (height == 8 && width == 8) {
    orc_sad_8x8_u8 (&metric, a, a_stride, b, b_stride);
  } else if (height == 12 && width == 12) {
    orc_sad_12x12_u8 (&metric, a, a_stride, b, b_stride);
  } else if (width == 16) {
    orc_sad_16xn_u8 (&metric, a, a_stride, b, b_stride, height);
  } else if (width == 32) {
    orc_sad_32xn_u8 (&metric, a, a_stride, b, b_stride, height);
  } else {
    orc_sad_nxm_u8 (&metric, a, a_stride, b, b_stride, width, height);
  }

  return metric;
}

void
schro_metric_scan_do_scan (SchroMetricScan *scan, int use_chroma)
{
  SCHRO_ASSERT (scan->ref_x + scan->block_width + scan->scan_width - 1 <= scan->frame->width + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y + scan->block_height + scan->scan_height - 1 <= scan->frame->height + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_x >= -scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y >= -scan->frame->extension);
  SCHRO_ASSERT (scan->scan_width > 0);
  SCHRO_ASSERT (scan->scan_height > 0);

  int k;
  uint32_t metrics[SCHRO_LIMIT_METRIC_SCAN*SCHRO_LIMIT_METRIC_SCAN];
  int shift_h = SCHRO_FRAME_FORMAT_H_SHIFT(scan->frame->format)
    , shift_v = SCHRO_FRAME_FORMAT_V_SHIFT(scan->frame->format);
  int components_number = use_chroma ? 3 : 1;
  orc_splat_u8_ns ((uint8_t *)scan->metrics, 0,
      sizeof(uint32_t)*scan->scan_width * scan->scan_height);
  for (k=0; components_number>k; ++k) {
    SchroFrameData *fd = scan->frame->components + k;
    SchroFrameData *fd_ref = scan->ref_frame->components + k;
    int i,j;
    int block_width = 0 == k ? scan->block_width : scan->block_width >> shift_h
      , block_height = 0 == k ? scan->block_height : scan->block_height >> shift_v
      , x = 0 == k ? scan->x : scan->x >> shift_h
      , y = 0 == k ? scan->y : scan->y >> shift_v
      , ref_x = 0 == k ? scan->ref_x : scan->ref_x >> shift_h
      , ref_y = 0 == k ? scan->ref_y : scan->ref_y >> shift_v;

    if (block_width == 8 && block_height == 8) {
      for(j=0;j<scan->scan_height;j++){
      for(i=0;i<scan->scan_width;i++){
        orc_sad_8x8_u8 (metrics + i * scan->scan_height + j,
            SCHRO_FRAME_DATA_GET_PIXEL_U8(fd, x, y),
            fd->stride,
            SCHRO_FRAME_DATA_GET_PIXEL_U8(fd_ref, ref_x + i, ref_y + j),
            fd_ref->stride);
      }
      }
      for (i=0; scan->scan_width * scan->scan_height>i; ++i) {
        scan->metrics[i] += metrics[i];
      }
    } else {

      uint32_t tmp;
      for(i=0;i<scan->scan_width;i++) {
        for(j=0;j<scan->scan_height;j++) {
          tmp = schro_metric_absdiff_u8 (
              SCHRO_FRAME_DATA_GET_PIXEL_U8(fd, x, y),
              fd->stride,
              SCHRO_FRAME_DATA_GET_PIXEL_U8(fd_ref, ref_x + i,
                ref_y + j), fd_ref->stride,
              block_width, block_height);
          scan->metrics[i*scan->scan_height+j] += tmp;
        }
      }
    }
  }
}

/* Note: dx and dy should contain the seed-MV so that we can
 * bias the search towards that point - a better solution needed - FIXME */
int
schro_metric_scan_get_min (SchroMetricScan *scan, int *dx, int *dy)
{
  int i,j;
  uint32_t min_metric;
  int min_gravity;
  uint32_t metric;
  int gravity;
  int x,y;

  SCHRO_ASSERT (scan->scan_width > 0);
  SCHRO_ASSERT (scan->scan_height > 0);

  i = *dx + scan->x - scan->ref_x;
  j = *dy + scan->y - scan->ref_y;
  min_metric = scan->metrics[j + i * scan->scan_height];
  min_gravity = scan->gravity_scale *
    (abs(*dx - scan->gravity_x) + abs(*dy - scan->gravity_y));

  for(i=0;i<scan->scan_width;i++) {
    for(j=0;j<scan->scan_height;j++) {
      metric = scan->metrics[i*scan->scan_height + j];
      x = scan->ref_x + i - scan->x;
      y = scan->ref_y + j - scan->y;
      gravity = scan->gravity_scale *
        (abs(x - scan->gravity_x) + abs(y - scan->gravity_y));
      //if (metric + gravity < min_metric + min_gravity) {
      if (metric < min_metric) {
        min_metric = metric;
        min_gravity = gravity;
        *dx = x;
        *dy = y;
      }
    }
  }
  return min_metric;
}


void
schro_metric_scan_setup (SchroMetricScan *scan, int dx, int dy, int dist)
{
  int xmin;
  int xmax;
  int ymin;
  int ymax;
  int xrange;
  int yrange;

  SCHRO_ASSERT(scan && scan->frame && scan->ref_frame && dist > 0);

  xmin = MAX (-scan->block_width, scan->x + dx - dist);
  xmax = MIN (scan->frame->width, scan->x + dx + dist);
  ymin = MAX (-scan->block_height, scan->y + dy - dist);
  ymax = MIN (scan->frame->height, scan->y + dy + dist);
  xmin = MAX (xmin, -scan->frame->extension);
  ymin = MAX (ymin, -scan->frame->extension);
  xmax = MIN (xmax, scan->frame->width - scan->block_width + scan->frame->extension);
  ymax = MIN (ymax, scan->frame->height - scan->block_height + scan->frame->extension);

  xrange = xmax - xmin;
  yrange = ymax - ymin;

  /* sets ref_x and ref_y */
  scan->ref_x = xmin;
  scan->ref_y = ymin;

  scan->scan_width = xrange + 1;
  scan->scan_height = yrange + 1;

  SCHRO_ASSERT (scan->scan_width <= SCHRO_LIMIT_METRIC_SCAN);
  SCHRO_ASSERT (scan->scan_height <= SCHRO_LIMIT_METRIC_SCAN);
}



int
schro_metric_get (SchroFrameData *src1, SchroFrameData *src2, int width,
    int height)
{
  uint32_t metric = 0;

#if 0
  SCHRO_ASSERT(src1->width >= width);
  SCHRO_ASSERT(src1->height >= height);
  SCHRO_ASSERT(src2->width >= width);
  SCHRO_ASSERT(src2->height >= height);
#endif

  if (height == 8 && width == 8) {
    orc_sad_8x8_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride);
  } else if (height == 12 && width == 12) {
    orc_sad_12x12_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride);
  } else if (width == 16) {
    orc_sad_16xn_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride,
        height);
#if 0
  } else if (width == 32) {
    orc_sad_32xn_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride,
        height);
#endif
  } else {
    orc_sad_nxm_u8 (&metric,
        src1->data, src1->stride, src2->data, src2->stride,
        width, height);
  }

  return metric;
}

int
schro_metric_get_dc (SchroFrameData *src, int value, int width, int height)
{
  int i,j;
  int metric = 0;
  uint8_t *line;

  SCHRO_ASSERT(src->width >= width);
  SCHRO_ASSERT(src->height >= height);

  for(j=0;j<height;j++){
    line = SCHRO_FRAME_DATA_GET_LINE(src, j);
    for(i=0;i<width;i++){
      metric += abs(value - line[i]);
    }
  }
  return metric;
}

int schro_metric_get_biref (SchroFrameData *fd, SchroFrameData *src1,
    int weight1, SchroFrameData *src2, int weight2, int shift, int width,
    int height)
{
  int i,j;
  int metric = 0;
  uint8_t *line;
  uint8_t *src1_line;
  uint8_t *src2_line;
  int offset = (1<<(shift-1));
  int x;

#if 0
  SCHRO_ASSERT(fd->width >= width);
  SCHRO_ASSERT(fd->height >= height);
  SCHRO_ASSERT(src1->width >= width);
  SCHRO_ASSERT(src1->height >= height);
  SCHRO_ASSERT(src2->width >= width);
  SCHRO_ASSERT(src2->height >= height);
#endif

  for(j=0;j<height;j++){
    line = SCHRO_FRAME_DATA_GET_LINE(fd, j);
    src1_line = SCHRO_FRAME_DATA_GET_LINE(src1, j);
    src2_line = SCHRO_FRAME_DATA_GET_LINE(src2, j);
    for(i=0;i<width;i++){
      x = (src1_line[i]*weight1 + src2_line[i]*weight2 + offset)>>shift;
      metric += abs(line[i] - x);
    }
  }
  return metric;
}

