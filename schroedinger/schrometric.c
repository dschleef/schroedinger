
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <schroedinger/schro.h>
#include <liboil/liboil.h>


int
schro_metric_absdiff_u8 (uint8_t *a, int a_stride, uint8_t *b, int b_stride,
    int width, int height)
{
  int i;
  int j;
  int metric = 0;

  if (height == 8 && width == 8) {
    uint32_t m;
    oil_sad8x8_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
  } else if (height == 12 && width == 12) {
    uint32_t m;
    oil_sad12x12_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
  } else if (height == 16 && width == 16) {
    uint32_t m;
    oil_sad16x16_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
  } else if (height == 32 && width == 16) {
    uint32_t m;
    oil_sad16x16_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
    a += a_stride * 16;
    b += b_stride * 16;
    oil_sad16x16_u8 (&m, a, a_stride, b, b_stride);
    metric += m;
  } else if (height == 32 && width == 32) {
    uint32_t m;
    oil_sad16x16_u8 (&m, a, a_stride, b, b_stride);
    metric = m;
    oil_sad16x16_u8 (&m, a + 16, a_stride, b + 16, b_stride);
    metric += m;
    a += a_stride * 16;
    b += b_stride * 16;
    oil_sad16x16_u8 (&m, a, a_stride, b, b_stride);
    metric += m;
    oil_sad16x16_u8 (&m, a + 16, a_stride, b + 16, b_stride);
    metric += m;
  } else if ((height&15) == 0 && (width&15) == 0) {
    uint32_t m;
    metric = 0;
    for(j=0;j<height;j+=16){
      for(i=0;i<width;i+=16){
        oil_sad16x16_u8 (&m, a + i + j*a_stride, a_stride,
            b + i + j*b_stride, b_stride);
        metric += m;
      }
    }
  } else if ((height&7) == 0 && (width&7) == 0) {
    uint32_t m;
    metric = 0;
    for(j=0;j<height;j+=8){
      for(i=0;i<width;i+=8){
        oil_sad8x8_u8 (&m, a + i + j*a_stride, a_stride,
            b + i + j*b_stride, b_stride);
        metric += m;
      }
    }
  } else {
    //SCHRO_ERROR("slow metric %dx%d", width, height);
    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
        int x;
        x = (int)(a[j*a_stride + i]) - (int)(b[j*b_stride + i]);
        metric += (x < 0) ? -x : x;
      }
    }
  }

  return metric;
}

void
schro_metric_scan_do_scan (SchroMetricScan *scan)
{
  SchroFrameData *fd = scan->frame->components + 0;
  SchroFrameData *fd_ref = scan->ref_frame->components + 0;
  int i,j;

  SCHRO_ASSERT (scan->ref_x + scan->block_width + scan->scan_width - 1 <= scan->frame->width + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y + scan->block_height + scan->scan_height - 1 <= scan->frame->height + scan->frame->extension);
  SCHRO_ASSERT (scan->ref_x >= -scan->frame->extension);
  SCHRO_ASSERT (scan->ref_y >= -scan->frame->extension);
  SCHRO_ASSERT (scan->scan_width > 0);
  SCHRO_ASSERT (scan->scan_height > 0);

  if (scan->block_width == 8 && scan->block_height == 8) {
    for(i=0;i<scan->scan_width;i++){
      oil_sad8x8_8xn_u8 (scan->metrics + i * scan->scan_height,
          SCHRO_FRAME_DATA_GET_PIXEL_U8(fd, scan->x, scan->y),
          fd->stride,
          SCHRO_FRAME_DATA_GET_PIXEL_U8(fd_ref, scan->ref_x + i, scan->ref_y),
          fd_ref->stride,
          scan->scan_height);
    }
    return;
  }

  for(i=0;i<scan->scan_width;i++) {
    for(j=0;j<scan->scan_height;j++) {
      scan->metrics[i*scan->scan_height + j] = schro_metric_absdiff_u8 (
          SCHRO_FRAME_DATA_GET_PIXEL_U8(fd, scan->x, scan->y),
          fd->stride,
          SCHRO_FRAME_DATA_GET_PIXEL_U8(fd_ref, scan->ref_x + i,
            scan->ref_y + j), fd_ref->stride,
          scan->block_width, scan->block_height);
    }
  }
}

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

  min_metric = scan->metrics[0];
  *dx = scan->ref_x + 0 - scan->x;
  *dy = scan->ref_y + 0 - scan->y;
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
  int xmin, ymin;
  int xmax, ymax;

  xmin = scan->x + dx - dist;
  xmax = scan->x + dx + dist;
  ymin = scan->y + dy - dist;
  ymax = scan->y + dy + dist;

  xmin = MAX (xmin, -scan->frame->extension);
  ymin = MAX (ymin, -scan->frame->extension);
  xmax = MIN (xmax, scan->frame->width - scan->block_width + scan->frame->extension);
  ymax = MIN (ymax, scan->frame->height - scan->block_height + scan->frame->extension);

  scan->ref_x = xmin;
  scan->ref_y = ymin;
  scan->scan_width = xmax - xmin + 1;
  scan->scan_height = ymax - ymin + 1;

  SCHRO_ASSERT (scan->scan_width <= SCHRO_LIMIT_METRIC_SCAN);
  SCHRO_ASSERT (scan->scan_height <= SCHRO_LIMIT_METRIC_SCAN);
}

int
schro_metric_get (SchroFrameData *src1, SchroFrameData *src2, int width,
    int height)
{
  int i,j;
  int metric = 0;
  uint8_t *line1;
  uint8_t *line2;

#if 0
  SCHRO_ASSERT(src1->width >= width);
  SCHRO_ASSERT(src1->height >= height);
  SCHRO_ASSERT(src2->width >= width);
  SCHRO_ASSERT(src2->height >= height);
#endif

  if (width == 8 && height == 8) {
    uint32_t m;
    oil_sad8x8_u8 (&m, src1->data, src1->stride, src2->data, src2->stride);
    metric = m;
  } else if (height == 12 && width == 12) {
    uint32_t m;
    oil_sad12x12_u8 (&m, src1->data, src1->stride, src2->data, src2->stride);
    metric = m;
  } else if (height == 16 && width == 16) {
    uint32_t m;
    oil_sad16x16_u8 (&m, src1->data, src1->stride, src2->data, src2->stride);
    metric = m;
  } else {
    for(j=0;j<height;j++){
      line1 = SCHRO_FRAME_DATA_GET_LINE(src1, j);
      line2 = SCHRO_FRAME_DATA_GET_LINE(src2, j);
      for(i=0;i<width;i++){
        metric += abs(line1[i] - line2[i]);
      }
    }
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

  SCHRO_ASSERT(fd->width >= width);
  SCHRO_ASSERT(fd->height >= height);
  SCHRO_ASSERT(src1->width >= width);
  SCHRO_ASSERT(src1->height >= height);
  SCHRO_ASSERT(src2->width >= width);
  SCHRO_ASSERT(src2->height >= height);

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

