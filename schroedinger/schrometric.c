
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
  } else {
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

  SCHRO_ASSERT (scan->ref_x + scan->block_width + scan->scan_width - 1 <= scan->frame->width);
  SCHRO_ASSERT (scan->ref_y + scan->block_height + scan->scan_height - 1 <= scan->frame->height);
  SCHRO_ASSERT (scan->ref_x >= 0);
  SCHRO_ASSERT (scan->ref_y >= 0);
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

  xmin = MAX (xmin, 0);
  ymin = MAX (ymin, 0);
  xmax = MIN (xmax, scan->frame->width - scan->block_width);
  ymax = MIN (ymax, scan->frame->height - scan->block_height);

  scan->ref_x = xmin;
  scan->ref_y = ymin;
  scan->scan_width = xmax - xmin + 1;
  scan->scan_height = ymax - ymin + 1;
}

