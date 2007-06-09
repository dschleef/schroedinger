
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrofilter.h>
#include <stdlib.h>
#include <string.h>

void
filter_cwm (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n)
{
  int i;
  int min, max;

  for(i=0;i<n;i++){
    min = MIN(s1[i+0],s1[i+1]);
    max = MAX(s1[i+0],s1[i+1]);
    min = MIN(min,s1[i+2]);
    max = MAX(max,s1[i+2]);
    min = MIN(min,s2[i+0]);
    max = MAX(max,s2[i+0]);
    min = MIN(min,s2[i+2]);
    max = MAX(max,s2[i+2]);
    min = MIN(min,s3[i+0]);
    max = MAX(max,s3[i+0]);
    min = MIN(min,s3[i+1]);
    max = MAX(max,s3[i+1]);
    min = MIN(min,s3[i+2]);
    max = MAX(max,s3[i+3]);

    d[i] = MIN(max,MAX(min,s2[i+1]));
  }
}

void
schro_frame_component_filter_cwm (SchroFrameComponent *comp)
{
  int i;
  uint8_t *tmp;
  uint8_t *tmp1;
  uint8_t *tmp2;

  tmp1 = malloc(comp->width);
  tmp2 = malloc(comp->width);

  filter_cwm (tmp1,
      OFFSET(comp->data, comp->stride * 0),
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2), comp->width - 2);
  filter_cwm (tmp2,
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2),
      OFFSET(comp->data, comp->stride * 3), comp->width - 2);

  for(i=3;i<comp->height - 1;i++) {
    memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
        tmp1, comp->width - 2);
    tmp = tmp1; tmp1 = tmp2; tmp2 = tmp;

    filter_cwm (tmp2,
        OFFSET(comp->data, comp->stride * (i-1)),
        OFFSET(comp->data, comp->stride * (i+0)),
        OFFSET(comp->data, comp->stride * (i+1)), comp->width - 2);
  }
  memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
      tmp1, comp->width - 2);
  memcpy (OFFSET(comp->data, comp->stride * (i-1) + 1),
      tmp2, comp->width - 2);

  free (tmp1);
  free (tmp2);
}

void
schro_frame_filter_cwm (SchroFrame *frame)
{
  schro_frame_component_filter_cwm (&frame->components[0]);
  schro_frame_component_filter_cwm (&frame->components[1]);
  schro_frame_component_filter_cwm (&frame->components[2]);
}

