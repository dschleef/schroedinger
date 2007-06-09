
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrofilter.h>
#include <stdlib.h>
#include <string.h>

void
sort_u8 (uint8_t *d, int n)
{
  int start = 0;
  int end = n;
  int i;
  int x;

  /* OMG bubble sort! */
  while(start < end) {
    for(i=start;i<end-1;i++){
      if (d[i] > d[i+1]) {
        x = d[i];
        d[i] = d[i+1];
        d[i+1] = x;
      }
    }
    end--;
    for(i=end-2;i>=start;i--){
      if (d[i] > d[i+1]) {
        x = d[i];
        d[i] = d[i+1];
        d[i+1] = x;
      }
    }
    start++;
  }
}

/* reference */
void
schro_filter_cwmN_ref (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n, int weight)
{
  int i;
  int j;
  uint8_t list[8+12];

  for(i=0;i<n;i++){
    list[0] = s1[i+0];
    list[1] = s1[i+1];
    list[2] = s1[i+2];
    list[3] = s2[i+0];
    list[4] = s2[i+2];
    list[5] = s3[i+0];
    list[6] = s3[i+1];
    list[7] = s3[i+2];
    for(j=0;j<weight;j++){
      list[8+j] = s2[i+1];
    }

    sort_u8 (list, 8+weight);

    d[i] = list[(8+weight)/2];
  }
}

void
schro_filter_cwmN (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n, int weight)
{
  int i;
  int j;
  uint8_t list[8+12];
  int low, hi;

  for(i=0;i<n;i++){
    list[0] = s1[i+0];
    list[1] = s1[i+1];
    list[2] = s1[i+2];
    list[3] = s2[i+0];
    list[4] = s2[i+2];
    list[5] = s3[i+0];
    list[6] = s3[i+1];
    list[7] = s3[i+2];

    low = 0;
    hi = 0;
    for(j=0;j<8;j++){
      if (list[j] < s2[i+1]) low++;
      if (list[j] > s2[i+1]) hi++;
    }

    if (low < ((9-weight)/2) || hi < ((9-weight)/2)) {
      for(j=0;j<weight;j++){
        list[8+j] = s2[i+1];
      }

      sort_u8 (list, 8+weight);

      d[i] = list[(8+weight)/2];
    } else {
      d[i] = s2[i+1];
    }
  }
}

void
schro_frame_component_filter_cwmN (SchroFrameComponent *comp, int weight)
{
  int i;
  uint8_t *tmp;
  uint8_t *tmp1;
  uint8_t *tmp2;

  tmp1 = malloc(comp->width);
  tmp2 = malloc(comp->width);

  schro_filter_cwmN (tmp1,
      OFFSET(comp->data, comp->stride * 0),
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2), comp->width - 2, weight);
  schro_filter_cwmN (tmp2,
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2),
      OFFSET(comp->data, comp->stride * 3), comp->width - 2, weight);

  for(i=3;i<comp->height - 1;i++) {
    memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
        tmp1, comp->width - 2);
    tmp = tmp1; tmp1 = tmp2; tmp2 = tmp;

    schro_filter_cwmN (tmp2,
        OFFSET(comp->data, comp->stride * (i-1)),
        OFFSET(comp->data, comp->stride * (i+0)),
        OFFSET(comp->data, comp->stride * (i+1)), comp->width - 2, weight);
  }
  memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
      tmp1, comp->width - 2);
  memcpy (OFFSET(comp->data, comp->stride * (i-1) + 1),
      tmp2, comp->width - 2);

  free (tmp1);
  free (tmp2);
}

void
schro_frame_filter_cwmN (SchroFrame *frame, int weight)
{
  schro_frame_component_filter_cwmN (&frame->components[0], weight);
  schro_frame_component_filter_cwmN (&frame->components[1], weight);
  schro_frame_component_filter_cwmN (&frame->components[2], weight);
}


void
schro_frame_component_filter_cwmN_ref (SchroFrameComponent *comp, int weight)
{
  int i;
  uint8_t *tmp;
  uint8_t *tmp1;
  uint8_t *tmp2;

  tmp1 = malloc(comp->width);
  tmp2 = malloc(comp->width);

  schro_filter_cwmN_ref (tmp1,
      OFFSET(comp->data, comp->stride * 0),
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2), comp->width - 2, weight);
  schro_filter_cwmN_ref (tmp2,
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2),
      OFFSET(comp->data, comp->stride * 3), comp->width - 2, weight);

  for(i=3;i<comp->height - 1;i++) {
    memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
        tmp1, comp->width - 2);
    tmp = tmp1; tmp1 = tmp2; tmp2 = tmp;

    schro_filter_cwmN_ref (tmp2,
        OFFSET(comp->data, comp->stride * (i-1)),
        OFFSET(comp->data, comp->stride * (i+0)),
        OFFSET(comp->data, comp->stride * (i+1)), comp->width - 2, weight);
  }
  memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
      tmp1, comp->width - 2);
  memcpy (OFFSET(comp->data, comp->stride * (i-1) + 1),
      tmp2, comp->width - 2);

  free (tmp1);
  free (tmp2);
}

void
schro_frame_filter_cwmN_ref (SchroFrame *frame, int weight)
{
  schro_frame_component_filter_cwmN_ref (&frame->components[0], weight);
  schro_frame_component_filter_cwmN_ref (&frame->components[1], weight);
  schro_frame_component_filter_cwmN_ref (&frame->components[2], weight);
}


#if 0
/* reference */
void
schro_filter_cwm7 (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n)
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
    max = MAX(max,s3[i+2]);

    d[i] = MIN(max,MAX(min,s2[i+1]));
  }
}
#endif

void
schro_filter_cwm7 (uint8_t *d, uint8_t *s1, uint8_t *s2, uint8_t *s3, int n)
{
  int i;
  int min, max;

  for(i=0;i<n;i++){
    if (s1[i+0] < s2[i+1]) {
      max = MAX(s1[i+0],s1[i+1]);
      max = MAX(max,s1[i+2]);
      max = MAX(max,s2[i+0]);
      max = MAX(max,s2[i+2]);
      max = MAX(max,s3[i+0]);
      max = MAX(max,s3[i+1]);
      max = MAX(max,s3[i+2]);
      d[i] = MIN(max,s2[i+1]);
    } else if (s1[i+0] > s2[i+1]) {
      min = MIN(s1[i+0],s1[i+1]);
      min = MIN(min,s1[i+2]);
      min = MIN(min,s2[i+0]);
      min = MIN(min,s2[i+2]);
      min = MIN(min,s3[i+0]);
      min = MIN(min,s3[i+1]);
      min = MIN(min,s3[i+2]);
      d[i] = MAX(min,s2[i+1]);
    } else {
      d[i] = s2[i+1];
    }
  }
}

void
schro_frame_component_filter_cwm7 (SchroFrameComponent *comp)
{
  int i;
  uint8_t *tmp;
  uint8_t *tmp1;
  uint8_t *tmp2;

  tmp1 = malloc(comp->width);
  tmp2 = malloc(comp->width);

  schro_filter_cwm7 (tmp1,
      OFFSET(comp->data, comp->stride * 0),
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2), comp->width - 2);
  schro_filter_cwm7 (tmp2,
      OFFSET(comp->data, comp->stride * 1),
      OFFSET(comp->data, comp->stride * 2),
      OFFSET(comp->data, comp->stride * 3), comp->width - 2);

  for(i=3;i<comp->height - 1;i++) {
    memcpy (OFFSET(comp->data, comp->stride * (i-2) + 1),
        tmp1, comp->width - 2);
    tmp = tmp1; tmp1 = tmp2; tmp2 = tmp;

    schro_filter_cwm7 (tmp2,
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
schro_frame_filter_cwm7 (SchroFrame *frame)
{
  schro_frame_component_filter_cwm7 (&frame->components[0]);
  schro_frame_component_filter_cwm7 (&frame->components[1]);
  schro_frame_component_filter_cwm7 (&frame->components[2]);
}

