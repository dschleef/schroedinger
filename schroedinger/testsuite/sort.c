
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrofilter.h>
#include <schroedinger/schro.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#define N 9

int
main (int argc, char *argv[])
{
  int i;
  int j;
  uint8_t list[N];
  int sum1, sum2;

  srand(time(NULL));
  for (i=0;i<100;i++){
    SCHRO_ERROR("%d:", i);

    sum1 = 0;
    for(j=0;j<N;j++){
      list[j]=rand();
      SCHRO_ERROR("  %d", list[j]);
      sum1 += list[j];
    }

    sort_u8 (list, N);

    sum2 = 0;
    for(j=0;j<N;j++){
      SCHRO_ERROR("* %d", list[j]);
      sum2 += list[j];
    }
    SCHRO_ASSERT(sum1 == sum2);
    for(j=0;j<N-1;j++){
      SCHRO_ASSERT(list[j] <= list[j+1]);
    }
  }

  return 0;
}

