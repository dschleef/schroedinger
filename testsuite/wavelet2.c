
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int
check_output (int type, int split)
{
  int16_t *src;
  int16_t *dest;
  int i;
  int n;
  int ret = 1;

  src = malloc (256 * 2);
  dest = malloc (256 * 2);

  for(n=2;n<16;n+=2) {
    for(i=0;i<256;i++){
      dest[i] = 100;
      src[i] = 0;
    }

    if (split) {
      schro_lift_split (type, dest + 10, src + 10, n);
    } else {
      schro_lift_synth (type, dest + 10, src + 10, n);
    }

    for(i=10;i<10+n;i++){
      if (dest[i] != 0) {
        printf("check_output failed type=%d, split=%d, n=%d, offset=%d\n",
            type, split, n, i - 10);
        ret = 0;
        break;
      }
    }
  }

  free(src);
  free(dest);
  return ret;
}

int
check_endpoints (int type, int split)
{
  int16_t *src;
  int16_t *dest;
  int i;
  int n;
  int x;
  int ret = 1;

  src = malloc (256 * 2);
  dest = malloc (256 * 2);

  for(n=2;n<16;n+=2) {
    for(x=-5;x<5;x++){
      for(i=0;i<256;i++){
        dest[i] = 100;
        src[i] = 0;
      }
      if (x<0) {
        src[10+x] = 100;
      } else {
        src[10+n+x] = 100;
      }

      if (split) {
        schro_lift_split (type, dest + 10, src + 10, n);
      } else {
        schro_lift_synth (type, dest + 10, src + 10, n);
      }

      for(i=10;i<10+n;i++){
        if (dest[i] != 0) {
          printf("check_endpoints failed type=%d, split=%d, n=%d, x=%d\n",
              type, split, n, x);
          ret = 0;
          break;
        }
      }
    }
  }

  free(src);
  free(dest);
  return ret;
}

int
check_constant (int type)
{
  int16_t *src;
  int16_t *dest;
  int i;
  int n;
  int ret = 1;

  src = malloc (256 * 2);
  dest = malloc (256 * 2);

  for(n=2;n<16;n+=2) {
    for(i=0;i<n;i++){
      src[i] = 100;
    }

    schro_lift_split (type, dest, src, n);

    for(i=0;i<n;i+=2){
      if (dest[i] != dest[0]) {
        printf("check_constant failed type=%d, n=%d, i=%d\n",
            type, n, i);
        ret = 0;
        break;
      }
    }
    for(i=0;i<n;i+=2){
      if (dest[i+1] != 0) {
        printf("check_constant failed type=%d, n=%d, i=%d\n",
            type, n, i);
        ret = 0;
        break;
      }
    }
  }

  free(src);
  free(dest);
  return ret;
}

int
check_random (int type)
{
  int16_t *src;
  int16_t *tmp;
  int16_t *dest;
  int i;
  int n;
  int ret = 1;

  src = malloc (256 * 2);
  tmp = malloc (256 * 2);
  dest = malloc (256 * 2);

  for(n=2;n<16;n+=2) {
    for(i=0;i<n;i++){
      src[i] = rand()&0xff;
    }

    schro_lift_split (type, tmp, src, n);
    schro_lift_synth (type, dest, tmp, n);

    for(i=0;i<n;i++){
      if (dest[i] != src[i]) {
        printf("check_random failed type=%d, n=%d, i=%d\n", type, n, i);
        ret = 0;
        break;
      }
    }
  }

  free(src);
  free(tmp);
  free(dest);
  return ret;
}

int
main (int argc, char *argv[])
{
  schro_init();

  check_output (SCHRO_WAVELET_DAUBECHIES_9_7, 1);
  check_output (SCHRO_WAVELET_DAUBECHIES_9_7, 0);

  check_output (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7, 1);
  check_output (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7, 0);

  check_output (SCHRO_WAVELET_LE_GALL_5_3, 1);
  check_output (SCHRO_WAVELET_LE_GALL_5_3, 0);

  check_output (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7, 1);
  check_output (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7, 0);
  
  check_endpoints (SCHRO_WAVELET_DAUBECHIES_9_7, 1);
  check_endpoints (SCHRO_WAVELET_DAUBECHIES_9_7, 0);

  check_endpoints (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7, 1);
  check_endpoints (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7, 0);

  check_endpoints (SCHRO_WAVELET_LE_GALL_5_3, 1);
  check_endpoints (SCHRO_WAVELET_LE_GALL_5_3, 0);

  check_endpoints (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7, 1);
  check_endpoints (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7, 0);
  

  check_constant (SCHRO_WAVELET_DAUBECHIES_9_7);
  check_constant (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7);
  check_constant (SCHRO_WAVELET_LE_GALL_5_3);
  check_constant (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7);


  check_random (SCHRO_WAVELET_DAUBECHIES_9_7);
  check_random (SCHRO_WAVELET_DESLAURIERS_DUBUC_9_7);
  check_random (SCHRO_WAVELET_LE_GALL_5_3);
  check_random (SCHRO_WAVELET_DESLAURIERS_DUBUC_13_7);


  return 0;
}


