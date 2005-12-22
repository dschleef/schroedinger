
#include <carid/caridwavelet.h>
#include <string.h>
#include <stdio.h>

/* reflect the values past the endpoint (reflect across 0) */
#define REFLECT_EVEN(i,n) (((i)<0)?(-(i)):(((i)>(n)-1)?(2*(n)-2 - (i)):(i)))

/* reflect the values past the endpoint (reflect across -0.5) */
#define REFLECT_ODD(i,n) (((i)<0)?(-1-(i)):(((i)>(n)-1)?(2*(n)-1 - (i)):(i)))

/* continue the last value past the end */
#define CONTINUE(i,n) (((i)<0)?0:(((i)>(n)-1)?((n)-1):(i)))


//#define EXTEND(i,n) CONTINUE((i),(n))
#define EXTEND(i,n) REFLECT_EVEN((i),(n))
//#define EXTEND(i,n) REFLECT_ODD((i),(n))

/*
 * Daub97:
 *   constant input: REFLECT_ODD/CONTIUE work poorly
 *   linear: REFLECT_ODD/CONTINUE work poorly
 *
 */

#define CHECK_RANGE_S16(x) do { \
  if ((x)<-32768 || (x)>32767) { \
    printf("out of range: %d\n", x); \
  } \
} while (0)

#define MIN_SIZE 2

void
carid_deinterleave (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  for(i=0;i<n/2;i++) {
    d_n[i] = s_n[2*i];
    d_n[n/2 + i] = s_n[2*i + 1];
  }
}

void
carid_interleave (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  for(i=0;i<n/2;i++) {
    d_n[2*i] = s_n[i];
    d_n[2*i + 1] = s_n[n/2 + i];
  }
}

void
carid_lift_synth_haar (int16_t *i_n, int n)
{
  int i;

  for(i=0;i<n/2;i++){
    i_n[2*i+1] -= i_n[2*i];
  }
  for(i=0;i<n/2;i++){
    i_n[2*i] += i_n[2*i + 1]/2;
  }
}

void
carid_lift_split_haar (int16_t *i_n, int n)
{
  int i;

  for(i=0;i<n/2;i++){
    i_n[2*i+1] -= i_n[2*i];
  }
  for(i=0;i<n/2;i++){
    i_n[2*i] += i_n[2*i + 1]/2;
  }
}

void
carid_lift_synth_daub97_ext (int16_t *i_n, int n)
{
  int i;

  /* predict */
  for(i=1;i<=n-1;i+=2){
    i_n[i-1] -= (1817 * (i_n[EXTEND(i-2,n)] + i_n[i])) >> 12;
  }
  for(i=1;i<=n-1;i+=2){
    i_n[i] -= (3616 * (i_n[EXTEND(i+1,n)] + i_n[i-1])) >> 12;
  }

  /* update */
  for(i=1;i<=n-1;i+=2){
    i_n[i-1] += (217 * (i_n[EXTEND(i-2,n)] + i_n[i])) >> 12;
  }
  for(i=1;i<=n-1;i+=2){
    i_n[i] += (6497 * (i_n[EXTEND(i+1,n)] + i_n[i-1])) >> 12;
  }
}

void
carid_lift_split_daub97_ext (int16_t *i_n, int n)
{
  int i;

  /* predict */
  for(i=1;i<=n-1;i+=2){
    i_n[i] -= (6497 * (i_n[EXTEND(i+1,n)] + i_n[i-1])) >> 12;
  }
  for(i=1;i<=n-1;i+=2){
    i_n[i-1] -= (217 * (i_n[EXTEND(i-2,n)] + i_n[i])) >> 12;
  }

  /* update */
  for(i=1;i<=n-1;i+=2){
    i_n[i] += (3616 * (i_n[EXTEND(i+1,n)] + i_n[i-1])) >> 12;
  }
  for(i=1;i<=n-1;i+=2){
    i_n[i-1] += (1817 * (i_n[EXTEND(i-2,n)] + i_n[i])) >> 12;
  }

}

void
carid_iwt_daub97 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=2) {
    carid_lift_split_daub97_ext (i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iiwt_daub97 (int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = 2;
  while(m<=n) {
    memcpy(tmp,i_n,m*2);
    carid_interleave (i_n, tmp, m);
    carid_lift_synth_daub97_ext (i_n, m);
    m<<=1;
  }
}

void
carid_lift_split_approx97_ext (int16_t *i_n, int n)
{
  int i;

  for(i=1;i<n-1;i+=2){
    /* predict */
    i_n[i] -= (9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) +
        -1 * (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 4;
  }
  for(i=0;i<n;i+=2){
    /* update */
    i_n[i] += (i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) >> 2;
  }

}

void
carid_lift_synth_approx97_ext (int16_t *i_n, int n)
{
  int i;

  for(i=0;i<n;i+=2){
    /* predict */
    i_n[i] -= (i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) >> 2;
  }

  for(i=1;i<n-1;i+=2){
    /* update */
    i_n[i] += (9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) +
        -1 * (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 4;
  }

}

void
carid_iwt_approx97 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=2) {
    carid_lift_split_approx97_ext (i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iiwt_approx97 (int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = 2;
  while(m<=n) {
    memcpy(tmp,i_n,m*2);
    carid_interleave (i_n, tmp, m);
    carid_lift_synth_approx97_ext (i_n, m);
    m<<=1;
  }
}

void
carid_lift_split_53_ext (int16_t *i_n, int n)
{
  int i;

  for(i=1;i<=n-1;i+=2){
    /* predict */
    i_n[i] -= (i_n[EXTEND(i+1,n)] + i_n[i-1]) >> 1;
  }
  for(i=1;i<=n-1;i+=2){
    /* update */
    i_n[i-1] += (i_n[EXTEND(i-2,n)] + i_n[i]) >> 2;
  }

}

void
carid_lift_synth_53_ext (int16_t *i_n, int n)
{
  int i;

  for(i=1;i<=n-1;i+=2){
    /* predict */
    i_n[i-1] -= (i_n[EXTEND(i-2,n)] + i_n[i]) >> 2;
  }
  for(i=1;i<=n-1;i+=2){
    /* update */
    i_n[i] += (i_n[EXTEND(i+1,n)] + i_n[i-1]) >> 1;
  }

}

void
carid_iwt_5_3 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=2) {
    carid_lift_split_53_ext (i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iiwt_5_3 (int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = 2;
  while(m<=n) {
    memcpy(tmp,i_n,m*2);
    carid_interleave (i_n, tmp, m);
    carid_lift_synth_53_ext (i_n, m);
    m<<=1;
  }
}


void
carid_lift_split_135_ext (int16_t *i_n, int n)
{
  int i;
  int x;

  /* special case n==2, since the EXTEND() macro doesn't work */
  if (n==2) {
    i_n[1] -= i_n[0];
    i_n[0] += i_n[1]>>1;
  } else {
    for(i=1;i<=n-1;i+=2){
      /* predict */
      x = i_n[i] - ((9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) -
          (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 4);
      CHECK_RANGE_S16(x);
      i_n[i] = x;
    }
    for(i=0;i<n;i+=2){
      /* update */
      x = i_n[i] + ((9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) -
          (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 5);
      CHECK_RANGE_S16(x);
      i_n[i] = x;
    }
  }

}


void
carid_lift_synth_135_ext (int16_t *i_n, int n)
{
  int i;
  int x;

  /* special case n==2, since the EXTEND() macro doesn't work */
  if (n==2) {
    i_n[0] -= i_n[1]>>1;
    i_n[1] += i_n[0];
  } else {
    for(i=0;i<n;i+=2){
      /* predict */
      x = i_n[i] - ((9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) -
          (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 5);
      CHECK_RANGE_S16(x);
      i_n[i] = x;
    }
    for(i=1;i<=n-1;i+=2){
      /* update */
      x = i_n[i] + ((9*(i_n[EXTEND(i-1,n)] + i_n[EXTEND(i+1,n)]) -
          (i_n[EXTEND(i-3,n)] + i_n[EXTEND(i+3,n)])) >> 4);
      CHECK_RANGE_S16(x);
      i_n[i] = x;
    }
  }
}

void
carid_iwt_13_5 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=MIN_SIZE) {
    carid_lift_split_135_ext (i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iiwt_13_5 (int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = MIN_SIZE;
  while(m<=n) {
    memcpy(tmp,i_n,m*2);
    carid_interleave (i_n, tmp, m);
    carid_lift_synth_135_ext (i_n, m);
    m<<=1;
  }
}



void
carid_lift_split (int type, int16_t *i_n, int n)
{
  switch (type) {
    case CARID_WAVELET_APPROX97:
      carid_lift_split_approx97_ext (i_n, n);
      break;
    case CARID_WAVELET_DAUB97:
      carid_lift_split_daub97_ext (i_n, n);
      break;
    case CARID_WAVELET_13_5:
      carid_lift_split_135_ext (i_n, n);
      break;
    case CARID_WAVELET_5_3:
      carid_lift_split_53_ext (i_n, n);
      break;
    default:
      printf("invalid type\n");
      break;
  }
}


void
carid_lift_synth (int type, int16_t *i_n, int n)
{
  switch (type) {
    case CARID_WAVELET_APPROX97:
      carid_lift_synth_approx97_ext (i_n, n);
      break;
    case CARID_WAVELET_DAUB97:
      carid_lift_synth_daub97_ext (i_n, n);
      break;
    case CARID_WAVELET_13_5:
      carid_lift_synth_135_ext (i_n, n);
      break;
    case CARID_WAVELET_5_3:
      carid_lift_synth_53_ext (i_n, n);
      break;
    default:
      printf("invalid type\n");
      break;
  }
}

void
carid_iwt_2d (int type, int16_t *i_n, int n, int stride)
{
  int16_t tmp[256];
  int16_t tmp2[256];
  int i;
  int j;

  while(n>=MIN_SIZE) {
    for(i=0;i<n;i++) {
      for(j=0;j<n;j++) {
        tmp[j] = i_n[i*stride + j];
      }
      carid_lift_split (type, tmp, n);
      carid_deinterleave (tmp2, tmp, n);
      for(j=0;j<n;j++) {
        i_n[i*stride + j] = tmp2[j];
      }
    }
    for(i=0;i<n;i++) {
      for(j=0;j<n;j++) {
        tmp[j] = i_n[j*stride + i];
      }
      carid_lift_split (type, tmp, n);
      carid_deinterleave (tmp2, tmp, n);
      for(j=0;j<n;j++) {
        i_n[j*stride + i] = tmp2[j];
      }
    }

    n>>=1;
  }
}

void
carid_iiwt_2d (int type, int16_t *i_n, int n, int stride)
{
  int16_t tmp[256];
  int16_t tmp2[256];
  int i;
  int j;
  int m;

  m = MIN_SIZE;
  while(m<=n) {
    for(i=0;i<m;i++) {
      for(j=0;j<m;j++) {
        tmp[j] = i_n[j*stride + i];
      }
      carid_interleave (tmp2, tmp, m);
      carid_lift_synth (type, tmp2, m);
      for(j=0;j<m;j++) {
        i_n[j*stride + i] = tmp2[j];
      }
    }
    for(i=0;i<m;i++) {
      for(j=0;j<m;j++) {
        tmp[j] = i_n[i*stride + j];
      }
      carid_interleave (tmp2, tmp, m);
      carid_lift_synth (type, tmp2, m);
      for(j=0;j<m;j++) {
        i_n[i*stride + j] = tmp2[j];
      }
    }

    m<<=1;
  }
}

