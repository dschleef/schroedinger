
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
carid_deinterleave_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr>>=1;
  for(i=0;i<n/2;i++) {
    d_n[i*dstr] = s_n[2*i];
    d_n[(n/2 + i)*dstr] = s_n[2*i + 1];
  }
}

void
carid_interleave_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr>>=1;
  for(i=0;i<n/2;i++) {
    d_n[2*i] = s_n[i*sstr];
    d_n[2*i + 1] = s_n[(n/2 + i)*sstr];
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
carid_lift_synth_daub97_ext (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  /* predict */
  d_n[0] = s_n[0] - ((1817 * s_n[1]) >> 11);
  for(i=2;i<n;i+=2){
    d_n[i] = s_n[i] - ((1817 * (s_n[i-1] + s_n[i+1])) >> 12);
  }
  for(i=1;i<n-2;i+=2){
    d_n[i] = s_n[i] - ((3616 * (d_n[i-1] + d_n[i+1])) >> 12);
  }
  d_n[n-1] = s_n[n-1] - ((3616 * d_n[n-2]) >> 11);

  /* update */
  d_n[0] += (217 * d_n[1]) >> 11;
  for(i=2;i<n;i+=2){
    d_n[i] += (217 * (d_n[i-1] + d_n[i+1])) >> 12;
  }
  for(i=1;i<n-2;i+=2){
    d_n[i] += (6497 * (d_n[i-1] + d_n[i+1])) >> 12;
  }
  d_n[n-1] += (6497 * d_n[n-2]) >> 11;
}

void
carid_lift_split_daub97_ext (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  /* predict */
  for(i=1;i<n-2;i+=2){
    d_n[i] = s_n[i] - ((6497 * (s_n[i-1] + s_n[i+1])) >> 12);
  }
  d_n[n-1] = s_n[n-1] - ((6497 * s_n[n-2]) >> 11);
  d_n[0] = s_n[0] - ((217 * d_n[1]) >> 11);
  for(i=2;i<n;i+=2){
    d_n[i] = s_n[i] - ((217 * (d_n[i-1] + d_n[i+1])) >> 12);
  }

  /* update */
  for(i=1;i<n-2;i+=2){
    d_n[i] += (3616 * (d_n[i-1] + d_n[i+1])) >> 12;
  }
  d_n[n-1] += (3616 * d_n[n-2]) >> 11;
  d_n[0] += (1817 * d_n[1]) >> 11;
  for(i=2;i<n;i+=2){
    d_n[i] += (1817 * (d_n[i-1] + d_n[i+1])) >> 12;
  }
}

void
carid_wt_daub97 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=2) {
    carid_lift_split_daub97_ext (i_n, i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iwt_daub97 (int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = 2;
  while(m<=n) {
    memcpy(tmp,i_n,m*2);
    carid_interleave (i_n, tmp, m);
    carid_lift_synth_daub97_ext (i_n, i_n, m);
    m<<=1;
  }
}

void
carid_lift_split_approx97_ext (int16_t *i_n, int n)
{
  int i;

  if (n==2) {
    i_n[1] -= i_n[0];
    i_n[0] += i_n[1] >> 1;
  } else if (n==4) {
    /* predict */
    i_n[1] -= (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[2])) >> 4;
    i_n[3] -= (9*i_n[2] - i_n[0]) >> 3;

    /* update */
    i_n[0] += i_n[1] >> 1;
    i_n[2] += (i_n[1] + i_n[3]) >> 2;
  } else {
    /* predict */
    i_n[1] -= (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[4])) >> 4;
    for(i=3;i<n-4;i+=2){
      i_n[i] -= (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 4;
    }
    i_n[n-3] -= (9*(i_n[n-4] + i_n[n-2]) - (i_n[n-6] + i_n[n-2])) >> 4;
    i_n[n-1] -= (9*i_n[n-2] - i_n[n-4]) >> 3;

    /* update */
    i_n[0] += i_n[1] >> 1;
    for(i=2;i<n;i+=2){
      i_n[i] += (i_n[i-1] + i_n[i+1]) >> 2;
    }
  }

}

void
carid_lift_synth_approx97_ext (int16_t *i_n, int n)
{
  int i;

  if (n==2) {
    i_n[0] -= i_n[1] >> 1;
    i_n[1] += i_n[0];
  } else if (n==4) {
    /* predict */
    i_n[0] -= i_n[1] >> 1;
    i_n[2] -= (i_n[1] + i_n[3]) >> 2;

    /* update */
    i_n[1] += (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[2])) >> 4;
    i_n[3] += (9*i_n[2] - i_n[0]) >> 3;
  } else {
    /* predict */
    i_n[0] -= i_n[1] >> 1;
    for(i=2;i<n;i+=2){
      i_n[i] -= (i_n[i-1] + i_n[i+1]) >> 2;
    }

    /* update */
    i_n[1] += (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[4])) >> 4;
    for(i=3;i<n-4;i+=2){
      i_n[i] += (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 4;
    }
    i_n[n-3] += (9*(i_n[n-4] + i_n[n-2]) - (i_n[n-6] + i_n[n-2])) >> 4;
    i_n[n-1] += (9*i_n[n-2] - i_n[n-4]) >> 3;
  }
}

void
carid_wt_approx97 (int16_t *i_n, int n)
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
carid_iwt_approx97 (int16_t *i_n, int n)
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

#if 0
/* original */
void
carid_lift_split_53_ext (int16_t *i_n, int n)
{
  int i;

  /* predict */
  for(i=1;i<n-2;i+=2){
    i_n[i] -= (i_n[i-1] + i_n[i+1]) >> 1;
  }
  i_n[n-1] -= i_n[n-2];

  /* update */
  i_n[0] += i_n[1] >> 1;
  for(i=2;i<n;i+=2){
    i_n[i] += (i_n[i-1] + i_n[i+1]) >> 2;
  }
}

void
carid_lift_synth_53_ext (int16_t *i_n, int n)
{
  int i;

  /* predict */
  i_n[0] -= i_n[1] >> 1;
  for(i=2;i<n;i+=2){
    i_n[i] -= (i_n[i-1] + i_n[i+1]) >> 2;
  }

  /* update */
  for(i=1;i<n-2;i+=2){
    i_n[i] += (i_n[i+1] + i_n[i-1]) >> 1;
  }
  i_n[n-1] += i_n[n-2];
}
#endif

void
carid_lift_split_53_ext (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n == 2) {
    d_n[1] = s_n[1] - s_n[0];
    d_n[0] = s_n[0] + (d_n[1] >> 1);
  } else {
    d_n[1] = s_n[1] - ((s_n[0] + s_n[2]) >> 1);
    d_n[0] = s_n[0] + (d_n[1] >> 1);
    for(i=2;i<n-2;i+=2){
      d_n[i+1] = s_n[i+1] - ((s_n[i] + s_n[i+2]) >> 1);
      d_n[i] = s_n[i] + ((d_n[i-1] + d_n[i+1]) >> 2);
    }
    d_n[n-1] = s_n[n-1] - s_n[n-2];
    d_n[n-2] = s_n[n-2] + ((d_n[n-3] + d_n[n-1]) >> 2);
  }
}

void
carid_lift_synth_53_ext (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n == 2) {
    d_n[0] = s_n[0] - (s_n[1] >> 1);
    d_n[1] = s_n[1] + d_n[0];
  } else {
    d_n[0] = s_n[0] - (s_n[1] >> 1);
    for(i=2;i<n-2;i+=2){
      d_n[i] = s_n[i] - ((s_n[i-1] + s_n[i+1]) >> 2);
      d_n[i-1] = s_n[i-1] + ((d_n[i] + d_n[i-2]) >> 1);
    }
    d_n[n-2] = s_n[n-2] - ((s_n[n-3] + s_n[n-1]) >> 2);
    d_n[n-3] = s_n[n-3] + ((d_n[n-2] + d_n[n-4]) >> 1);
    d_n[n-1] = s_n[n-1] + d_n[n-2];
  }
}

void
carid_lift_split_53_ext_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr>>=1;
  if (n == 2) {
    d_n[1] = s_n[1*sstr] - s_n[0*sstr];
    d_n[0] = s_n[0*sstr] + (d_n[1] >> 1);
  } else {
    d_n[1] = s_n[1*sstr] - ((s_n[0*sstr] + s_n[2*sstr]) >> 1);
    d_n[0] = s_n[0*sstr] + (d_n[1] >> 1);
    for(i=2;i<n-2;i+=2){
      d_n[i+1] = s_n[(i+1)*sstr] - ((s_n[i*sstr] + s_n[(i+2)*sstr]) >> 1);
      d_n[i] = s_n[i*sstr] + ((d_n[i-1] + d_n[i+1]) >> 2);
    }
    d_n[n-1] = s_n[(n-1)*sstr] - s_n[(n-2)*sstr];
    d_n[n-2] = s_n[(n-2)*sstr] + ((d_n[n-3] + d_n[n-1]) >> 2);
  }
}

void
carid_lift_synth_53_ext_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr>>=1;

  if (n == 2) {
    d_n[0*dstr] = s_n[0] - (s_n[1] >> 1);
    d_n[1*dstr] = s_n[1] + s_n[0];
  } else {
    d_n[0*dstr] = s_n[0] - (s_n[1] >> 1);
    for(i=2;i<n-2;i+=2){
      d_n[i*dstr] = s_n[i] - ((s_n[i-1] + s_n[i+1]) >> 2);
      d_n[(i-1)*dstr] = s_n[i-1] + ((d_n[i*dstr] + d_n[(i-2)*dstr]) >> 1);
    }
    d_n[(n-2)*dstr] = s_n[n-2] - ((s_n[n-3] + s_n[n-1]) >> 2);
    d_n[(n-3)*dstr] = s_n[n-3] + ((d_n[(n-2)*dstr] + d_n[(n-4)*dstr]) >> 1);
    d_n[(n-1)*dstr] = s_n[n-1] + d_n[(n-2)*dstr];
  }
}

#if 0
void
carid_wt_5_3 (int16_t *i_n, int n)
{
  int16_t tmp[256];

  while(n>=2) {
    carid_lift_split_53_ext (i_n, i_n, n);
    carid_deinterleave (tmp, i_n, n);
    memcpy(i_n,tmp,n*2);
    n>>=1;
  }
}

void
carid_iwt_5_3 (int16_t *i_n, int n)
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
#endif


void
carid_lift_split_135_ext (int16_t *i_n, int n)
{
  int i;

  if (n==2) {
    i_n[1] -= i_n[0];
    i_n[0] += i_n[1]>>1;
  } else if (n==4) {
    /* predict */
    i_n[1] -= (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[2])) >> 4;
    i_n[3] -= (9*i_n[2] - i_n[0]) >> 3;

    /* update */
    i_n[0] += (9*i_n[1] - i_n[3]) >> 4;
    i_n[2] += (9*(i_n[1] + i_n[3]) - (i_n[1] + i_n[1])) >> 5;
  } else {
    /* predict */
    i_n[1] -= (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[4])) >> 4;
    for(i=3;i<n-4;i+=2){
      i_n[i] -= (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 4;
    }
    i_n[n-3] -= (9*(i_n[n-4] + i_n[n-2]) - (i_n[n-6] + i_n[n-2])) >> 4;
    i_n[n-1] -= (9*i_n[n-2] - i_n[n-4]) >> 3;

    /* update */
    i_n[0] += (9*i_n[1] - i_n[3]) >> 4;
    i_n[2] += (9*(i_n[1] + i_n[3]) - (i_n[1] + i_n[5])) >> 5;
    for(i=4;i<n-2;i+=2){
      i_n[i] += (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 5;
    }
    i_n[n-2] += (9*(i_n[n-3] + i_n[n-1]) - (i_n[n-5] + i_n[n-1])) >> 5;
  }

}


void
carid_lift_synth_135_ext (int16_t *i_n, int n)
{
  int i;

  if (n==2) {
    i_n[0] -= i_n[1]>>1;
    i_n[1] += i_n[0];
  } else if (n==4) {
    /* predict */
    i_n[0] -= (9*i_n[1] - i_n[3]) >> 4;
    i_n[2] -= (9*(i_n[1] + i_n[3]) - (i_n[1] + i_n[1])) >> 5;

    /* update */
    i_n[1] += (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[2])) >> 4;
    i_n[3] += (9*i_n[2] - i_n[0]) >> 3;
  } else {
    /* predict */
    i_n[0] -= (9*i_n[1] - i_n[3]) >> 4;
    i_n[2] -= (9*(i_n[1] + i_n[3]) - (i_n[1] + i_n[5])) >> 5;
    for(i=4;i<n-2;i+=2){
      i_n[i] -= (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 5;
    }
    i_n[n-2] -= (9*(i_n[n-3] + i_n[n-1]) - (i_n[n-5] + i_n[n-1])) >> 5;

    /* update */
    i_n[1] += (9*(i_n[0] + i_n[2]) - (i_n[2] + i_n[4])) >> 4;
    for(i=3;i<n-4;i+=2){
      i_n[i] += (9*(i_n[i-1] + i_n[i+1]) - (i_n[i-3] + i_n[i+3])) >> 4;
    }
    i_n[n-3] += (9*(i_n[n-4] + i_n[n-2]) - (i_n[n-6] + i_n[n-2])) >> 4;
    i_n[n-1] += (9*i_n[n-2] - i_n[n-4]) >> 3;
  }
}

void
carid_wt_13_5 (int16_t *i_n, int n)
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
carid_iwt_13_5 (int16_t *i_n, int n)
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
carid_lift_split (int type, int16_t *d_n, int16_t *s_n, int n)
{
  switch (type) {
    case CARID_WAVELET_DAUB97:
      carid_lift_split_daub97_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_APPROX97:
      carid_lift_split_approx97_ext (i_n, n);
      break;
#endif
    case CARID_WAVELET_5_3:
      carid_lift_split_53_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_13_5:
      carid_lift_split_135_ext (i_n, n);
      break;
#endif
    default:
      printf("invalid type\n");
      break;
  }
}

void
carid_lift_split_str (int type, int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  switch (type) {
    case CARID_WAVELET_DAUB97:
      carid_lift_split_daub97_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_APPROX97:
      carid_lift_split_approx97_ext (i_n, n);
      break;
#endif
    case CARID_WAVELET_5_3:
      carid_lift_split_53_ext_str (d_n, s_n, sstr, n);
      break;
#if 0
    case CARID_WAVELET_13_5:
      carid_lift_split_135_ext (i_n, n);
      break;
#endif
    default:
      printf("invalid type\n");
      break;
  }
}

void
carid_lift_synth (int type, int16_t *d_n, int16_t *s_n, int n)
{
  switch (type) {
    case CARID_WAVELET_DAUB97:
      carid_lift_synth_daub97_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_APPROX97:
      carid_lift_synth_approx97_ext (i_n, n);
      break;
#endif
    case CARID_WAVELET_5_3:
      carid_lift_synth_53_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_13_5:
      carid_lift_synth_135_ext (i_n, n);
      break;
#endif
    default:
      printf("invalid type\n");
      break;
  }
}

void
carid_lift_synth_str (int type, int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  switch (type) {
    case CARID_WAVELET_DAUB97:
      carid_lift_synth_daub97_ext (d_n, s_n, n);
      break;
#if 0
    case CARID_WAVELET_APPROX97:
      carid_lift_synth_approx97_ext (i_n, n);
      break;
#endif
    case CARID_WAVELET_5_3:
      carid_lift_synth_53_ext_str (d_n, dstr, s_n, n);
      break;
#if 0
    case CARID_WAVELET_13_5:
      carid_lift_synth_135_ext (i_n, n);
      break;
#endif
    default:
      printf("invalid type\n");
      break;
  }
}

#if 0
void
carid_wt_2d (int type, int16_t *i_n, int n, int stride)
{
  int16_t tmp[256];
  int16_t tmp2[256];
  int i;
  int j;

  while(n>=MIN_SIZE) {
    for(i=0;i<n;i++) {
      memcpy (tmp, i_n + i*stride, n * 2);
      carid_lift_split (type, tmp, n);
      carid_deinterleave (i_n + i*stride, tmp, n);
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
carid_iwt_2d (int type, int16_t *i_n, int n, int stride)
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
      carid_interleave (tmp2, i_n + i*stride, m);
      carid_lift_synth (type, tmp2, m);
      memcpy (i_n + i*stride, tmp2, m * 2);
    }

    m<<=1;
  }
}
#endif

#if 0
void
carid_wt (int type, int16_t *d_n, int16_t *s_n, int n)
{
  int16_t tmp[256];

  carid_lift_split (type, tmp, s_n, n);
  carid_deinterleave (d_n, s_n, n);
}

void
carid_iwt (int type, int16_t *i_n, int n)
{
  int16_t tmp[256];
  int m;

  m = MIN_SIZE;
  while(m<=n) {
    carid_interleave (tmp, i_n, m);
    carid_lift_synth (type, tmp, m);
    memcpy (i_n, tmp, m * 2);

    m<<=1;
  }
}
#endif

