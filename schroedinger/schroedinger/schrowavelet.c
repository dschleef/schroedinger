
#include <schroedinger/schrointernal.h>
#include <string.h>
#include <stdio.h>
#include <liboil/liboil.h>

#define MIN_SIZE 2

void
schro_deinterleave (int16_t *d_n, int16_t *s_n, int n)
{
  oil_deinterleave (d_n, s_n, n/2);
}

void
schro_deinterleave_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr>>=1;
  for(i=0;i<n/2;i++) {
    d_n[i*dstr] = s_n[2*i];
    d_n[(n/2 + i)*dstr] = s_n[2*i + 1];
  }
}

void
schro_interleave_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr>>=1;
  for(i=0;i<n/2;i++) {
    d_n[2*i] = s_n[i*sstr];
    d_n[2*i + 1] = s_n[(n/2 + i)*sstr];
  }
}

void
schro_interleave (int16_t *d_n, int16_t *s_n, int n)
{
  oil_interleave (d_n, s_n, n/2);
}

void
schro_lift_synth_daub97 (int16_t *d_n, int16_t *s_n, int n)
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
schro_lift_split_daub97 (int16_t *d_n, int16_t *s_n, int n)
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
schro_lift_synth_daub97_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr>>=1;

  /* predict */
  d_n[0*dstr] = s_n[0] - ((1817 * s_n[1]) >> 11);
  for(i=2;i<n;i+=2){
    d_n[i*dstr] = s_n[i] - ((1817 * (s_n[i-1] + s_n[i+1])) >> 12);
  }
  for(i=1;i<n-2;i+=2){
    d_n[i*dstr] = s_n[i] - ((3616 * (d_n[(i-1)*dstr] + d_n[(i+1)*dstr])) >> 12);
  }
  d_n[(n-1)*dstr] = s_n[n-1] - ((3616 * d_n[(n-2)*dstr]) >> 11);

  /* update */
  d_n[0*dstr] += (217 * d_n[1*dstr]) >> 11;
  for(i=2;i<n;i+=2){
    d_n[i*dstr] += (217 * (d_n[(i-1)*dstr] + d_n[(i+1)*dstr])) >> 12;
  }
  for(i=1;i<n-2;i+=2){
    d_n[i*dstr] += (6497 * (d_n[(i-1)*dstr] + d_n[(i+1)*dstr])) >> 12;
  }
  d_n[(n-1)*dstr] += (6497 * d_n[(n-2)*dstr]) >> 11;
}

void
schro_lift_split_daub97_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr>>=1;

  /* predict */
  for(i=1;i<n-2;i+=2){
    d_n[i] = s_n[i*sstr] - ((6497 * (s_n[(i-1)*sstr] + s_n[(i+1)*sstr])) >> 12);
  }
  d_n[n-1] = s_n[(n-1)*sstr] - ((6497 * s_n[(n-2)*sstr]) >> 11);
  d_n[0] = s_n[0*sstr] - ((217 * d_n[1]) >> 11);
  for(i=2;i<n;i+=2){
    d_n[i] = s_n[i*sstr] - ((217 * (d_n[i-1] + d_n[i+1])) >> 12);
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
schro_lift_split_desl93 (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n==2) {
    d_n[1] = s_n[1] - s_n[0];
    d_n[0] = s_n[0] + ((d_n[1] + 1) >> 1);
  } else if (n==4) {
    /* predict */
    d_n[1] = s_n[1] - ((9*(s_n[0] + s_n[2]) - (s_n[2] + s_n[2]) + 8) >> 4);
    d_n[3] = s_n[3] - ((9*s_n[2] - s_n[0] + 4) >> 3);

    /* update */
    d_n[0] = s_n[0] + ((d_n[1] + 1) >> 1);
    d_n[2] = s_n[2] + ((d_n[1] + d_n[3] + 2) >> 2);
  } else {
    /* predict */
    d_n[1] = s_n[1] - ((9*(s_n[0] + s_n[2]) - (s_n[2] + s_n[4]) + 8) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i] - ((9*(s_n[i-1] + s_n[i+1]) - (s_n[i-3] + s_n[i+3]) + 8) >> 4);
    }
    d_n[n-3] = s_n[n-3] - ((9*(s_n[n-4] + s_n[n-2]) - (s_n[n-6] + s_n[n-2]) + 8) >> 4);
    d_n[n-1] = s_n[n-1] - ((9*s_n[n-2] - s_n[n-4] + 4) >> 3);

    /* update */
    d_n[0] = s_n[0] + ((d_n[1] + 1) >> 1);
    for(i=2;i<n;i+=2){
      d_n[i] = s_n[i] + ((d_n[i-1] + d_n[i+1] + 2) >> 2);
    }
  }

}

void
schro_lift_synth_desl93 (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n==2) {
    d_n[0] = s_n[0] - ((s_n[1] + 1) >> 1);
    d_n[1] = s_n[1] + d_n[0];
  } else if (n==4) {
    /* predict */
    d_n[0] = s_n[0] - ((s_n[1] + 1)>> 1);
    d_n[2] = s_n[2] - ((s_n[1] + s_n[3] + 2) >> 2);

    /* update */
    d_n[1] = s_n[1] + ((9*(d_n[0] + d_n[2]) - (d_n[2] + d_n[2]) + 8) >> 4);
    d_n[3] = s_n[3] + ((9*d_n[2] - d_n[0] + 4) >> 3);
  } else {
    /* predict */
    d_n[0] = s_n[0] - ((s_n[1] + 1) >> 1);
    for(i=2;i<n;i+=2){
      d_n[i] = s_n[i] - ((s_n[i-1] + s_n[i+1] + 2) >> 2);
    }

    /* update */
    d_n[1] = s_n[1] + ((9*(d_n[0] + d_n[2]) - (d_n[2] + d_n[4]) + 8) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i] + ((9*(d_n[i-1] + d_n[i+1]) - (d_n[i-3] + d_n[i+3]) + 8) >> 4);
    }
    d_n[n-3] = s_n[n-3] + ((9*(d_n[n-4] + d_n[n-2]) - (d_n[n-6] + d_n[n-2]) + 8) >> 4);
    d_n[n-1] = s_n[n-1] + ((9*d_n[n-2] - d_n[n-4] + 4) >> 3);
  }
}

void
schro_lift_split_desl93_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr >>= 1;
  if (n==2) {
    d_n[1] = s_n[1] - s_n[0*sstr];
    d_n[0] = s_n[0] + ((d_n[1] + 1) >> 1);
  } else if (n==4) {
    /* predict */
    d_n[1] = s_n[1*sstr] - ((9*(s_n[0*sstr] + s_n[2*sstr]) - (s_n[2*sstr] + s_n[2*sstr]) + 8) >> 4);
    d_n[3] = s_n[3*sstr] - ((9*s_n[2*sstr] - s_n[0*sstr] + 4) >> 3);

    /* update */
    d_n[0] = s_n[0*sstr] + ((d_n[1] + 1) >> 1);
    d_n[2] = s_n[2*sstr] + ((d_n[1] + d_n[3] + 2) >> 2);
  } else {
    /* predict */
    d_n[1] = s_n[1*sstr] - ((9*(s_n[0*sstr] + s_n[2*sstr]) - (s_n[2*sstr] + s_n[4*sstr]) + 8) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i*sstr] - ((9*(s_n[(i-1)*sstr] + s_n[(i+1)*sstr]) - (s_n[(i-3)*sstr] + s_n[(i+3)*sstr]) + 8) >> 4);
    }
    d_n[n-3] = s_n[(n-3)*sstr] - ((9*(s_n[(n-4)*sstr] + s_n[(n-2)*sstr]) - (s_n[(n-6)*sstr] + s_n[(n-2)*sstr]) + 8) >> 4);
    d_n[n-1] = s_n[(n-1)*sstr] - ((9*s_n[(n-2)*sstr] - s_n[(n-4)*sstr] + 4) >> 3);

    /* update */
    d_n[0] = s_n[0*sstr] + ((d_n[1] + 1) >> 1);
    for(i=2;i<n;i+=2){
      d_n[i] = s_n[i*sstr] + ((d_n[i-1] + d_n[i+1] + 2) >> 2);
    }
  }

}

void
schro_lift_synth_desl93_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr >>= 1;
  if (n==2) {
    d_n[0*dstr] = s_n[0] - ((s_n[1] + 1) >> 1);
    d_n[1*dstr] = s_n[1] + d_n[0*dstr];
  } else if (n==4) {
    /* predict */
    d_n[0*dstr] = s_n[0] - ((s_n[1] + 1) >> 1);
    d_n[2*dstr] = s_n[2] - ((s_n[1] + s_n[3] + 2) >> 2);

    /* update */
    d_n[1*dstr] = s_n[1] + ((9*(d_n[0*dstr] + d_n[2*dstr]) - (d_n[2*dstr] + d_n[2*dstr]) + 8) >> 4);
    d_n[3*dstr] = s_n[3] + ((9*d_n[2*dstr] - d_n[0*dstr] + 4) >> 3);
  } else {
    /* predict */
    d_n[0*dstr] = s_n[0] - ((s_n[1] + 1) >> 1);
    for(i=2;i<n;i+=2){
      d_n[i*dstr] = s_n[i] - ((s_n[i-1] + s_n[i+1] + 2) >> 2);
    }

    /* update */
    d_n[1*dstr] = s_n[1] + ((9*(d_n[0*dstr] + d_n[2*dstr]) - (d_n[2*dstr] + d_n[4*dstr]) + 8) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i*dstr] = s_n[i] + ((9*(d_n[(i-1)*dstr] + d_n[(i+1)*dstr]) - (d_n[(i-3)*dstr] + d_n[(i+3)*dstr]) + 8) >> 4);
    }
    d_n[(n-3)*dstr] = s_n[n-3] + ((9*(d_n[(n-4)*dstr] + d_n[(n-2)*dstr]) - (d_n[(n-6)*dstr] + d_n[(n-2)*dstr]) + 8) >> 4);
    d_n[(n-1)*dstr] = s_n[n-1] + ((9*d_n[(n-2)*dstr] - d_n[(n-4)*dstr] + 4) >> 3);
  }
}

void
schro_lift_split_53 (int16_t *d_n, int16_t *s_n, int n)
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
schro_lift_synth_53 (int16_t *d_n, int16_t *s_n, int n)
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
schro_lift_split_53_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
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
schro_lift_synth_53_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
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


void
schro_lift_split_135 (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n==2) {
    d_n[1] = s_n[1] - (s_n[0]);
    d_n[0] = s_n[0] + (d_n[1]>>1);
  } else if (n==4) {
    /* predict */
    d_n[1] = s_n[1] - ((9*(s_n[0] + s_n[2]) - (s_n[2] + s_n[2])) >> 4);
    d_n[3] = s_n[3] - ((9*s_n[2] - s_n[0]) >> 3);

    /* update */
    d_n[0] = s_n[0] + ((9*d_n[1] - d_n[3]) >> 4);
    d_n[2] = s_n[2] + ((9*(d_n[1] + d_n[3]) - (d_n[1] + d_n[1])) >> 5);
  } else {
    /* predict */
    d_n[1] = s_n[1] - ((9*(s_n[0] + s_n[2]) - (s_n[2] + s_n[4])) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i] - ((9*(s_n[i-1] + s_n[i+1]) - (s_n[i-3] + s_n[i+3])) >> 4);
    }
    d_n[n-3] = s_n[n-3] - ((9*(s_n[n-4] + s_n[n-2]) - (s_n[n-6] + s_n[n-2])) >> 4);
    d_n[n-1] = s_n[n-1] - ((9*s_n[n-2] - s_n[n-4]) >> 3);

    /* update */
    d_n[0] = s_n[0] + ((9*d_n[1] - d_n[3]) >> 4);
    d_n[2] = s_n[2] + ((9*(d_n[1] + d_n[3]) - (d_n[1] + d_n[5])) >> 5);
    for(i=4;i<n-2;i+=2){
      d_n[i] = s_n[i] + ((9*(d_n[i-1] + d_n[i+1]) - (d_n[i-3] + d_n[i+3])) >> 5);
    }
    d_n[n-2] = s_n[n-2] + ((9*(d_n[n-3] + d_n[n-1]) - (d_n[n-5] + d_n[n-1])) >> 5);
  }

}

void
schro_lift_synth_135 (int16_t *d_n, int16_t *s_n, int n)
{
  int i;

  if (n==2) {
    d_n[0] = s_n[0] - (s_n[1]>>1);
    d_n[1] = s_n[1] + (d_n[0]);
  } else if (n==4) {
    /* predict */
    d_n[0] = s_n[0] - ((9*s_n[1] - s_n[3]) >> 4);
    d_n[2] = s_n[2] - ((9*(s_n[1] + s_n[3]) - (s_n[1] + s_n[1])) >> 5);

    /* update */
    d_n[1] = s_n[1] + ((9*(d_n[0] + d_n[2]) - (d_n[2] + d_n[2])) >> 4);
    d_n[3] = s_n[3] + ((9*d_n[2] - d_n[0]) >> 3);
  } else {
    /* predict */
    d_n[0] = s_n[0] - ((9*s_n[1] - s_n[3]) >> 4);
    d_n[2] = s_n[2] - ((9*(s_n[1] + s_n[3]) - (s_n[1] + s_n[5])) >> 5);
    for(i=4;i<n-2;i+=2){
      d_n[i] = s_n[i] - ((9*(s_n[i-1] + s_n[i+1]) - (s_n[i-3] + s_n[i+3])) >> 5);
    }
    d_n[n-2] = s_n[n-2] - ((9*(s_n[n-3] + s_n[n-1]) - (s_n[n-5] + s_n[n-1])) >> 5);

    /* update */
    d_n[1] = s_n[1] + ((9*(d_n[0] + d_n[2]) - (d_n[2] + d_n[4])) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i] + ((9*(d_n[i-1] + d_n[i+1]) - (d_n[i-3] + d_n[i+3])) >> 4);
    }
    d_n[n-3] = s_n[n-3] + ((9*(d_n[n-4] + d_n[n-2]) - (d_n[n-6] + d_n[n-2])) >> 4);
    d_n[n-1] = s_n[n-1] + ((9*d_n[n-2] - d_n[n-4]) >> 3);
  }
}
void
schro_lift_split_135_str (int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  int i;

  sstr>>=1;
  if (n==2) {
    d_n[1] = s_n[1*sstr] - (s_n[0*sstr]);
    d_n[0] = s_n[0*sstr] + (d_n[1]>>1);
  } else if (n==4) {
    /* predict */
    d_n[1] = s_n[1*sstr] - ((9*(s_n[0*sstr] + s_n[2*sstr]) - (s_n[2*sstr] + s_n[2*sstr])) >> 4);
    d_n[3] = s_n[3*sstr] - ((9*s_n[2*sstr] - s_n[0*sstr]) >> 3);

    /* update */
    d_n[0] = s_n[0*sstr] + ((9*d_n[1] - d_n[3]) >> 4);
    d_n[2] = s_n[2*sstr] + ((9*(d_n[1] + d_n[3]) - (d_n[1] + d_n[1])) >> 5);
  } else {
    /* predict */
    d_n[1] = s_n[1*sstr] - ((9*(s_n[0*sstr] + s_n[2*sstr]) - (s_n[2*sstr] + s_n[4*sstr])) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i] = s_n[i*sstr] - ((9*(s_n[(i-1)*sstr] + s_n[(i+1)*sstr]) - (s_n[(i-3)*sstr] + s_n[(i+3)*sstr])) >> 4);
    }
    d_n[n-3] = s_n[(n-3)*sstr] - ((9*(s_n[(n-4)*sstr] + s_n[(n-2)*sstr]) - (s_n[(n-6)*sstr] + s_n[(n-2)*sstr])) >> 4);
    d_n[n-1] = s_n[(n-1)*sstr] - ((9*s_n[(n-2)*sstr] - s_n[(n-4)*sstr]) >> 3);

    /* update */
    d_n[0] = s_n[0*sstr] + ((9*d_n[1] - d_n[3]) >> 4);
    d_n[2] = s_n[2*sstr] + ((9*(d_n[1] + d_n[3]) - (d_n[1] + d_n[5])) >> 5);
    for(i=4;i<n-2;i+=2){
      d_n[i] = s_n[i*sstr] + ((9*(d_n[i-1] + d_n[i+1]) - (d_n[i-3] + d_n[i+3])) >> 5);
    }
    d_n[n-2] = s_n[(n-2)*sstr] + ((9*(d_n[n-3] + d_n[n-1]) - (d_n[n-5] + d_n[n-1])) >> 5);
  }

}

void
schro_lift_synth_135_str (int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  int i;

  dstr>>=1;
  if (n==2) {
    d_n[0*dstr] = s_n[0] - (s_n[1]>>1);
    d_n[1*dstr] = s_n[1] + (d_n[0*dstr]);
  } else if (n==4) {
    /* predict */
    d_n[0*dstr] = s_n[0] - ((9*s_n[1] - s_n[3]) >> 4);
    d_n[2*dstr] = s_n[2] - ((9*(s_n[1] + s_n[3]) - (s_n[1] + s_n[1])) >> 5);

    /* update */
    d_n[1*dstr] = s_n[1] + ((9*(d_n[0*dstr] + d_n[2*dstr]) - (d_n[2*dstr] + d_n[2*dstr])) >> 4);
    d_n[3*dstr] = s_n[3] + ((9*d_n[2*dstr] - d_n[0*dstr]) >> 3);
  } else {
    /* predict */
    d_n[0*dstr] = s_n[0] - ((9*s_n[1] - s_n[3]) >> 4);
    d_n[2*dstr] = s_n[2] - ((9*(s_n[1] + s_n[3]) - (s_n[1] + s_n[5])) >> 5);
    for(i=4;i<n-2;i+=2){
      d_n[i*dstr] = s_n[i] - ((9*(s_n[i-1] + s_n[i+1]) - (s_n[i-3] + s_n[i+3])) >> 5);
    }
    d_n[(n-2)*dstr] = s_n[n-2] - ((9*(s_n[n-3] + s_n[n-1]) - (s_n[n-5] + s_n[n-1])) >> 5);

    /* update */
    d_n[1*dstr] = s_n[1] + ((9*(d_n[0*dstr] + d_n[2*dstr]) - (d_n[2*dstr] + d_n[4*dstr])) >> 4);
    for(i=3;i<n-4;i+=2){
      d_n[i*dstr] = s_n[i] + ((9*(d_n[(i-1)*dstr] + d_n[(i+1)*dstr]) - (d_n[(i-3)*dstr] + d_n[(i+3)*dstr])) >> 4);
    }
    d_n[(n-3)*dstr] = s_n[n-3] + ((9*(d_n[(n-4)*dstr] + d_n[(n-2)*dstr]) - (d_n[(n-6)*dstr] + d_n[(n-2)*dstr])) >> 4);
    d_n[(n-1)*dstr] = s_n[n-1] + ((9*d_n[(n-2)*dstr] - d_n[(n-4)*dstr]) >> 3);
  }
}



void
schro_lift_split (int type, int16_t *d_n, int16_t *s_n, int n)
{
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      oil_split_daub97 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_DESL_9_3:
      oil_split_approx97 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_5_3:
      oil_split_53 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_13_5:
      oil_split_135 (d_n, s_n, n/2);
      break;
    default:
      SCHRO_ERROR("invalid type");
      break;
  }
}

void
schro_lift_split_str (int type, int16_t *d_n, int16_t *s_n, int sstr, int n)
{
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      schro_lift_split_daub97_str (d_n, s_n, sstr, n);
      break;
    case SCHRO_WAVELET_DESL_9_3:
      schro_lift_split_desl93_str (d_n, s_n, sstr, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_lift_split_53_str (d_n, s_n, sstr, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_lift_split_135_str (d_n, s_n, sstr, n);
      break;
    default:
      SCHRO_ERROR("invalid type");
      break;
  }
}

void
schro_lift_synth (int type, int16_t *d_n, int16_t *s_n, int n)
{
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      oil_synth_daub97 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_DESL_9_3:
      oil_synth_approx97 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_5_3:
      oil_synth_53 (d_n, s_n, n/2);
      break;
    case SCHRO_WAVELET_13_5:
      oil_synth_135 (d_n, s_n, n/2);
      break;
    default:
      SCHRO_ERROR("invalid type");
      break;
  }
}

void
schro_lift_synth_str (int type, int16_t *d_n, int dstr, int16_t *s_n, int n)
{
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      schro_lift_synth_daub97_str (d_n, dstr, s_n, n);
      break;
    case SCHRO_WAVELET_DESL_9_3:
      schro_lift_synth_desl93_str (d_n, dstr, s_n, n);
      break;
    case SCHRO_WAVELET_5_3:
      schro_lift_synth_53_str (d_n, dstr, s_n, n);
      break;
    case SCHRO_WAVELET_13_5:
      schro_lift_synth_135_str (d_n, dstr, s_n, n);
      break;
    default:
      SCHRO_ERROR("invalid type");
      break;
  }
}

void
schro_wavelet_transform_2d (int type, int16_t *i_n, int stride, int width,
    int height, int16_t *tmp)
{
  int i;
  int a;
  int n;
  int16_t c6497 = 6497;
  int16_t c217 = 217;
  int16_t c3616 = 3616;
  int16_t c1817 = 1817;

  stride >>= 1;
  n = width/2;
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      for(i=0;i<height-2;i+=2) {
        oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (i+2), &c6497, width);
      }
      oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
          i_n + stride * (i+0), i_n + stride * (i+0), &c6497, width);

      i = 0;
      oil_lift_sub_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
          i_n + stride * (i+1), i_n + stride * (i+1), &c217, width);
      for(i=2;i<height;i+=2) {
        oil_lift_sub_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (i-1), i_n + stride * (i+1), &c217, width);
      }

      for(i=0;i<height-2;i+=2) {
        oil_lift_add_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (i+2), &c3616, width);
      }
      oil_lift_add_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
          i_n + stride * (i+0), i_n + stride * (i+0), &c3616, width);

      i = 0;
      oil_lift_add_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
          i_n + stride * (i+1), i_n + stride * (i+1), &c1817, width);
      for(i=2;i<height;i+=2) {
        oil_lift_add_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (i-1), i_n + stride * (i+1), &c1817, width);
      }

      for(i=0;i<height;i++) {
        oil_split_daub97 (tmp, i_n + stride * i, n);
        oil_deinterleave (i_n + stride * i, tmp, n);
      }
      break;
    case SCHRO_WAVELET_5_3:
      for(i=0;i<height;i+=2){
        a = i+2;
        if (a >= height) a = 2*height - 2 - a;
        oil_lift_sub_shift1 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (a), width);
        a = i-1;
        if (a < 0) a = -a;
        oil_lift_add_shift2 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (a), i_n + stride * (i+1), width);
        if (i>=2) {
          oil_split_53 (tmp, i_n + stride * (i-2), n);
          oil_deinterleave (i_n + stride * (i-2), tmp, n);
          oil_split_53 (tmp, i_n + stride * (i-1), n);
          oil_deinterleave (i_n + stride * (i-1), tmp, n);
        }
      }
      oil_split_53 (tmp, i_n + stride * (height-2), n);
      oil_deinterleave (i_n + stride * (height-2), tmp, n);
      oil_split_53 (tmp, i_n + stride * (height-1), n);
      oil_deinterleave (i_n + stride * (height-1), tmp, n);

      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      for(i=0;i<height;i+=2){
        oil_split_haar (i_n + stride * i, i_n + stride * (i+1), width);
        oil_split_haar (tmp, i_n + stride * i, n);
        oil_deinterleave (i_n + stride * i, tmp, n);
        oil_split_haar (tmp, i_n + stride * (i+1), n);
        oil_deinterleave (i_n + stride * (i+1), tmp, n);
      }

      break;
#if 0
    case SCHRO_WAVELET_135:
#define REFLECT_0(value) ((value<0)?(-value):(value))
#define REFLECT_N(value,endpoint) ((value)>=(endpoint)?(2*(endpoint)-1-(value)):(value))
      for(i=0;i<height-2;i+=2) {
        oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (i+2), &c6497, n);
      }
      oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
          i_n + stride * (i+0), i_n + stride * (i+0), &c6497, n);

      for(i=0;i<height;i++) {
        oil_split_daub97 (tmp, i_n + stride * i, n);
        oil_deinterleave (i_n + stride * i, tmp, n);
      }
      break;
#endif
    default:
      SCHRO_ERROR("invalid type %d", type);
      break;
  }
}

void
schro_wavelet_inverse_transform_2d (int type, int16_t *i_n, int stride, int width,
    int height, int16_t *tmp)
{
  int i;
  int n;
  int16_t c6497 = 6497;
  int16_t c217 = 217;
  int16_t c3616 = 3616;
  int16_t c1817 = 1817;

  SCHRO_ASSERT((height&1)==0);
  SCHRO_ASSERT((width&1)==0);

  stride >>= 1;
  n = width/2;
  switch (type) {
    case SCHRO_WAVELET_DAUB_9_7:
      for(i=0;i<height;i++) {
        oil_interleave (tmp, i_n + stride * i, n);
        oil_synth_daub97 (i_n + stride * i, tmp, n);
      }

      i = 0;
      oil_lift_sub_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
          i_n + stride * (i+1), i_n + stride * (i+1), &c1817, width);
      for(i=2;i<height;i+=2) {
        oil_lift_sub_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (i-1), i_n + stride * (i+1), &c1817, width);
      }

      for(i=0;i<height-2;i+=2) {
        oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (i+2), &c3616, width);
      }
      oil_lift_sub_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
          i_n + stride * (i+0), i_n + stride * (i+0), &c3616, width);

      i = 0;
      oil_lift_add_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
          i_n + stride * (i+1), i_n + stride * (i+1), &c217, width);
      for(i=2;i<height;i+=2) {
        oil_lift_add_mult_shift12 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (i-1), i_n + stride * (i+1), &c217, width);
      }
      
      for(i=0;i<height-2;i+=2) {
        oil_lift_add_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
            i_n + stride * (i+0), i_n + stride * (i+2), &c6497, width);
      }
      oil_lift_add_mult_shift12 (i_n + stride * (i+1), i_n + stride * (i+1),
          i_n + stride * (i+0), i_n + stride * (i+0), &c6497, width);

      break;
    case SCHRO_WAVELET_DESL_9_3:
      for(i=0;i<height;i++) {
        oil_interleave (tmp, i_n + stride * i, n);
        oil_synth_approx97 (i_n + stride * i, tmp, n);
      }
      for(i=0;i<width;i+=2){
        schro_interleave_str (tmp, i_n + i, stride/2, height/2);
        schro_lift_synth_desl93_str (i_n + i, stride/2, tmp, n);
      }
      break;
    case SCHRO_WAVELET_5_3:
      oil_interleave (tmp, i_n + stride * (0), n);
      oil_synth_53 (i_n + stride * (0), tmp, n);
      oil_interleave (tmp, i_n + stride * (1), n);
      oil_synth_53 (i_n + stride * (1), tmp, n);
      oil_lift_sub_shift2 (i_n + stride * (0), i_n + stride * (0),
          i_n + stride * (1), i_n + stride * (1), width);
      for(i=2;i<height;i+=2){
        oil_interleave (tmp, i_n + stride * (i + 0), n);
        oil_synth_53 (i_n + stride * (i + 0), tmp, n);
        oil_interleave (tmp, i_n + stride * (i + 1), n);
        oil_synth_53 (i_n + stride * (i + 1), tmp, n);
        oil_lift_sub_shift2 (i_n + stride * (i+0), i_n + stride * (i+0),
            i_n + stride * (i-1), i_n + stride * (i+1), width);
        oil_lift_add_shift1 (i_n + stride * (i-1), i_n + stride * (i-1),
            i_n + stride * (i-2), i_n + stride * (i), width);
      }
      i = height;
      oil_lift_add_shift1 (i_n + stride * (i-1), i_n + stride * (i-1),
          i_n + stride * (i-2), i_n + stride * (i-2), width);

      break;
    case SCHRO_WAVELET_HAAR_0:
    case SCHRO_WAVELET_HAAR_1:
    case SCHRO_WAVELET_HAAR_2:
      for(i=0;i<height;i+=2){
        oil_interleave (tmp, i_n + stride * (i + 0), n);
        oil_synth_haar (i_n + stride * (i + 0), tmp, n);
        oil_interleave (tmp, i_n + stride * (i + 1), n);
        oil_synth_haar (i_n + stride * (i + 1), tmp, n);
        oil_lift_haar_synth (i_n + stride * i, i_n + stride * (i+1), width);
      }

      break;
    default:
      SCHRO_ERROR("invalid type %d", type);
      break;
  }
}







void
schro_split_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 7, 4 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };

  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage1_weights, stage1_offset_shift, n);

  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);
}

void
schro_split_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 0, 1 };
  static const int16_t stage2_offset_shift[] = { 2, 2 };

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage2_offset_shift, n);
}

void
schro_split_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 7, 4 };
  static const int16_t stage2_offset_shift[] = { 16, 5 };

  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi-1, stage1_weights, stage1_offset_shift, n);

  lo[-1] = lo[0];
  lo[-2] = lo[1];
  lo[n] = lo[n-2];

  oil_mas4_add_s16 (hi, hi, lo - 2, stage2_weights, stage2_offset_shift, n);
}

void
schro_split_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

  for(i=0;i<n;i++) {
    lo[i] -= hi[i];
  }
  for(i=0;i<n;i++) {
    hi[i] += ((lo[i] + 1)>>1);
  }
}

void
schro_split_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
  static const int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };
  static const int16_t stage1_offset_shift[] = { 128, 8 };
  static const int16_t stage2_offset_shift[] = { 127, 8 };

  lo[-4] = lo[3];
  lo[-3] = lo[2];
  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];
  lo[n+2] = lo[n-4];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage1_weights, stage1_offset_shift, n);

  hi[-3] = hi[3];
  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  hi[n+2] = hi[n-3];
  hi[n+3] = hi[n-4];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage2_weights, stage2_offset_shift, n);
}

void
schro_split_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -6497, -6497 };
  static const int16_t stage2_weights[] = { -217, -217 };
  static const int16_t stage3_weights[] = { 3616, 3616 };
  static const int16_t stage4_weights[] = { 1817, 1817 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage1_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage2_weights, stage12_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage3_weights, stage34_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage4_weights, stage34_offset_shift, n);

}

void
schro_synth_ext_desl93 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 1, 2 };
  static const int16_t stage2_offset_shift[] = { 8, 4 };

  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];

  oil_mas4_add_s16 (lo, lo, hi - 1, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_53 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1, -1 };
  static const int16_t stage2_weights[] = { 1, 1 };
  static const int16_t stage1_offset_shift[] = { 1, 2 };
  static const int16_t stage2_offset_shift[] = { 1, 1 };

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_135 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { 1, -9, -9, 1 };
  static const int16_t stage2_weights[] = { -1, 9, 9, -1 };
  static const int16_t stage1_offset_shift[] = { 15, 5 };
  static const int16_t stage2_offset_shift[] = { 8, 4 };

  lo[-1] = lo[0];
  lo[-2] = lo[1];
  lo[n] = lo[n-2];
  oil_mas4_add_s16 (hi, hi, lo - 2, stage1_weights, stage1_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  oil_mas4_add_s16 (lo, lo, hi-1, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_haar (int16_t *hi, int16_t *lo, int n)
{
  int i;

  for(i=0;i<n;i++) {
    hi[i] -= ((lo[i] + 1)>>1);
  }
  for(i=0;i<n;i++) {
    lo[i] += hi[i];
  }
}

void
schro_synth_ext_fidelity (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -2, 10, -25, 81, 81, -25, 10, -2 };
  static const int16_t stage2_weights[] = { 8, -21, 46, -161, -161, 46, -21, 8 };
  static const int16_t stage1_offset_shift[] = { 128, 8 };
  static const int16_t stage2_offset_shift[] = { 127, 8 };

  hi[-3] = hi[3];
  hi[-2] = hi[2];
  hi[-1] = hi[1];
  hi[n] = hi[n-1];
  hi[n+1] = hi[n-2];
  hi[n+2] = hi[n-3];
  hi[n+3] = hi[n-4];

  oil_mas8_add_s16 (lo, lo, hi - 3, stage1_weights, stage1_offset_shift, n);

  lo[-4] = lo[3];
  lo[-3] = lo[2];
  lo[-2] = lo[1];
  lo[-1] = lo[0];
  lo[n] = lo[n-2];
  lo[n+1] = lo[n-3];
  lo[n+2] = lo[n-4];

  oil_mas8_add_s16 (hi, hi, lo - 4, stage2_weights, stage2_offset_shift, n);
}

void
schro_synth_ext_daub97 (int16_t *hi, int16_t *lo, int n)
{
  static const int16_t stage1_weights[] = { -1817, -1817 };
  static const int16_t stage2_weights[] = { -3616, -3616 };
  static const int16_t stage3_weights[] = { 217, 217 };
  static const int16_t stage4_weights[] = { 6497, 6497 };
  static const int16_t stage12_offset_shift[] = { 2047, 12 };
  static const int16_t stage34_offset_shift[] = { 2048, 12 };

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage1_weights, stage12_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage2_weights, stage12_offset_shift, n);

  lo[-1] = lo[0];
  lo[n] = lo[n-2];

  oil_mas2_add_s16 (hi, hi, lo - 1, stage3_weights, stage34_offset_shift, n);

  hi[-1] = hi[1];
  hi[n] = hi[n-1];

  oil_mas2_add_s16 (lo, lo, hi, stage4_weights, stage34_offset_shift, n);
}

