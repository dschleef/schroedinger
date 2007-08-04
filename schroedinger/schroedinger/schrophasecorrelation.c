
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schro.h>
#include <math.h>


#define oil_rand_f64() (((rand()/(RAND_MAX+1.0))+rand())/(RAND_MAX+1.0))


void
discrete_fourier_transform (double *d1, double *d2, double *s1, double *s2,
    double *s3, int n)
{
  int mask = n-1;
  double x;
  double y;
  int i;
  int j;

  for(i=0;i<n;i++){
    x = 0;
    y = 0;
    for(j=0;j<n;j++){
      x += s1[j] * s2[(i*j)&mask];
      y += s1[j] * s3[(i*j)&mask];
    }
    d1[i] = x;
    d2[i] = y;
  }
}

void
complex_discrete_fourier_transform (double *d1, double *d2,
    double *s1, double *s2, double *s3, double *s4, int n)
{
  int mask = n-1;
  double x;
  double y;
  int i;
  int j;

  for(i=0;i<n;i++){
    x = 0;
    y = 0;
    for(j=0;j<n;j++){
      x += s1[j] * s3[(i*j)&mask] - s2[j] * s4[(j*i)&mask];
      y += s1[j] * s4[(i*j)&mask] + s2[j] * s3[(j*i)&mask];
    }
    d1[i] = x;
    d2[i] = y;
  }
}


void sincos_array (double *d1, double *d2, double inc, int n)
{
  int i;

  for(i=0;i<n;i++){
    d1[i] = cos(inc*i);
    d2[i] = sin(inc*i);
  }
}

void complex_mult (double *d1, double *d2, double *s1, double *s2,
    double *s3, double *s4, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = s1[i] * s3[i] - s2[i] * s4[i];
    d2[i] = s1[i] * s4[i] + s2[i] * s3[i];
  }
}

void complex_normalize (double *i1, double *i2, int n)
{
  int i;
  double x;
  for(i=0;i<n;i++){
    x = sqrt(i1[i]*i1[i] + i2[i]*i2[i]);
    if (x > 0) x = 1/x;
    i1[i] *= x;
    i2[i] *= x;
  }
}

int
get_max_f64 (double *src, int n)
{
  int i;
  double max;
  int max_i;

  max = src[0];
  max_i = 0;

  for(i=1;i<n;i++){
    if (src[i] > max) {
      max_i = i;
      max = src[i];
    }
  }

  return max_i;
}

#define N 4096

void
schro_encoder_phasecorr_prediction (SchroEncoderFrame *frame)
{
  SchroFrame *ref;
  SchroFrame *src;
  uint8_t *line;
  double image1[N];
  double image2[N];
  int i,j;
  double s[N], c[N];
  double ft1r[N];
  double ft1i[N];
  double ft2r[N];
  double ft2i[N];
  double conv_r[N], conv_i[N];
  double resr[N], resi[N];
  double weight[N];
  double sum;
  double weight2;

/* disable this for checkin */
return;
  src = frame->downsampled_frames[0];
  ref = frame->ref_frame0->downsampled_frames[0];

  SCHRO_ASSERT(src);
  SCHRO_ASSERT(ref);

  sum = 0;
  for(j=0;j<64;j++){
    for(i=0;i<64;i++){
      double d2;
      d2 = ((i-32)*(i-32) + (j-32)*(j-32))/(32.0*32.0);
      //weight = 1 - d2;
      //if (weight < 0) weight = 0;
      weight[j*64+i] = exp(-5*d2);
      sum += weight[j*64+i];
    }
  }
  weight2 = 1.0/sum;
  for(j=0;j<64;j++){
    for(i=0;i<64;i++){
      weight[j*64+i] *= weight2;
    }
  }

  sum = 0;
  for(j=0;j<64;j++){
    line = OFFSET(src->components[0].data, src->components[0].stride*(j+28));
    for(i=0;i<64;i++){
      sum += line[i+58] * weight[j*64 + i];
    }
  }
  weight2 = 1.0/sum;
  for(j=0;j<64;j++){
    line = OFFSET(src->components[0].data, src->components[0].stride*(j+28));
    for(i=0;i<64;i++){
      image1[j*64+i] = line[i+58] * weight[j*64+i] * weight2;
    }
  }

  sum = 0;
  for(j=0;j<64;j++){
    line = OFFSET(ref->components[0].data, ref->components[0].stride*(j+28));
    for(i=0;i<64;i++){
      sum += line[i+58] * weight[j*64 + i];
    }
  }
  weight2 = 1.0/sum;
  for(j=0;j<64;j++){
    line = OFFSET(ref->components[0].data, ref->components[0].stride*(j+28));
    for(i=0;i<64;i++){
      image2[j*64+i] = line[i+58] * weight[j*64+i] * weight2;
    }
  }

  sincos_array (c, s, 2*M_PI/N, N);

  discrete_fourier_transform (ft1r, ft1i, image1, c, s, N);
  discrete_fourier_transform (ft2r, ft2i, image2, c, s, N);

  for(i=0;i<N;i++){
    ft2i[i] = -ft2i[i];
  }
  complex_mult (conv_r, conv_i, ft1r, ft1i, ft2r, ft2i, N);

  complex_normalize (conv_r, conv_i, N);

  complex_discrete_fourier_transform (resi, resr, conv_i, conv_r, c, s, N);

  {
    int x,y;

    i = get_max_f64 (resr, N);

    x = i&0x3f;
    if (x >= 32) x -= 64;
    y = i>>6;
    if (y >= 32) y -= 64;

    SCHRO_ERROR("%d index %d x,y %d,%d", frame->frame_number, i, x, y);
  }
#if 0
  if (frame->frame_number == 87) {
  for(i=0;i<N;i++){
  SCHRO_ERROR("ACK: %d %d %d %g %g %g %g %g %g", i, (i>>6), (i&0x3f),
      image1[i], image2[i], ft1r[i], ft1i[i], resr[i], resi[i]);
  }
  exit(0);
  }
#endif

}

