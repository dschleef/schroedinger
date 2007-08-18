
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrofft.h>
#include <math.h>
#include <string.h>


#define COMPLEX_MULT_R(a,b,c,d) ((a)*(c) - (b)*(d))
#define COMPLEX_MULT_I(a,b,c,d) ((a)*(d) + (b)*(c))


static void
complex_mult_f32 (float *d1, float *d2, float *s1, float *s2,
    float *s3, float *s4, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = COMPLEX_MULT_R(s1[i], s2[i], s3[i], s4[i]);
    d2[i] = COMPLEX_MULT_I(s1[i], s2[i], s3[i], s4[i]);
  }
}

static void
complex_normalize_f32 (float *i1, float *i2, int n)
{
  int i;
  float x;
  for(i=0;i<n;i++){
    x = sqrt(i1[i]*i1[i] + i2[i]*i2[i]);
    if (x > 0) x = 1/x;
    i1[i] *= x;
    i2[i] *= x;
  }
}

int
get_max_f32 (float *src, int n)
{
  int i;
  float max;
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


void
schro_encoder_phasecorr_prediction (SchroEncoderFrame *frame)
{
  SchroFrame *ref;
  SchroFrame *src;
  int shift;
  int n;
  uint8_t *line;
  float *image1;
  float *image2;
  int i,j;
  float *s, *c;
  float *zero;
  float *ft1r;
  float *ft1i;
  float *ft2r;
  float *ft2i;
  float *conv_r, *conv_i;
  float *resr, *resi;
  float *weight;
  float sum;
  float weight2;

  shift = 12;
  n = 1<<shift;

  /* tables */
  s = malloc(n*sizeof(float));
  c = malloc(n*sizeof(float));
  weight = malloc(n*sizeof(float));
  zero = malloc(n*sizeof(float));
  memset (zero, 0, n*sizeof(float));

  image1 = malloc(n*sizeof(float));
  image2 = malloc(n*sizeof(float));

  ft1r = malloc(n*sizeof(float));
  ft1i = malloc(n*sizeof(float));
  ft2r = malloc(n*sizeof(float));
  ft2i = malloc(n*sizeof(float));
  conv_r = malloc(n*sizeof(float));
  conv_i = malloc(n*sizeof(float));
  resr = malloc(n*sizeof(float));
  resi = malloc(n*sizeof(float));


  src = frame->downsampled_frames[0];
  ref = frame->ref_frame0->downsampled_frames[0];

  SCHRO_ASSERT(src);
  SCHRO_ASSERT(ref);

  sum = 0;
  for(j=0;j<64;j++){
    for(i=0;i<64;i++){
      float d2;
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

  schro_fft_generate_tables_f32 (c, s, shift);

  schro_fft_fwd_f32 (ft1r, ft1i, image1, zero, c, s, shift);
  schro_fft_fwd_f32 (ft2r, ft2i, image2, zero, c, s, shift);

  for(i=0;i<n;i++){
    ft2i[i] = -ft2i[i];
  }

  complex_mult_f32 (conv_r, conv_i, ft1r, ft1i, ft2r, ft2i, n);
  complex_normalize_f32 (conv_r, conv_i, n);

  schro_fft_rev_f32 (resr, resi, conv_r, conv_i, c, s, shift);

  {
    int x,y;

    i = get_max_f32 (resr, n);

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

  free(s);
  free(c);
  free(weight);
  free(zero);

  free(image1);
  free(image2);

  free(ft1r);
  free(ft1i);
  free(ft2r);
  free(ft2i);
  free(conv_r);
  free(conv_i);
  free(resr);
  free(resi);
}

