
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

static void
generate_weights (float *weight, int width, int height)
{
  int i;
  int j;
  double d2;
  double mid_x, mid_y;
  double scale_x, scale_y;
  double sum;
  double weight2;

  mid_x = 0.5*(width-1);
  mid_y = 0.5*(height-1);
  scale_x = 1.0/mid_x;
  scale_y = 1.0/mid_y;

  sum = 0;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      d2 = (i-mid_x)*(i-mid_x)*scale_x*scale_x +
        (j-mid_y)*(j-mid_y)*scale_y*scale_y;
      weight[j*width+i] = exp(-2*d2);
      sum += weight[j*width+i];
    }
  }
  weight2 = 1.0/sum;
  for(j=0;j<height;j++){
    for(i=0;i<width;i++){
      weight[j*width+i] *= weight2;
    }
  }
}

static void
get_image (float *image, SchroFrame *frame, int x, int y, int width,
    int height, float *weight)
{
  double sum;
  int i,j;
  uint8_t *line;
  double weight2;

  sum = 0;
  for(j=0;j<height;j++){
    line = OFFSET(frame->components[0].data, frame->components[0].stride*(j+y));
    for(i=0;i<width;i++){
      sum += line[i+x] * weight[j*width + i];
    }
  }
  weight2 = 1.0/sum;
  for(j=0;j<height;j++){
    line = OFFSET(frame->components[0].data, frame->components[0].stride*(j+y));
    for(i=0;i<width;i++){
      image[j*width+i] = line[i+x] * weight[j*width+i] * weight2;
    }
  }
}

static void
find_peak (float *ccorr, int hshift, int vshift, double *dx, double *dy)
{
  int idx,idy;
  int width = 1<<hshift;
  int height = 1<<vshift;
  int i;
  float peak;
  float a,b;

  i = get_max_f32 (ccorr, width*height);
  peak = ccorr[i];
  if (peak == 0) {
    *dx = 0;
    *dy = 0;
  }

  idx = i&(width-1);
  if (idx >= width/2) idx -= width;
  idy = i>>hshift;
  if (idy >= height/2) idy -= height;

#define get_ccorr_value(x,y) ccorr[((x)&(width-1)) + (((y)&(height-1))<<hshift)]
  a = get_ccorr_value (idx+1, idy);
  b = get_ccorr_value (idx-1, idy);
  if (a>b) {
    *dx = idx + 0.5*a/peak;
  } else {
    *dx = idx - 0.5*b/peak;
  }

  a = get_ccorr_value (idx, idy+1);
  b = get_ccorr_value (idx, idy-1);
  if (a>b) {
    *dy = idy + 0.5*a/peak;
  } else {
    *dy = idy - 0.5*b/peak;
  }
}

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

void
schro_encoder_phasecorr_prediction (SchroEncoderFrame *frame)
{
  SchroParams *params = &frame->params;
  SchroFrame *ref;
  SchroFrame *src;
  SchroMotionField *mf;
  int shift;
  int n;
  float *image1;
  float *image2;
  int i, j;
  float *s, *c;
  float *zero;
  float *ft1r;
  float *ft1i;
  float *ft2r;
  float *ft2i;
  float *conv_r, *conv_i;
  float *resr, *resi;
  float *weight;
  int width;
  int height;
  int hshift;
  int vshift;
  int x,y;

  hshift = 6;
  vshift = 5;
  width = 1<<hshift;
  height = 1<<vshift;
  shift = hshift+vshift;
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

#define SHIFT 2

  src = frame->downsampled_frames[SHIFT-1];
  SCHRO_ASSERT(src);

  generate_weights(weight, width, height);

  for(i=0;i<params->num_refs;i++){
    mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
    {
      int l,k;
      SchroMotionVector *mv;
      for(l=0;l<params->y_num_blocks;l++){
        for(k=0;k<params->x_num_blocks;k++){
          mv = motion_field_get (mf, k, l);
          mv->split = 2;
          mv->pred_mode = 1;
          mv->x1 = 0;
          mv->y1 = 0;
          mv->metric = 32000;
        }
      }
    }
    if (i == 0) {
      ref = frame->ref_frame0->downsampled_frames[SHIFT-1];
      frame->motion_fields[SCHRO_MOTION_FIELD_PHASECORR_REF0] = mf;
    } else {
      ref = frame->ref_frame1->downsampled_frames[SHIFT-1];
      frame->motion_fields[SCHRO_MOTION_FIELD_PHASECORR_REF1] = mf;
    }
    SCHRO_ASSERT(ref);

    for(y=0;y<=(src->height-height);y+=height/2){
      for(x=0;x<=(src->width-width);x+=width/2){
        double dx, dy;

        get_image (image1, src, x, y, width, height, weight);
        get_image (image2, ref, x, y, width, height, weight);

        schro_fft_generate_tables_f32 (c, s, shift);

        schro_fft_fwd_f32 (ft1r, ft1i, image1, zero, c, s, shift);
        schro_fft_fwd_f32 (ft2r, ft2i, image2, zero, c, s, shift);

        for(j=0;j<n;j++){
          ft2i[j] = -ft2i[j];
        }

        complex_mult_f32 (conv_r, conv_i, ft1r, ft1i, ft2r, ft2i, n);
        complex_normalize_f32 (conv_r, conv_i, n);

        schro_fft_rev_f32 (resr, resi, conv_r, conv_i, c, s, shift);

        find_peak (resr, hshift, vshift, &dx, &dy);

        SCHRO_ERROR("%d x,y %d,%d dx,dy %g,%g", frame->frame_number,
            x, y, dx, dy);

#if 1
        {
          int k,l;
          SchroMotionVector *mv;

          for(l=0;l<height>>(4-SHIFT);l++){
            for(k=0;k<width>>(4-SHIFT);k++){
              mv = motion_field_get (mf,
                  (x + width/4)*(1<<SHIFT)/8 + k,
                  (y + height/4)*(1<<SHIFT)/8 + l);
              mv->pred_mode = 1<<i;
              mv->x1 = rint(-dx * (1<<SHIFT) * 8);
              mv->x1 = (mv->x1 + 4)&(~0x07);
              mv->y1 = rint(-dy * (1<<SHIFT) * 8);
              mv->y1 = (mv->y1 + 4)&(~0x07);
              mv->metric = 0;
            }
          }
        }
#endif
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
    }
  }

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

