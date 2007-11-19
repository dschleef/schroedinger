
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

static void
negate_f32 (float *i1, int n)
{
  int j;
  for(j=0;j<n;j++){
    i1[j] = -i1[j];
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

  get_ccorr_value (idx-1, idy-1) = 0;
  get_ccorr_value (idx, idy-1) = 0;
  get_ccorr_value (idx+1, idy-1) = 0;
  get_ccorr_value (idx-1, idy) = 0;
  get_ccorr_value (idx, idy) = 0;
  get_ccorr_value (idx+1, idy) = 0;
  get_ccorr_value (idx-1, idy+1) = 0;
  get_ccorr_value (idx, idy+1) = 0;
  get_ccorr_value (idx+1, idy+1) = 0;
}

#define motion_field_get(mf,x,y) \
  ((mf)->motion_vectors + (y)*(mf)->x_num_blocks + (x))

static SchroFrame *
get_downsampled(SchroEncoderFrame *frame, int i)
{
  SCHRO_ASSERT(frame->have_downsampling);

  if (i==0) {
    return frame->filtered_frame;
  }
  return frame->downsampled_frames[i-1];
}

typedef struct _SchroMVComp SchroMVComp;
struct _SchroMVComp {
  int metric;
  SchroFrame *frame;
  SchroFrame *ref;
  int dx, dy;
};

void
schro_mvcomp_init (SchroMVComp *mvcomp, SchroFrame *frame, SchroFrame *ref)
{
  memset (mvcomp, 0, sizeof(*mvcomp));

  mvcomp->metric = 32000;
  mvcomp->frame = frame;
  mvcomp->ref = ref;
}

void
schro_mvcomp_add (SchroMVComp *mvcomp, int i, int j, int dx, int dy)
{
  int metric;

  metric = schro_frame_get_metric (mvcomp->frame,
      i*8, j*8, mvcomp->ref, i*8 + dx, j*8 + dy);
  if (metric < mvcomp->metric) {
    mvcomp->metric = metric;
    mvcomp->dx = dx;
    mvcomp->dy = dy;
  }
}

typedef struct _SchroPhaseCorr SchroPhaseCorr;
struct _SchroPhaseCorr {

  int hshift;
  int vshift;
  int width;
  int height;
  int shift;
  int n;

  /* static tables */
  float *s, *c;
  float *zero;
  float *weight;

  float *image1;
  float *image2;
  float *ft1r;
  float *ft1i;
  float *ft2r;
  float *ft2i;
  float *conv_r, *conv_i;
  float *resr, *resi;
};

SchroPhaseCorr *
schro_phasecorr_new (void)
{
  SchroPhaseCorr *pc;

  pc = malloc(sizeof(SchroPhaseCorr));
  memset (pc, 0, sizeof(SchroPhaseCorr));

  pc->hshift = 5;
  pc->vshift = 4;
  pc->width = 1<<pc->hshift;
  pc->height = 1<<pc->vshift;
  pc->shift = pc->hshift+pc->vshift;
  pc->n = 1<<pc->shift;

  pc->s = malloc(pc->n*sizeof(float));
  pc->c = malloc(pc->n*sizeof(float));
  pc->weight = malloc(pc->n*sizeof(float));
  pc->zero = malloc(pc->n*sizeof(float));
  memset (pc->zero, 0, pc->n*sizeof(float));

  pc->image1 = malloc(pc->n*sizeof(float));
  pc->image2 = malloc(pc->n*sizeof(float));

  pc->ft1r = malloc(pc->n*sizeof(float));
  pc->ft1i = malloc(pc->n*sizeof(float));
  pc->ft2r = malloc(pc->n*sizeof(float));
  pc->ft2i = malloc(pc->n*sizeof(float));
  pc->conv_r = malloc(pc->n*sizeof(float));
  pc->conv_i = malloc(pc->n*sizeof(float));
  pc->resr = malloc(pc->n*sizeof(float));
  pc->resi = malloc(pc->n*sizeof(float));

  generate_weights(pc->weight, pc->width, pc->height);
  schro_fft_generate_tables_f32 (pc->c, pc->s, pc->shift);

  return pc;
}

void
schro_phasecorr_free (SchroPhaseCorr *pc)
{
  free(pc->s);
  free(pc->c);
  free(pc->weight);
  free(pc->zero);

  free(pc->image1);
  free(pc->image2);

  free(pc->ft1r);
  free(pc->ft1i);
  free(pc->ft2r);
  free(pc->ft2i);
  free(pc->conv_r);
  free(pc->conv_i);
  free(pc->resr);
  free(pc->resi);

  free(pc);
}


void
schro_encoder_phasecorr_prediction (SchroEncoderFrame *frame)
{
  SchroPhaseCorr *pc;
  SchroParams *params = &frame->params;
  SchroFrame *ref;
  SchroFrame *src;
  SchroMotionField *mf;
  int i;
  int x,y;
  int num_x;
  int num_y;
  int *vecs_dx;
  int *vecs_dy;
  int *vecs2_dx;
  int *vecs2_dy;
  int ix, iy;
  int k,l;

  pc = schro_phasecorr_new ();

#define SHIFT 2

  for(i=0;i<params->num_refs;i++){
    mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);

    src = get_downsampled(frame, SHIFT);
    SCHRO_ASSERT(src);

    if (i == 0) {
      ref = get_downsampled(frame->ref_frame0, SHIFT);
      frame->motion_fields[SCHRO_MOTION_FIELD_PHASECORR_REF0] = mf;
    } else {
      ref = get_downsampled(frame->ref_frame1, SHIFT);
      frame->motion_fields[SCHRO_MOTION_FIELD_PHASECORR_REF1] = mf;
    }
    SCHRO_ASSERT(ref);

    num_x = (src->width - pc->width)/(pc->width/2);
    num_y = (src->height - pc->height)/(pc->height/2);
    vecs_dx = malloc(sizeof(int)*num_x*num_y);
    vecs_dy = malloc(sizeof(int)*num_x*num_y);
    vecs2_dx = malloc(sizeof(int)*num_x*num_y);
    vecs2_dy = malloc(sizeof(int)*num_x*num_y);

    for(iy=0;iy<num_y;iy++){
      for(ix=0;ix<num_x;ix++){
        double dx, dy;

        x = ((src->width - pc->width) * ix) / (num_x - 1);
        y = ((src->height - pc->height) * iy) / (num_y - 1);

        get_image (pc->image1, src, x, y, pc->width, pc->height, pc->weight);
        get_image (pc->image2, ref, x, y, pc->width, pc->height, pc->weight);

        schro_fft_fwd_f32 (pc->ft1r, pc->ft1i, pc->image1, pc->zero, pc->c, pc->s, pc->shift);
        schro_fft_fwd_f32 (pc->ft2r, pc->ft2i, pc->image2, pc->zero, pc->c, pc->s, pc->shift);

        negate_f32 (pc->ft2i, pc->n);

        complex_mult_f32 (pc->conv_r, pc->conv_i, pc->ft1r, pc->ft1i, pc->ft2r, pc->ft2i, pc->n);
        complex_normalize_f32 (pc->conv_r, pc->conv_i, pc->n);

        schro_fft_rev_f32 (pc->resr, pc->resi, pc->conv_r, pc->conv_i, pc->c, pc->s, pc->shift);

        find_peak (pc->resr, pc->hshift, pc->vshift, &dx, &dy);

        schro_dump(SCHRO_DUMP_PHASE_CORR,"%d %d %d %g %g\n",
            frame->frame_number, x, y, dx, dy);

        vecs_dx[iy*num_x + ix] = rint(-dx * (1<<SHIFT));
        vecs_dy[iy*num_x + ix] = rint(-dy * (1<<SHIFT));

        find_peak (pc->resr, pc->hshift, pc->vshift, &dx, &dy);

        vecs2_dx[iy*num_x + ix] = rint(-dx * (1<<SHIFT));
        vecs2_dy[iy*num_x + ix] = rint(-dy * (1<<SHIFT));
      }
    }

    src = get_downsampled(frame, 0);
    if (i == 0) {
      ref = get_downsampled(frame->ref_frame0, 0);
    } else {
      ref = get_downsampled(frame->ref_frame1, 0);
    }
    for(l=0;l<params->y_num_blocks;l++){
      for(k=0;k<params->x_num_blocks;k++){
        SchroMotionVector *mv;
        int ymin, ymax;
        int xmin, xmax;
        SchroMVComp mvcomp;

        /* FIXME real block params */
        xmin = k*8 - 2;
        xmax = k*8 + 10;
        ymin = l*8 - 2;
        ymax = l*8 + 10;

        schro_mvcomp_init (&mvcomp, src, ref);

        for(iy=0;iy<num_y;iy++){
          for(ix=0;ix<num_x;ix++){
            x = ((src->width - (pc->width<<SHIFT)) * ix) / (num_x - 1);
            y = ((src->height - (pc->height<<SHIFT)) * iy) / (num_y - 1);

            if (xmax < x || ymax < y ||
                xmin >= x + (pc->width<<SHIFT) ||
                ymin >= y + (pc->height<<SHIFT)) {
              continue;
            }

            SCHRO_DEBUG("%d %d trying %d %d (%d %d)", k, l, ix, iy,
                vecs_dx[iy*num_x + ix], vecs_dy[iy*num_x + ix]);
            schro_mvcomp_add (&mvcomp, k, l,
                vecs_dx[iy*num_x + ix], vecs_dy[iy*num_x + ix]);
            schro_mvcomp_add (&mvcomp, k, l,
                vecs2_dx[iy*num_x + ix], vecs2_dy[iy*num_x + ix]);
          }
        }

        mv = motion_field_get (mf, k, l);
        mv->split = 2;
        mv->pred_mode = 1;
        mv->x1 = mvcomp.dx;
        mv->y1 = mvcomp.dy;
        mv->metric = mvcomp.metric;
      }
    }
    free(vecs_dx);
    free(vecs_dy);
    free(vecs2_dx);
    free(vecs2_dy);
  }

  schro_phasecorr_free (pc);
}

