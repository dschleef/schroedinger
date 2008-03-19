
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrofft.h>
#include <schroedinger/schrophasecorrelation.h>
#include <math.h>
#include <string.h>

typedef struct _SchroPhaseCorr SchroPhaseCorr;
struct _SchroPhaseCorr {
  SchroEncoderFrame *frame;

  int hshift;
  int vshift;
  int width;
  int height;
  int shift;
  int n;
  int picture_shift;

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

  int num_x;
  int num_y;
  int *vecs_dx;
  int *vecs_dy;
  int *vecs2_dx;
  int *vecs2_dy;
};


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

static int
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
  SCHRO_ASSERT(frame);
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

static void
schro_mvcomp_init (SchroMVComp *mvcomp, SchroFrame *frame, SchroFrame *ref)
{
  memset (mvcomp, 0, sizeof(*mvcomp));

  mvcomp->metric = 32000;
  mvcomp->frame = frame;
  mvcomp->ref = ref;
}

static void
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

static SchroPhaseCorr *
schro_phasecorr_new (int width, int height, int picture_shift,
    int hshift, int vshift)
{
  SchroPhaseCorr *pc;

  pc = schro_malloc0 (sizeof(SchroPhaseCorr));

  pc->picture_shift = picture_shift;

  pc->hshift = hshift;
  pc->vshift = vshift;
  pc->width = 1<<pc->hshift;
  pc->height = 1<<pc->vshift;
  pc->shift = pc->hshift+pc->vshift;
  pc->n = 1<<pc->shift;

  pc->s = schro_malloc(pc->n*sizeof(float));
  pc->c = schro_malloc(pc->n*sizeof(float));
  pc->weight = schro_malloc(pc->n*sizeof(float));
  pc->zero = schro_malloc(pc->n*sizeof(float));
  memset (pc->zero, 0, pc->n*sizeof(float));

  pc->image1 = schro_malloc(pc->n*sizeof(float));
  pc->image2 = schro_malloc(pc->n*sizeof(float));

  pc->ft1r = schro_malloc(pc->n*sizeof(float));
  pc->ft1i = schro_malloc(pc->n*sizeof(float));
  pc->ft2r = schro_malloc(pc->n*sizeof(float));
  pc->ft2i = schro_malloc(pc->n*sizeof(float));
  pc->conv_r = schro_malloc(pc->n*sizeof(float));
  pc->conv_i = schro_malloc(pc->n*sizeof(float));
  pc->resr = schro_malloc(pc->n*sizeof(float));
  pc->resi = schro_malloc(pc->n*sizeof(float));

  generate_weights(pc->weight, pc->width, pc->height);
  schro_fft_generate_tables_f32 (pc->c, pc->s, pc->shift);

  pc->num_x = ((width>>picture_shift) - pc->width)/(pc->width/2) + 2;
  pc->num_y = ((height>>picture_shift) - pc->height)/(pc->height/2) + 2;
  pc->vecs_dx = schro_malloc(sizeof(int)*pc->num_x*pc->num_y);
  pc->vecs_dy = schro_malloc(sizeof(int)*pc->num_x*pc->num_y);
  pc->vecs2_dx = schro_malloc(sizeof(int)*pc->num_x*pc->num_y);
  pc->vecs2_dy = schro_malloc(sizeof(int)*pc->num_x*pc->num_y);

  return pc;
}

static void
schro_phasecorr_free (SchroPhaseCorr *pc)
{
  schro_free(pc->s);
  schro_free(pc->c);
  schro_free(pc->weight);
  schro_free(pc->zero);

  schro_free(pc->image1);
  schro_free(pc->image2);

  schro_free(pc->ft1r);
  schro_free(pc->ft1i);
  schro_free(pc->ft2r);
  schro_free(pc->ft2i);
  schro_free(pc->conv_r);
  schro_free(pc->conv_i);
  schro_free(pc->resr);
  schro_free(pc->resi);

  schro_free(pc->vecs_dx);
  schro_free(pc->vecs_dy);
  schro_free(pc->vecs2_dx);
  schro_free(pc->vecs2_dy);

  schro_free(pc);
}

static void
schro_phasecorr_set_frame (SchroPhaseCorr *pc, SchroEncoderFrame *src)
{
  pc->frame = src;
}

static void
do_phase_corr (SchroPhaseCorr *pc, int ref)
{
  int ix, iy;
  int x, y;
  SchroFrame *src_frame;
  SchroFrame *ref_frame;

  src_frame = get_downsampled(pc->frame, pc->picture_shift);
  if (ref == 0) {
    ref_frame = get_downsampled(pc->frame->ref_frame[0], pc->picture_shift);
  } else {
    ref_frame = get_downsampled(pc->frame->ref_frame[1], pc->picture_shift);
  }

  for(iy=0;iy<pc->num_y;iy++){
    for(ix=0;ix<pc->num_x;ix++){
      double dx, dy;

      x = ((src_frame->width - pc->width) * ix) / (pc->num_x - 1);
      y = ((src_frame->height - pc->height) * iy) / (pc->num_y - 1);

      get_image (pc->image1, src_frame, x, y, pc->width, pc->height, pc->weight);
      get_image (pc->image2, ref_frame, x, y, pc->width, pc->height, pc->weight);

      schro_fft_fwd_f32 (pc->ft1r, pc->ft1i, pc->image1, pc->zero, pc->c, pc->s, pc->shift);
      schro_fft_fwd_f32 (pc->ft2r, pc->ft2i, pc->image2, pc->zero, pc->c, pc->s, pc->shift);

      negate_f32 (pc->ft2i, pc->n);

      complex_mult_f32 (pc->conv_r, pc->conv_i, pc->ft1r, pc->ft1i, pc->ft2r, pc->ft2i, pc->n);
      complex_normalize_f32 (pc->conv_r, pc->conv_i, pc->n);

      schro_fft_rev_f32 (pc->resr, pc->resi, pc->conv_r, pc->conv_i, pc->c, pc->s, pc->shift);

      find_peak (pc->resr, pc->hshift, pc->vshift, &dx, &dy);

#if 0
      schro_dump(SCHRO_DUMP_PHASE_CORR,"%d %d %d %g %g\n",
          frame->frame_number, x, y, dx, dy);
#endif

      pc->vecs_dx[iy*pc->num_x + ix] = rint(-dx * (1<<pc->picture_shift));
      pc->vecs_dy[iy*pc->num_x + ix] = rint(-dy * (1<<pc->picture_shift));

      find_peak (pc->resr, pc->hshift, pc->vshift, &dx, &dy);

      pc->vecs2_dx[iy*pc->num_x + ix] = rint(-dx * (1<<pc->picture_shift));
      pc->vecs2_dy[iy*pc->num_x + ix] = rint(-dy * (1<<pc->picture_shift));
    }
  }

}

static void
do_motion_field (SchroPhaseCorr *pc, int i)
{
  SchroParams *params = &pc->frame->params;
  SchroMotionField *mf;
  SchroFrame *ref;
  SchroFrame *src;
  int x,y;
  int ix, iy;
  int k,l;

  mf = schro_motion_field_new (params->x_num_blocks, params->y_num_blocks);
    src = get_downsampled(pc->frame, 0);
    if (i == 0) {
      ref = get_downsampled(pc->frame->ref_frame[0], 0);
    } else {
      ref = get_downsampled(pc->frame->ref_frame[1], 0);
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

        for(iy=0;iy<pc->num_y;iy++){
          for(ix=0;ix<pc->num_x;ix++){
            x = ((src->width - (pc->width<<pc->picture_shift)) * ix) / (pc->num_x - 1);
            y = ((src->height - (pc->height<<pc->picture_shift)) * iy) / (pc->num_y - 1);

            if (xmax < x || ymax < y ||
                xmin >= x + (pc->width<<pc->picture_shift) ||
                ymin >= y + (pc->height<<pc->picture_shift)) {
              continue;
            }

            schro_mvcomp_add (&mvcomp, k, l,
                pc->vecs_dx[iy*pc->num_x + ix], pc->vecs_dy[iy*pc->num_x + ix]);
            schro_mvcomp_add (&mvcomp, k, l,
                pc->vecs2_dx[iy*pc->num_x + ix], pc->vecs2_dy[iy*pc->num_x + ix]);
          }
        }

        mv = motion_field_get (mf, k, l);
        mv->split = 2;
        mv->pred_mode = 1;
        mv->dx[0] = mvcomp.dx;
        mv->dy[0] = mvcomp.dy;
        mv->metric = mvcomp.metric;
      }
    }

    schro_motion_field_scan (mf, params, src, ref, 2);

    schro_motion_field_lshift (mf, params->mv_precision);

    schro_list_append (pc->frame->motion_field_list, mf);
  }

void
schro_encoder_phasecorr_estimation (SchroMotionEst *me)
{
  SchroParams *params = me->params;
  SchroPhaseCorr *pc;
  int ref;
  int i;

  for(i=0;i<4;i++) {
    SCHRO_DEBUG("block size %dx%d", 1<<(2+5+i), 1<<(2+4+i));
    if (me->encoder_frame->filtered_frame->width < 1<<(2+5+i) ||
        me->encoder_frame->filtered_frame->height < 1<<(2+4+i)) {
      continue;
    }

    pc = schro_phasecorr_new (me->encoder_frame->filtered_frame->width,
        me->encoder_frame->filtered_frame->height, 2, 5+i, 4+i);
    schro_phasecorr_set_frame (pc, me->encoder_frame);

    for(ref=0;ref<params->num_refs;ref++){
      do_phase_corr (pc, ref);
      do_motion_field (pc, ref);
    }

    schro_phasecorr_free (pc);
  }
}

