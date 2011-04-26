
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrowavelet.h>
#include <schroedinger/schrofft.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "common.h"

#define SHIFT 8
#define SIZE (1<<SHIFT)
#define N_TRIALS 100
#define CHUNK_SIZE (SIZE>>5)
#define AMPLITUDE 100

int fail = 0;

float costable[SIZE*SIZE];
float sintable[SIZE*SIZE];
float dr[SIZE*SIZE];
float di[SIZE*SIZE];
float sr[SIZE*SIZE];
float si[SIZE*SIZE];
float power[SIZE*SIZE];


void
generate_noise (SchroFrame *frame, int n_transforms, double *weights)
{
  int i;
  int j;
  int k;
  int l;
  int pos;
  int w, h, x_offset, y_offset, y_skip;
  int16_t *data;
  SchroFrameData *comp;

  for(k=0;k<3;k++){
    comp = frame->components + k;

    for(l=0;l<1+3*n_transforms;l++){
      pos = schro_subband_get_position (l);

      w = comp->width >> (n_transforms - SCHRO_SUBBAND_SHIFT(pos));
      h = comp->height >> (n_transforms - SCHRO_SUBBAND_SHIFT(pos));
      y_skip = 1<<(n_transforms - SCHRO_SUBBAND_SHIFT(pos));
      if (pos&1) {
        x_offset = w;
      } else {
        x_offset = 0;
      }
      if (pos&2) {
        y_offset = y_skip / 2;
      } else {
        y_offset = 0;
      }
//printf("%d %d w %d h %d x_offset %d y_offset %d y_skip %d\n", l, pos, w, h, x_offset, y_offset, y_skip);

      for(j=0;j<h;j++){
        data = OFFSET(comp->data,
            (j*y_skip + y_offset)*comp->stride);
        data += x_offset;
        for(i=0;i<w;i++){
          data[i] = floor(0.5 + random_std()*AMPLITUDE*weights[l]);
          //data[i] = floor(0.5 + random_std()*AMPLITUDE);
        }
      }
    }
  }
}

int
main (int argc, char *argv[])
{
  SchroEncoder *encoder;
  SchroParams params;
  SchroVideoFormat video_format;
  int filter;
  SchroFrame *frame;
  int transform_depth;
  int i,j;
  int k;
  int16_t *tmp;

  schro_init();

  encoder = schro_encoder_new ();

  frame = schro_frame_new_and_alloc (NULL, SCHRO_FRAME_FORMAT_S16_444, SIZE, SIZE);
  tmp = schro_malloc (sizeof(int16_t) * SIZE * 2);

  schro_fft_generate_tables_f32 (costable, sintable, 2*SHIFT);

  filter = 6;
  transform_depth = 4;

  schro_video_format_set_std_video_format (&video_format, 0);
  video_format.width = SIZE;
  video_format.height = SIZE;

  memset (&params, 0, sizeof(params));
  params.video_format = &video_format;
  schro_params_init (&params, 0);
  params.wavelet_filter_index = filter;
  params.transform_depth = transform_depth;
  schro_params_calculate_iwt_sizes (&params);

  for(i=0;i<SIZE*SIZE;i++) power[i] = 0;

  for(k=0;k<N_TRIALS;k++){
    generate_noise (frame, transform_depth,
        encoder->intra_subband_weights[filter][transform_depth-1]);

    schro_wavelet_inverse_transform_2d (frame->components + 0,
        frame->components + 1, filter, tmp);

    for(j=0;j<SIZE;j++){
      int16_t *line;
      line = OFFSET(frame->components[0].data,
          frame->components[0].stride * j);
      for(i=0;i<SIZE;i++){
        sr[j*SIZE+i] = line[i];
        si[j*SIZE+i] = 0;
      }
    }

    schro_fft_fwd_f32 (dr, di, sr, si, costable, sintable, 2*SHIFT);

    for(i=0;i<SIZE*SIZE;i++) {
      power[i] += (dr[i]*dr[i]+di[i]*di[i])*(1.0/(SIZE*SIZE));
    }
  }

  //power[0] *= 4.0/SIZE;
  //power[0] = 0;
  for(j=0;j<SIZE/2;j+=CHUNK_SIZE){
    for(i=0;i<SIZE/2;i+=CHUNK_SIZE){
      int ii,jj;
      double sum = 0;
      for(jj=0;jj<CHUNK_SIZE;jj++){
        for(ii=0;ii<CHUNK_SIZE;ii++){
          sum += power[(j+jj)*SIZE+(i+ii)];
        }
      }
      sum /= N_TRIALS*CHUNK_SIZE*CHUNK_SIZE;
      printf("%d %d %g\n", j, i, sqrt(sum)/AMPLITUDE);
    }
    printf("\n");
  }

  return fail;
}

