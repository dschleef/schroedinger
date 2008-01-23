
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schrodebug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define OIL_ENABLE_UNSTABLE_API
#include <liboil/liboilrandom.h>


int
test_full_field (int width, int height, double *a, double *b, int r, int hole)
{
  SchroMotionField *mf;
  SchroMotionVector *mv;
  SchroGlobalMotion gm;
  double mult;
  int i,j;

  mf = schro_motion_field_new (100, 80);

  printf("test_full_field: r=%d hole=%d\n"
      "[%6.4f %6.4f %6.4f %6.4f] [%g %g]\n",
      r, hole,
      a[0], a[1], a[2], a[3], b[0], b[1]);
  for(j=0;j<mf->y_num_blocks;j++){
    for(i=0;i<mf->x_num_blocks;i++){
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;

      mv->x1 = rint((a[0]-1)*8*i + a[1]*8*j + b[0] + r * oil_rand_f64());
      mv->y1 = rint(a[2]*8*i + (a[3]-1)*8*j + b[1] + r * oil_rand_f64());
      if (hole && abs(mf->y_num_blocks/2 - j) < 10 &&
          abs(mf->x_num_blocks/2 - i) < 10) {
        mv->x1 = 0;
        mv->y1 = 0;
      }
    }
  }
  schro_motion_field_global_prediction (mf, &gm, 0);

  mult = (1<<gm.a_exp);
  printf("[%6.4f %6.4f %6.4f %6.4f] [%d %d]\n", gm.a00/mult, gm.a01/mult,
      gm.a10/mult, gm.a11/mult, gm.b0, gm.b1);

  if (fabs(gm.a00/mult - a[0]) > 0.01) return 0;
  if (fabs(gm.a01/mult - a[1]) > 0.01) return 0;
  if (fabs(gm.a10/mult - a[2]) > 0.01) return 0;
  if (fabs(gm.a11/mult - a[3]) > 0.01) return 0;
  if (fabs(gm.b0 - b[0]) > 1) return 0;
  if (fabs(gm.b1 - b[1]) > 1) return 0;

  return 1;
}


double matrices[][6] = {
  { 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 },
  { 1.0, 0.0, 0.0, 1.0, 0.0, 1.0 },
  { 1.1, 0.0, 0.0, 1.0, 0.0, 0.0 },
  { 1.0, 0.1, 0.0, 1.0, 0.0, 0.0 },
  { 1.0, 0.0, 0.1, 1.0, 0.0, 0.0 },
  { 1.0, 0.0, 0.0, 1.1, 0.0, 0.0 }
};

int
main (int argc, char *argv[])
{
  int i;

  schro_init();

  for(i=0;i<6;i++){
    test_full_field(720/8, 576/8, matrices[i], matrices[i] + 4, 0, 0);
    test_full_field(720/8, 576/8, matrices[i], matrices[i] + 4, 1, 0);
    test_full_field(720/8, 576/8, matrices[i], matrices[i] + 4, 0, 1);
  }

  return 0;
}

