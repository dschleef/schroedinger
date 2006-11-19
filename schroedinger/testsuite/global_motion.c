
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schromotion.h>
#include <schroedinger/schropredict.h>
#include <schroedinger/schrodebug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <liboil/liboilrandom.h>

int
main (int argc, char *argv[])
{
  SchroMotionVector *mv;
  SchroMotionField *mf;
  SchroGlobalMotion gm;
  int i;
  int j;
  double mult;

  schro_init();

  //mf = schro_motion_field_new (720/8, 576/8);
  mf = schro_motion_field_new (100, 80);

  for(j=0;j<mf->y_num_blocks;j++){
    for(i=0;i<mf->x_num_blocks;i++){
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;

      mv->u.xy.x = 10;
      mv->u.xy.y = 10;
    }
  }
  schro_motion_field_global_prediction (mf, &gm);
  printf("pan %d %d\n", gm.b0, gm.b1);
  mult = (1<<gm.a_exp);
  printf("a[] %f %f %f %f\n", gm.a00/mult, gm.a01/mult, gm.a10/mult, gm.a11/mult);

  for(j=0;j<mf->y_num_blocks;j++){
    for(i=0;i<mf->x_num_blocks;i++){
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;

      mv->u.xy.x = rint(0 + 1.0*i + 2.0*j + oil_rand_f64());
      mv->u.xy.y = rint(0 + 3.0*i + 4.0*j + oil_rand_f64());
    }
  }
  schro_motion_field_global_prediction (mf, &gm);
  printf("pan %d %d\n", gm.b0, gm.b1);
  mult = (1<<gm.a_exp);
  printf("a[] %f %f %f %f\n", gm.a00/mult, gm.a01/mult, gm.a10/mult, gm.a11/mult);

  for(j=0;j<mf->y_num_blocks;j++){
    for(i=0;i<mf->x_num_blocks;i++){
      mv = mf->motion_vectors + j*mf->x_num_blocks + i;

      mv->u.xy.x = rint(0 + 1.0*i + 2.0*j + oil_rand_f64());
      mv->u.xy.y = rint(0 + 3.0*i + 4.0*j + oil_rand_f64());
      if (abs(mf->y_num_blocks/2 - j) < 10 &&
          abs(mf->x_num_blocks/2 - i) < 10) {
        mv->u.xy.x = 0;
        mv->u.xy.y = 0;
      }
    }
  }
  schro_motion_field_global_prediction (mf, &gm);
  printf("pan %d %d\n", gm.b0, gm.b1);
  mult = (1<<gm.a_exp);
  printf("a[] %f %f %f %f\n", gm.a00/mult, gm.a01/mult, gm.a10/mult, gm.a11/mult);

  return 0;
}

