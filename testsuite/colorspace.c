
#include <schroedinger/schro.h>
#include <schroedinger/schroutils.h>

#include <stdio.h>
#include <math.h>
#include <string.h>

typedef struct _ColorMatrix ColorMatrix;
struct _ColorMatrix {
  double m[4][4];
};

void color_matrix_build (ColorMatrix *dst);
void color_matrix_apply_u8 (ColorMatrix *m, uint8_t *dest, uint8_t *src);
void color_matrix_apply_f64_u8 (ColorMatrix *m, double *dest, uint8_t *src);
void color_matrix_dump(ColorMatrix *m);

uint8_t colors[][3] = {
  { 168, 44, 136 },
  { 61, 153, 99 },
  { 35, 174, 152 },
#if 0
  { 235, 128, 128 },
  { 226, 0, 155 },
  { 179, 170, 0 },
  { 150, 46, 21 },
  { 105, 212, 235 },
  { 76, 85, 255 },
  { 29, 255, 107 },
  { 16, 128, 128 },
#endif
};

int
main (int argc, char *argv[])
{
  ColorMatrix m;
  double dest[3];
  //uint8_t src[3];
  int i;

  color_matrix_build (&m);
  color_matrix_dump (&m);
  for(i=0;i<8;i++){
    color_matrix_apply_f64_u8 (&m, dest, colors[i]);
#if 0
    src[0] = (i&2)?191:0;
    src[1] = (i&4)?191:0;
    src[2] = (i&1)?191:0;
    color_matrix_apply_f64_u8 (&m, dest, src);
#endif
    printf("%8.4g %8.4g %8.4g\n", dest[0], dest[1], dest[2]);
  }

  return 0;
}


void
color_matrix_set_identity (ColorMatrix *m)
{
  int i,j;

  for(i=0;i<4;i++){
    for(j=0;j<4;j++){
      m->m[i][j] = (i==j);
    }
  }
}

/* Prettyprint a 4x4 matrix @m@ */
void
color_matrix_dump(ColorMatrix *m)
{
  int i,j;

  printf("[\n");
  for (i = 0; i < 4; i++) {
    printf("  ");
    for (j = 0; j < 4; j++) {
      printf(" %8.5g", m->m[i][j]);
    }
    printf("\n");
  }
  printf("]\n");
}

/* Perform 4x4 matrix multiplication:
 *  - @dst@ = @a@ * @b@
 *  - @dst@ may be a pointer to @a@ andor @b@
 */
void
color_matrix_multiply (ColorMatrix *dst, ColorMatrix *a, ColorMatrix *b)
{
  ColorMatrix tmp;
  int i,j,k;

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      double x = 0;
      for (k = 0; k < 4; k++) {
        x += a->m[i][k] * b->m[k][j];
      }
      tmp.m[i][j] = x;
    }
  }

  memcpy(dst, &tmp, sizeof(ColorMatrix));
}



void
color_matrix_offset_components (ColorMatrix *m, double a1, double a2,
    double a3)
{
  ColorMatrix a;

  color_matrix_set_identity (&a);
  a.m[0][3] = a1;
  a.m[1][3] = a2;
  a.m[2][3] = a3;
  color_matrix_multiply (m, &a, m);
}

void
color_matrix_scale_components (ColorMatrix *m, double a1, double a2,
    double a3)
{
  ColorMatrix a;

  color_matrix_set_identity (&a);
  a.m[0][0] = a1;
  a.m[1][1] = a2;
  a.m[2][2] = a3;
  color_matrix_multiply (m, &a, m);
}

void
color_matrix_YCbCr_to_RGB (ColorMatrix *m, double Kr, double Kb)
{
  double Kg = 1.0 - Kr - Kb;
  ColorMatrix k = {
    {
      {1.,  0.,              2*(1-Kr),       0.},
      {1., -2*Kb*(1-Kb)/Kg, -2*Kr*(1-Kr)/Kg, 0.},
      {1.,  2*(1-Kb),        0.,             0.},
      {0.,  0.,              0.,             1.},
    }
  };

  color_matrix_multiply (m, &k, m);
}

void
color_matrix_RGB_to_YCbCr (ColorMatrix *m, double Kr, double Kb)
{
  double Kg = 1.0 - Kr - Kb;
  ColorMatrix k;
  double x;

  k.m[0][0] = Kr;
  k.m[0][1] = Kg;
  k.m[0][2] = Kb;
  k.m[0][3] = 0;

  x = 1/(2*(1-Kb));
  k.m[1][0] = -x*Kr;
  k.m[1][1] = -x*Kg;
  k.m[1][2] = x*(1-Kb);
  k.m[1][3] = 0;

  x = 1/(2*(1-Kr));
  k.m[2][0] = x*(1-Kr);
  k.m[2][1] = -x*Kg;
  k.m[2][2] = -x*Kb;
  k.m[2][3] = 0;

  k.m[3][0] = 0;
  k.m[3][1] = 0;
  k.m[3][2] = 0;
  k.m[3][3] = 1;

  color_matrix_multiply (m, &k, m);
}

void
color_matrix_build (ColorMatrix *dst)
{
#if 0
  /*
   * At this point, everything is in YCbCr
   * All components are in the range [0,255]
   */
  color_matrix_set_identity (dst);

  /* offset required to get input video black to (0.,0.,0.) */
  color_matrix_offset_components (dst, -16, -128, -128);

  /* scale required to get input video black to (0.,0.,0.) */
  color_matrix_scale_components (dst, (1/219.0), (1/224.0), (1/224.0));

  /* colour matrix, YCbCr -> RGB */
  /* Requires Y in [0,1.0], Cb&Cr in [-0.5,0.5] */
  color_matrix_YCbCr_to_RGB (dst, 0.2990, 0.1140);  // SD
  //color_matrix_YCbCr_to_RGB (dst, 0.2126, 0.0722);  // HD

  /*
   * We are now in RGB space
   */

  /* scale to output range. */
  color_matrix_scale_components (dst, 255.0, 255.0, 255.0);
#else
  color_matrix_set_identity (dst);

  color_matrix_scale_components (dst, (1/255.0), (1/255.0), (1/255.0));

  color_matrix_RGB_to_YCbCr (dst, 0.2990, 0.1140); // SD
  //color_matrix_RGB_to_YCbCr (dst, 0.2126, 0.0722); // HD
  color_matrix_RGB_to_YCbCr (dst, 0.212, 0.087); // SMPTE 240M

  color_matrix_scale_components (dst, 219.0, 224.0, 224.0);

  color_matrix_offset_components (dst, 16, 128, 128);
#endif
}

void
color_matrix_apply_u8 (ColorMatrix *m, uint8_t *dest, uint8_t *src)
{
  int i;

  for (i = 0; i < 3; i++) {
    double x = 0;
    x += m->m[i][0] * src[0];
    x += m->m[i][1] * src[1];
    x += m->m[i][2] * src[2];
    x += m->m[i][3];
    dest[i] = CLAMP(floor(x + 0.5),0,255);
  }
}

void
color_matrix_apply_f64_u8 (ColorMatrix *m, double *dest, uint8_t *src)
{
  int i;

  for (i = 0; i < 3; i++) {
    double x = 0;
    x += m->m[i][0] * src[0];
    x += m->m[i][1] * src[1];
    x += m->m[i][2] * src[2];
    x += m->m[i][3];
    dest[i] = CLAMP(floor(x + 0.5),0,255);
    //dest[i] = x;
  }
}

/* 
 * SMPTE 170M
 * ==========
 *
 * NTSC, SD, 525 lines
 *
 * color primaries:
 * "SMPTE C set"
 *  r 0.630 0.340
 *  g 0.310 0.595
 *  b 0.155 0.070
 * w D65
 *
 * "NTSC 1953"
 * r 0.67 0.33
 * g 0.21 0.71
 * b 0.14 0.08
 * w D65
 *
 * transfer function:
 *
 * LT= [(Vr + 0.099)/1.099]^(1/0.4500) for 0.0812 ≤ Vr ≤ 1
 * LT = Vr/4.500                       for 0 ≤ Vr < 0.0812
 *
 * also defines SD SMPTE color bars
 */

/*
 * EBU 3213-E
 * ==========
 *
 *  r 0.64 0.33
 *  g 0.29 0.60
 *  b 0.15 0.06
 *  w D65
 *
 */


