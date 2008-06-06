
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrooil.h>
#include <orc/orc.h>
#include <math.h>
#include <stdio.h>


void
orc_add2_rshift_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int16_t *s4_2, int n)
{
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] + ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
}

void
orc_add2_rshift_sub_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int16_t *s4_2, int n)
{
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] - ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
}

void
orc_add_const_rshift_s16 (int16_t *d1, int16_t *s1, int16_t *s3_2, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = (s1[i] + s3_2[0])>>s3_2[1];
  }
}

void
orc_add_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n)
{
  static OrcProgram *p = NULL;
  static int s1;
  static int s2;
  static int d1;
  OrcExecutor *ex;

  if (p == NULL) {
#if 0
  int i;
  for(i=0;i<n;i++){
    d[i] = src1[i] + src2[i];
  }
#endif
    p = orc_program_new ();

    d1 = orc_program_add_destination (p, "s16", "d1");
    s1 = orc_program_add_source (p, "s16", "s1");
    s2 = orc_program_add_source (p, "s16", "s2");

    orc_program_append (p, "add_s16", d1, s1, s2);

    orc_program_compile (p);
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array (ex, s1, src1);
  orc_executor_set_array (ex, s2, src2);
  orc_executor_set_array (ex, d1, d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_addc_rshift_s16 (int16_t *d1, int16_t *s1, int16_t *s2_2, int n)
{
  int i;
  int16_t x;

  for(i=0;i<n;i++){
    x = s1[i] + s2_2[0];
    d1[i] = x>>s2_2[1];
  }
}

void
orc_lshift_s16 (int16_t *d1, int16_t *s1, int16_t *s3_1, int n)
{
  int i;
  for(i=0;i<n;i++){
    d1[i] = s1[i]<<s3_1[0];
  }
}

void
orc_mas2_across_add_s16 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int16_t *s4_2, int16_t *s5_2, int n)
{
  int i;
  int x;
  for(i=0;i<n;i++){
    x = s5_2[0];
    x += s2[i]*s4_2[0] + s3[i]*s4_2[1];
    x >>= s5_2[1];
    d[i] = s1[i] + x;
  }
}

void
orc_mas2_add_s16 (int16_t *d1, int16_t *s1, int16_t *s2, int16_t *s3_2,
    int16_t *s4_2, int n)
{
  int i;
  int x;

  for(i=0;i<n;i++){
    x = s4_2[0] + s2[i]*s3_2[0] + s2[i+1]*s3_2[1];
    x >>= s4_2[1];
    d1[i] = s1[i] + x;
  }
}

#if 0
void
orc_mas4_across_add_s16 (int16_t *d, int16_t *s1, int16_t *s2_nx4, int sstr2,
    int16_t *s3_4, int16_t *s4_2, int n)
{
  int i;
  int j;
  int x;
  for(i=0;i<n;i++){
    x = s4_2[0];
    for(j=0;j<4;j++){
      x += OIL_GET(s2_nx4, i*sizeof(int16_t) + j*sstr2, int16_t)*s3_4[j];
    }
    x >>= s4_2[1];
    d[i] = s1[i] + x;
  }
}
#endif

void
orc_mas4_add_s16 (int16_t *d1, int16_t *s1, int16_t *s2, int16_t *s3_4,
    int16_t *s4_2, int n)
{
  int i;
  int x;

  for(i=0;i<n;i++){
    x = s4_2[0] + s2[i]*s3_4[0] + s2[i+1]*s3_4[1] + s2[i+2]*s3_4[2] +
        s2[i+3]*s3_4[3];
    x >>= s4_2[1];
    d1[i] = s1[i] + x;
  }
}

void
orc_subtract_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n)
{
  int i;
  for(i=0;i<n;i++){
    d[i] = src1[i] - src2[i];
  }
}


#if 0
/* within current orc scope */
orc_add_s16_u8
orc_convert_s16_u8
orc_convert_u8_s16
orc_copy_u8
orc_mas8_across_add_s16
orc_mas8_across_u8
orc_mas8_add_s16
orc_mas8_u8_sym_l15
orc_merge_linear_u8
orc_multiply_and_add_s16_u8
orc_subtract_s16_u8
orc_splat_u16_ns
orc_splat_u8_ns

/* 2D */
orc_avg2_12xn_u8
orc_avg2_16xn_u8
orc_avg2_8xn_u8
orc_combine2_16xn_u8
orc_combine2_8xn_u8
orc_combine4_12xn_u8
orc_combine4_16xn_u8
orc_combine4_8xn_u8
orc_multiply_and_acc_12xn_s16_u8
orc_multiply_and_acc_16xn_s16_u8
orc_multiply_and_acc_24xn_s16_u8
orc_multiply_and_acc_6xn_s16_u8
orc_multiply_and_acc_8xn_s16_u8
orc_sad12x12_u8
orc_sad16x16_u8
orc_sad8x8_8xn_u8
orc_sad8x8_u8

/* hard? */
orc_deinterleave2_s16
orc_interleave2_s16
orc_mas10_u8
orc_mas12_addc_rshift_decim2_u8

/* special */
orc_md5
orc_packyuyv
#endif

