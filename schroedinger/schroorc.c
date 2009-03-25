
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schrooil.h>
#include <schroedinger/schrodebug.h>
#include <orc/orc.h>
#include <math.h>
#include <stdio.h>


void
orc_add2_rshift_add_s16_22 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int n)
{
#if 0
  static const int16_t s4_2[] = { 2, 2 };
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] + ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int d1;
    int ret;

    p = orc_program_new ();
    d1 = orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s2");
    orc_program_add_source (p, 2, "s3");
    orc_program_add_constant (p, 2, 2, "c1");
    orc_program_add_constant (p, 2, 2, "c2");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_str (p, "addw", "t1", "s2", "s3");
    orc_program_append_str (p, "addw", "t1", "t1", "c1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "c2");
    orc_program_append_str (p, "addw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s2", s2);
  orc_executor_set_array_str (ex, "s3", s3);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add2_rshift_sub_s16_22 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int n)
{
#if 0
  static const int16_t s4_2[] = { 2, 2 };
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] + ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int d1;
    int ret;

    p = orc_program_new ();
    d1 = orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s2");
    orc_program_add_source (p, 2, "s3");
    orc_program_add_constant (p, 2, 2, "c1");
    orc_program_add_constant (p, 2, 2, "c2");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_str (p, "addw", "t1", "s2", "s3");
    orc_program_append_str (p, "addw", "t1", "t1", "c1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "c2");
    orc_program_append_str (p, "subw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s2", s2);
  orc_executor_set_array_str (ex, "s3", s3);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add2_rshift_add_s16_11 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int n)
{
#if 0
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] - ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new ();
    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s2");
    orc_program_add_source (p, 2, "s3");
#if 1
    orc_program_add_constant (p, 2, 1, "c1");
    orc_program_add_constant (p, 2, 1, "c2");
#else
    orc_program_add_constant (p, 2, 0x8000, "c1");
#endif
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 2, "t2");
    orc_program_add_temporary (p, 2, "t3");

#if 1
    orc_program_append_str (p, "addw", "t1", "s2", "s3");
    orc_program_append_str (p, "addw", "t1", "t1", "c1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "c2");
#else
    orc_program_append_str (p, "xorw", "t1", "c1", "s2");
    orc_program_append_str (p, "xorw", "t2", "c1", "s3");
    orc_program_append_str (p, "avguw", "t3", "t1", "t2");
    orc_program_append_str (p, "xorw", "t1", "c1", "t3");
#endif
    orc_program_append_str (p, "addw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s2", s2);
  orc_executor_set_array_str (ex, "s3", s3);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add2_rshift_sub_s16_11 (int16_t *d, int16_t *s1, int16_t *s2, int16_t *s3,
    int n)
{
#if 0
  int i;
  for(i=0;i<n;i++) {
    d[i] = s1[i] - ((s2[i] + s3[i] + s4_2[0])>>s4_2[1]);
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new ();
    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s2");
    orc_program_add_source (p, 2, "s3");
#if 1
    orc_program_add_constant (p, 2, 1, "c1");
    orc_program_add_constant (p, 2, 1, "c2");
#else
    orc_program_add_constant (p, 2, 0x8000, "c1");
#endif
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 2, "t2");
    orc_program_add_temporary (p, 2, "t3");

#if 1
    orc_program_append_str (p, "addw", "t1", "s2", "s3");
    orc_program_append_str (p, "addw", "t1", "t1", "c1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "c2");
#else
    orc_program_append_str (p, "xorw", "t1", "c1", "s2");
    orc_program_append_str (p, "xorw", "t2", "c1", "s3");
    orc_program_append_str (p, "avguw", "t3", "t1", "t2");
    orc_program_append_str (p, "xorw", "t1", "c1", "t3");
#endif
    orc_program_append_str (p, "subw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s2", s2);
  orc_executor_set_array_str (ex, "s3", s3);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add_const_rshift_s16_11 (int16_t *d1, int16_t *s1, int n)
{
#if 0
  int i;
  for(i=0;i<n;i++){
    d1[i] = (s1[i] + s3_2[0])>>s3_2[1];
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,2);
    orc_program_add_constant (p, 2, 1, "c1");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_str (p, "addw", "t1", "s1", "c1");
    orc_program_append_str (p, "shrsw", "d1", "t1", "c1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "d1", d1);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,2);

    orc_program_append_str (p, "addw", "d1", "s1", "s2");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "s2", src2);
  orc_executor_set_array_str (ex, "d1", d);

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
orc_lshift1_s16 (int16_t *d1, int16_t *s1, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_ds (2,2);
    orc_program_add_constant (p, 2, 1, "c1");

    orc_program_append_str (p, "shlw", "d1", "s1", "c1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "d1", d1);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_lshift2_s16 (int16_t *d1, int16_t *s1, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_ds (2,2);
    orc_program_add_constant (p, 2, 2, "c1");

    orc_program_append_str (p, "shlw", "d1", "s1", "c1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "d1", d1);

  orc_executor_run (ex);
  orc_executor_free (ex);
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
orc_mas2_add_s16 (int16_t *d1, int16_t *s1, int16_t *s2, int mult, int offset,
    int shift, int n)
{
#if 1
  int i;
  int x;

  for(i=0;i<n;i++){
    x = offset + (s2[i] + s2[i+1])*mult;
    x >>= shift;
    d1[i] = s1[i] + x;
  }
#else
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,2);
    orc_program_add_src (p, 2, "s3");
    orc_program_add_temporary (p, 4, "t1");
    orc_program_add_temporary (p, 2, "t2");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_parameter (p, 4, "p2");
    orc_program_add_parameter (p, 2, "p3");

    orc_program_append_str (p, "addw", "t1", "s2", "s3");
    orc_program_append_str (p, "mulswl", "t1", "t1", "p1");
    orc_program_append_str (p, "addl", "t1", "t1", "p2");
    orc_program_append_str (p, "shll", "t1", "s1", "p3");
    orc_program_append_str (p, "convlw", "t2", "t1");
    orc_program_append_str (p, "addl", "d1", "t2", "s1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "d1", d1);

  orc_executor_run (ex);
  orc_executor_free (ex);
#endif
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
orc_mas4_add_s16_1991 (int16_t *d1, int16_t *s1, int16_t *s2, int shift, int n)
{
#if 0
  int i;
  int x;
  static const int16_t s3_4[] = { 1, -9, -9, 1 };
  static const int16_t s4_2[] = { 7, 4 };

  for(i=0;i<n;i++){
    x = s4_2[0] + s2[i]*s3_4[0] + s2[i+1]*s3_4[1] + s2[i+2]*s3_4[2] +
        s2[i+3]*s3_4[3];
    x >>= s4_2[1];
    d1[i] = s1[i] + x;
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new ();
    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s20");
    orc_program_add_source (p, 2, "s21");
    orc_program_add_source (p, 2, "s22");
    orc_program_add_source (p, 2, "s23");
    orc_program_add_constant (p, 2, 9, "c1");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_parameter (p, 2, "p2");
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 2, "t2");

    orc_program_append_str (p, "addw", "t1", "s21", "s22");
    orc_program_append_str (p, "mullw", "t1", "t1", "c1");
    orc_program_append_str (p, "addw", "t2", "s20", "s23");
    orc_program_append_str (p, "subw", "t1", "t1", "t2");
    orc_program_append_str (p, "addw", "t1", "t1", "p1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "p2");
    orc_program_append_str (p, "addw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s20", s2+0);
  orc_executor_set_array_str (ex, "s21", s2+1);
  orc_executor_set_array_str (ex, "s22", s2+2);
  orc_executor_set_array_str (ex, "s23", s2+3);
  orc_executor_set_array_str (ex, "d1", d1);
  orc_executor_set_param_str (ex, "p1", (1<<(shift-1)));
  orc_executor_set_param_str (ex, "p2", shift);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_mas4_sub_s16_1991 (int16_t *d1, int16_t *s1, int16_t *s2, int shift, int n)
{
#if 0
  int i;
  int x;
  static const int16_t s3_4[] = { 1, -9, -9, 1 };
  static const int16_t s4_2[] = { 7, 4 };

  for(i=0;i<n;i++){
    x = s4_2[0] + s2[i]*s3_4[0] + s2[i+1]*s3_4[1] + s2[i+2]*s3_4[2] +
        s2[i+3]*s3_4[3];
    x >>= s4_2[1];
    d1[i] = s1[i] + x;
  }
#endif
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new ();
    orc_program_add_destination (p, 2, "d1");
    orc_program_add_source (p, 2, "s1");
    orc_program_add_source (p, 2, "s20");
    orc_program_add_source (p, 2, "s21");
    orc_program_add_source (p, 2, "s22");
    orc_program_add_source (p, 2, "s23");
    orc_program_add_constant (p, 2, 9, "c1");
    orc_program_add_parameter (p, 2, "p1");
    orc_program_add_parameter (p, 2, "p2");
    orc_program_add_temporary (p, 2, "t1");
    orc_program_add_temporary (p, 2, "t2");

    orc_program_append_str (p, "addw", "t1", "s21", "s22");
    orc_program_append_str (p, "mullw", "t1", "t1", "c1");
    orc_program_append_str (p, "addw", "t2", "s20", "s23");
    orc_program_append_str (p, "subw", "t1", "t1", "t2");
    orc_program_append_str (p, "addw", "t1", "t1", "p1");
    orc_program_append_str (p, "shrsw", "t1", "t1", "p2");
    orc_program_append_str (p, "subw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", s1);
  orc_executor_set_array_str (ex, "s20", s2+0);
  orc_executor_set_array_str (ex, "s21", s2+1);
  orc_executor_set_array_str (ex, "s22", s2+2);
  orc_executor_set_array_str (ex, "s23", s2+3);
  orc_executor_set_array_str (ex, "d1", d1);
  orc_executor_set_param_str (ex, "p1", (1<<(shift-1)));
  orc_executor_set_param_str (ex, "p2", shift);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_subtract_s16 (int16_t *d, int16_t *src1, int16_t *src2, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,2);

    orc_program_append_str (p, "subw", "d1", "s1", "s2");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "s2", src2);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_memcpy (void *dest, void *src, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_ds (1,1);
    orc_program_append_ds_str (p, "copyb", "d1", "s1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src);
  orc_executor_set_array_str (ex, "d1", dest);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_add_s16_u8 (int16_t *d, int16_t *src1, uint8_t *src2, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,1);
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_ds_str (p, "convubw", "t1", "s2");
    orc_program_append_str (p, "addw", "d1", "t1", "s1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "s2", src2);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_convert_s16_u8 (int16_t *d, uint8_t *src1, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_ds (2,1);

    orc_program_append_ds_str (p, "convubw", "d1", "s1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_convert_u8_s16 (uint8_t *d, int16_t *src1, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_ds (1,2);

    orc_program_append_ds_str (p, "convsuswb", "d1", "s1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_subtract_s16_u8 (int16_t *d, int16_t *src1, uint8_t *src2, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,1);
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_ds_str (p, "convubw", "t1", "s2");
    orc_program_append_str (p, "subw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "s2", src2);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}

void
orc_multiply_and_add_s16_u8 (int16_t *d, int16_t *src1, int16_t *src2,
    uint8_t *src3, int n)
{
  static OrcProgram *p = NULL;
  OrcExecutor *ex;

  if (p == NULL) {
    int ret;

    p = orc_program_new_dss (2,2,2);
    orc_program_add_source (p, 1, "s3");
    orc_program_add_temporary (p, 2, "t1");

    orc_program_append_ds_str (p, "convubw", "t1", "s3");
    orc_program_append_str (p, "mullw", "t1", "t1", "s2");
    orc_program_append_str (p, "addw", "d1", "s1", "t1");

    ret = orc_program_compile (p);
    if (!ret) {
      SCHRO_ERROR("Orc compiler failure");
    }
  }

  ex = orc_executor_new (p);
  orc_executor_set_n (ex, n);
  orc_executor_set_array_str (ex, "s1", src1);
  orc_executor_set_array_str (ex, "s2", src2);
  orc_executor_set_array_str (ex, "s3", src3);
  orc_executor_set_array_str (ex, "d1", d);

  orc_executor_run (ex);
  orc_executor_free (ex);
}


#if 0
/* within current orc scope */
orc_copy_u8
orc_mas8_across_add_s16
orc_mas8_across_u8
orc_mas8_add_s16
orc_mas8_u8_sym_l15
orc_merge_linear_u8
orc_splat_u16_ns
orc_splat_u8_ns
orc_deinterleave2_s16
orc_interleave2_s16
orc_packyuyv

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
orc_mas10_u8
orc_mas12_addc_rshift_decim2_u8

/* special */
orc_md5
#endif

