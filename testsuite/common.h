
#ifndef __COMMON_H__
#define __COMMON_H__

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <schroedinger/schroframe.h>
#include <orc-test/orcrandom.h>

extern OrcRandomContext context;

double sgn(double x);
double random_std (void);
double random_triangle (void);
int gain_to_quant_index (double x);
double sum_f64 (double *a, int n);
double multsum_f64 (double *a, double *b, int n);

#define TEST_PATTERN_NAME_SIZE 100

int test_pattern_get_n_generators (void);
void test_pattern_generate (SchroFrameData *frame, char *name, int n);

int frame_data_compare (SchroFrameData *dest, SchroFrameData *src);
void frame_data_dump (SchroFrameData *dest, SchroFrameData *src);
void frame_data_dump_full (SchroFrameData *dest, SchroFrameData *src,
    SchroFrameData *orig);

int frame_compare (SchroFrame *dest, SchroFrame *src);
void frame_dump (SchroFrame *test, SchroFrame *ref);

int parse_packet (FILE *file, void **data, int *size);

#define rand_u8() (rand()&0xff)
#define rand_f64() (((rand()/(RAND_MAX+1.0))+rand())/(RAND_MAX+1.0))

void interleave (int16_t *a, int n);
void deinterleave (int16_t *a, int n);
void extend(int16_t *a, int n);
void synth(int16_t *a, int n, int filter);
void split (int16_t *a, int n, int filter);

#endif

