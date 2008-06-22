
#ifndef __OPENGL_UTIL_H__
#define __OPENGL_UTIL_H__

#include <schroedinger/schro.h>
#include <schroedinger/opengl/schroopengl.h>

#define OPENGL_CUSTOM_PATTERN_NONE             1
#define OPENGL_CUSTOM_PATTERN_RANDOM           2
#define OPENGL_CUSTOM_PATTERN_RANDOM_U8        3
#define OPENGL_CUSTOM_PATTERN_RANDOM_S8        4
#define OPENGL_CUSTOM_PATTERN_CONST_1          5
#define OPENGL_CUSTOM_PATTERN_CONST_16         6
#define OPENGL_CUSTOM_PATTERN_CONST_MIN        7
#define OPENGL_CUSTOM_PATTERN_CONST_MIN_U8     8
#define OPENGL_CUSTOM_PATTERN_CONST_MIDDLE     9
#define OPENGL_CUSTOM_PATTERN_CONST_MIDDLE_U8 10
#define OPENGL_CUSTOM_PATTERN_CONST_MAX       11
#define OPENGL_CUSTOM_PATTERN_CONST_MAX_U8    12

extern int _benchmark;
extern int _failed;
extern int _generators;
extern SchroMemoryDomain *_cpu_domain;
extern SchroMemoryDomain *_opengl_domain;
extern SchroOpenGL *_opengl;

int opengl_format_name (SchroFrameFormat format, char *format_name, int size);
int opengl_filter_name (int filter, char *filter_name, int size);
void opengl_custom_pattern_generate (SchroFrame *cpu_frame, int pattern_type,
    int pattern_index, char* pattern_name);

#endif
