#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>

#define DATATYPE short
#define WORKTYPE int

#include <liboil/liboil.h>

#include "cudawavelet.h"

#define COUNT 100

#define HORIZONTAL
#define VERTICAL

// Test pattern
#define TEST_RANDOM
//#define TEST_HORIZONTAL
//#define TEST_HORIZONTALQ
//#define TEST_VERTICAL
//#define TEST_VERTICALQ

#define SEED_FIXED

#include "wavelet.h"

struct wl_type {
    const char *name;
    int filter;
} wl_types[] = {
    {"DESL_9_3",SCHRO_WAVELET_DESL_9_3},
    {"5_3",SCHRO_WAVELET_5_3},
    {"13_5",SCHRO_WAVELET_13_5},
    {"HAAR_0",SCHRO_WAVELET_HAAR_0},
    {"HAAR_1",SCHRO_WAVELET_HAAR_1},
    {"HAAR_2",SCHRO_WAVELET_HAAR_2},
    {"FIDELITY",SCHRO_WAVELET_FIDELITY},
    {"DAUB_9_7",SCHRO_WAVELET_DAUB_9_7},
    {NULL}
};

int main( int argc, char** argv) 
{
    cutPrintInfo();
    oil_init();

    // HDTV 1920x1080
    //unsigned int width = 2048;//1920;
    //unsigned int height = 1080;
    unsigned int width = 1536;
    unsigned int height = 1152;

    /// Compare timings on wavelets
    unsigned int size = width*height*sizeof(DATATYPE);
    unsigned int levels = 3;

    /* Data on CPU */
    DATATYPE *data = (DATATYPE*)malloc(size);
#ifdef SEED_FIXED
    srand(20);
#else
    srand(time(NULL));
#endif
    for(int y=0; y<height; ++y)
        for(int x=0; x<width; ++x)
#ifdef TEST_RANDOM
            data[y*width+x] = rand()&0xFF;
#endif
#ifdef TEST_HORIZONTAL
            data[y*width+x] = x;
#endif
#ifdef TEST_VERTICAL
            data[y*width+x] = y;
#endif
#ifdef TEST_HORIZONTALQ
            data[y*width+x] = (x*x)%256;
#endif
#ifdef TEST_VERTICALQ
            data[y*width+x] = (y*y)%256;
#endif
#ifdef VERBOSE
    printf("Orig:\n");
    for(int y=0; y<height; ++y)
    {
        for(int x=0; x<width; ++x)
            printf("%4i ", (int)data[y*width+x]);
        printf("\n");
    }
#endif
    DATATYPE *data_orig = (DATATYPE*)malloc(size);
    memcpy((void*)data_orig, (void*)data, size);

    DATATYPE *data2 = (DATATYPE*)malloc(size);


    /** Data on GPU */
    DATATYPE *d_data = NULL;
    DATATYPE *d_data_orig = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, size));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_orig, size));

    /** Copy to GPU */
    CUDA_SAFE_CALL(cudaMemcpy(d_data_orig, data, size, cudaMemcpyHostToDevice));

    printf("Wavelet\t\t\thw(ms)\tsw(ms)\tspeedup\n\n");
    for(int i=0; wl_types[i].name!=NULL; ++i)
    {
        for(int j=0; j<2; ++j)
        {
            bool inverse = j;
            int filter = wl_types[i].filter;
            printf("%-8s", wl_types[i].name);
            if(inverse)
                printf("(s)\t");
            else
                printf("(a)\t");
            fflush(stdout);

            /** Invoke kernel */
            unsigned int timer;

            CUDA_SAFE_CALL(cutCreateTimer(&timer));
            cudaThreadSynchronize();
            CUDA_SAFE_CALL(cutStartTimer(timer));
            for(int iter=0; iter<COUNT; ++iter)
            {
                cudaMemcpy(d_data, d_data_orig, size, cudaMemcpyDeviceToDevice);
                for(int l=0; l<levels; ++l)
                    if(inverse)
                        cuda_wavelet_inverse_transform_2d(filter, d_data, (width<<l)*2, width>>l, height>>l);
                    else
                        cuda_wavelet_transform_2d(filter, d_data, (width<<l)*2, width>>l, height>>l);
            }
            cudaThreadSynchronize();
            CUDA_SAFE_CALL(cutStopTimer(timer));

            short *tmp = (short*)malloc(size);

            unsigned int timer_sw;

            CUDA_SAFE_CALL(cutCreateTimer(&timer_sw));
            CUDA_SAFE_CALL(cutStartTimer(timer_sw));
            for(int iter=0; iter<COUNT; ++iter)
            {
                memcpy((void*)data2, (void*)data_orig, size);
                for(int l=0; l<levels; ++l)
                    if(inverse)
                        schro_wavelet_inverse_transform_2d(filter, data2, (width<<l)*2, width>>l, height>>l, tmp);
                    else
                        schro_wavelet_transform_2d(filter, data2, (width<<l)*2, width>>l, height>>l, tmp);
            }
            CUDA_SAFE_CALL(cutStopTimer(timer_sw));

            /** Copy back */
            CUDA_SAFE_CALL(cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost));

            int errors = 0;
            for(int y=0; y<height; ++y)
            {
                for(int x=0; x<width; ++x)
                {
                    if(data2[y*width+x]!=data[y*width+x])
                    {
                        errors++;
                    }
                }
            }
            if(errors)
            {
                printf(" There were %i errors! ",errors);
            }
            printf("\t%3.2f\t%3.2f\t%3.2f\n", cutGetTimerValue(timer)/COUNT, cutGetTimerValue(timer_sw)/COUNT, cutGetTimerValue(timer_sw)/cutGetTimerValue(timer));

            cutDeleteTimer(timer);
            cutDeleteTimer(timer_sw);
        }
    }

    free(data);
    cudaFree(d_data);

}

