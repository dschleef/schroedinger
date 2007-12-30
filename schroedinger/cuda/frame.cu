/// XXX speed up by using shared memory / coalescing?

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>

#define THREADS 256

#include "common.h"
#include "convert_base_coalesce.h"
//#include "convert_base.h"
#include "convert_packed.h"
#include "arith_coalesce.h"
//#include "arith.h"



extern "C" {

void cuda_convert_u8_s16(uint8_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_s16<<<grid_size, block_size, shared_size>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_s16_u8<<<grid_size, block_size, shared_size>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_u8_u8(uint8_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_u8<<<grid_size, block_size, shared_size>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_s16_s16<<<grid_size, block_size, shared_size>>>(dst, dstride, dwidth, src, sstride, swidth, sheight);
}
void cuda_convert_u8_422_yuyv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_422_yuyv<<<grid_size, block_size, shared_size>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_u8_422_uyvy(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_422_uyvy<<<grid_size, block_size, shared_size>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_u8_444_ayuv(uint8_t* dsty, int ystride, uint8_t* dstu, int ustride, uint8_t* dstv, int vstride, int dwidth, int dheight, uint8_t* _src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_u8_444_ayuv<<<grid_size, block_size, shared_size>>>(dsty, ystride, dstu, ustride, dstv, vstride, dwidth, _src, sstride, swidth, sheight);
}
void cuda_convert_yuyv_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_yuyv_u8_422<<<grid_size, block_size, shared_size>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_convert_uyvy_u8_422 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    dwidth >>= 1;  swidth >>= 1;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_uyvy_u8_422<<<grid_size, block_size, shared_size>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_convert_ayuv_u8_444 (uint8_t* _dst, int dstride, int dwidth, int dheight, uint8_t* srcy, int ystride, uint8_t* srcu, int ustride, uint8_t* srcv, int vstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    convert_ayuv_u8_444<<<grid_size, block_size, shared_size>>>(_dst, dstride, dwidth, srcy, ystride, srcu, ustride, srcv, vstride, swidth, sheight);
}
void cuda_subtract_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    subtract_s16_u8<<<grid_size, block_size, shared_size>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}
void cuda_subtract_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    subtract_s16_s16<<<grid_size, block_size, shared_size>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}
void cuda_add_s16_u8(int16_t* dst, int dstride, int dwidth, int dheight, uint8_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    add_s16_u8<<<grid_size, block_size, shared_size>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}

void cuda_add_s16_s16(int16_t* dst, int dstride, int dwidth, int dheight, int16_t* src, int sstride, int swidth, int sheight)
{
    dim3 block_size, grid_size;
    int shared_size;

    block_size.x = THREADS;
    block_size.y = block_size.z = 1;
    grid_size.x = std::min(dheight, sheight);
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    add_s16_s16<<<grid_size, block_size, shared_size>>>(dst, dstride, src, sstride, std::min(swidth, dwidth));
}

}

#if 0

//#define TEST_HORIZONTAL
#define TEST_RANDOM
#define VERBOSE
#define SRCTYPE uint8_t
#define DSTTYPE uint8_t

int main( int argc, char** argv) 
{
    cutPrintInfo();

    // HDTV 1920x1080
#ifdef BIGSET
    int width = 1920;
    int height = 1080;
#else
    int swidth = 20;
    int sheight = 20;
    int dwidth = 20;
    int dheight = 20;

    int sstride = swidth*sizeof(SRCTYPE);
    int dstride = dwidth*sizeof(DSTTYPE);
#endif

    unsigned int ssize = sstride*sheight;
    unsigned int dsize = dstride*dheight;

    /* Data on CPU */
    SRCTYPE *data = (SRCTYPE*)malloc(ssize);
    DSTTYPE *ddata = (DSTTYPE*)malloc(dsize);

#ifdef SEED_FIXED
    srand(20);
#else
    srand(time(NULL));
#endif
    for(int y=0; y<sheight; ++y)
        for(int x=0; x<swidth; ++x)
#ifdef TEST_RANDOM
            data[y*swidth+x] = (rand()&0xFF);
#endif
#ifdef TEST_HORIZONTAL
            data[y*swidth+x] = x;
#endif
#ifdef TEST_VERTICAL
            data[y*swidth+x] = y;
#endif
#ifdef TEST_HORIZONTALQ
            data[y*swidth+x] = (x*x)%256;
#endif
#ifdef TEST_VERTICALQ
            data[y*swidth+x] = (y*y)%256;
#endif
#ifdef VERBOSE
    printf("Orig:\n");
    for(int y=0; y<sheight; ++y)
    {
        for(int x=0; x<swidth; ++x)
            printf("%4i ", (int)data[y*swidth+x]);
        printf("\n");
    }
#endif

    /** Data on GPU */
    SRCTYPE *d_data_src = NULL;
    DSTTYPE *d_data_dst = NULL;
    DSTTYPE *d_data_dst2 = NULL;
    DSTTYPE *d_data_dst3 = NULL;
    DSTTYPE *d_data_dst4 = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_src, ssize));
    cudaMemset(d_data_src, 0, ssize);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_dst, dsize));
    cudaMemset(d_data_dst, 0, ssize);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_dst2, dsize));
    cudaMemset(d_data_dst2, 0, ssize);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_dst3, dsize));
    cudaMemset(d_data_dst3, 0, ssize);
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data_dst4, dsize));
    cudaMemset(d_data_dst4, 0, ssize);

    /** Copy to GPU */
    CUDA_SAFE_CALL(cudaMemcpy(d_data_src, data, ssize, cudaMemcpyHostToDevice));

    /** Invoke kernel */
    dim3 block_size;
    dim3 grid_size;
    int shared_size;

    block_size.x = 256;
    block_size.y = block_size.z = 1;
    grid_size.x = dheight;
    grid_size.y = grid_size.z = 1;
    shared_size = 0;

    //convert_u8_s16<<<grid_size, block_size, shared_size>>>(d_data_dst, dwidth, dstride, d_data_src, swidth, sheight, sstride);

    //convert_u8_422_yuyv<<<grid_size, block_size, shared_size>>>(d_data_dst, d_data_dst2, d_data_dst3, dwidth/2, dstride, dstride, d_data_src, swidth/4, sheight, sstride/4);
    //convert_u8_422_uyvy<<<grid_size, block_size, shared_size>>>(d_data_dst, d_data_dst2, d_data_dst3, dwidth/2, dstride, dstride, d_data_src, swidth/4, sheight, sstride/4); 
    // uint8_t* _dst, int dwidth, int dstride, uint8_t* srcy, uint8_t* srcu, uint8_t* srcv, int swidth, int sheight, int ystride, int uvstride
    //convert_uyvy_u8_422<<<grid_size, block_size, shared_size>>>(d_data_dst4, dwidth/4, dstride/4, d_data_dst, d_data_dst2, d_data_dst3, dwidth/2, dheight, dstride, dstride);

    convert_u8_444_ayuv<<<grid_size, block_size, shared_size>>>(d_data_dst, dstride, d_data_dst2, dstride, d_data_dst3, dstride, dwidth, d_data_src, sstride, swidth/4, sheight); 
    convert_ayuv_u8_444<<<grid_size, block_size, shared_size>>>(d_data_dst4, dstride, dwidth/4, d_data_dst, dstride, d_data_dst2, dstride, d_data_dst3, dstride, dwidth, dheight);

    /** Copy back */
    CUDA_SAFE_CALL(cudaMemcpy(ddata, d_data_dst, dsize, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
    printf("1:\n");
    for(int y=0; y<dheight; ++y)
    {
        for(int x=0; x<dwidth; ++x)
        {
            printf("%4i ", (int)ddata[y*dwidth+x]);
        }
        printf("\n");
    }
#endif
    printf("\n");

    CUDA_SAFE_CALL(cudaMemcpy(ddata, d_data_dst2, dsize, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
    printf("2:\n");
    for(int y=0; y<dheight; ++y)
    {
        for(int x=0; x<dwidth; ++x)
        {
            printf("%4i ", (int)ddata[y*dwidth+x]);
        }
        printf("\n");
    }
#endif
    printf("\n");

    CUDA_SAFE_CALL(cudaMemcpy(ddata, d_data_dst3, dsize, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
    printf("3:\n");
    for(int y=0; y<dheight; ++y)
    {
        for(int x=0; x<dwidth; ++x)
        {
            printf("%4i ", (int)ddata[y*dwidth+x]);
        }
        printf("\n");
    }
#endif
    printf("\n");

    CUDA_SAFE_CALL(cudaMemcpy(ddata, d_data_dst4, dsize, cudaMemcpyDeviceToHost));
#ifdef VERBOSE
    printf("4:\n");
    for(int y=0; y<dheight; ++y)
    {
        for(int x=0; x<dwidth; ++x)
        {
            printf("%4i ", (int)ddata[y*dwidth+x]);
        }
        printf("\n");
    }
#endif
    printf("\n");


    free(data);
    cudaFree(d_data_src);
    cudaFree(d_data_dst);
    cudaFree(d_data_dst2);
    cudaFree(d_data_dst3);
}

#endif
