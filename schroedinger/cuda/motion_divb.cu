/** 
Overlapped block motion compensation for CUDA

Efficient implementation that divides up the image into regions based on the 
amount of blocks that overlap it, which is 1 (in the middle), 2 (horizontal 
or vertical overlap) or 4 (diagonal overlap). 

By processing these regions in different cuda blocks, any divergence between
threads is prevented.

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
//#include <cutil.h>
#include <cuda.h>

#include <algorithm>
#include <cassert>

#define SPECIAL

#define MOTION_MAX_BLOCKS 8160
//#define USING_GLOBAL

#define THREADSX_LOG2 4
#define THREADSY_LOG2 4

#ifdef SPECIAL
#define WIDTHX_LOG2 (THREADSX_LOG2+1)
#define WIDTHY_LOG2 THREADSY_LOG2
#else
#define WIDTHX_LOG2 THREADSX_LOG2
#define WIDTHY_LOG2 THREADSY_LOG2
#endif

#define THREADSX (1<<THREADSX_LOG2)
#define THREADSY (1<<THREADSY_LOG2)

#define WIDTHX (1<<WIDTHX_LOG2)
#define WIDTHY (1<<WIDTHY_LOG2)

#include "common.h"
#include "cudamotion.h"

#include "motion_kernel_divb.h"

#if 0
_MotionVector *motion_vectors = 0;
//uint8_t *motion_flags = 0;
#endif

static inline int div_roundup(int x, int y)
{
    return (x+y-1)/y;
}

extern "C" {

void cuda_motion_initialize(CudaMotionData *d)
{
#if 0
    /// XXX hacky initialisation
    if(!motion_vectors)
    {
        cudaMalloc((void**)&motion_vectors, sizeof(struct _MotionVector)*MOTION_MAX_BLOCKS);
        //cudaMalloc((void**)&motion_flags, MOTION_MAX_BLOCKS);
    }
#endif
    /// Upload constants
    int nblocks = d->obmc.blocksx * d->obmc.blocksy;

    cudaMemcpyToSymbol(obmc, &d->obmc, sizeof(obmc));

    //printf("%p %p\n", &motion_vectors, d->vectors);

    assert(nblocks <= MOTION_MAX_BLOCKS);
    cudaMemcpyToSymbol(motion_vectors, d->vectors, sizeof(struct _MotionVector)*nblocks);
    //cudaMemcpyToSymbol(motion_flags, d->flags, nblocks);

    //cudaMemcpy(motion_vectors, d->vectors, sizeof(struct _MotionVector)*nblocks, cudaMemcpyHostToDevice);
    //cudaMemcpy(motion_flags,   d->flags, nblocks, cudaMemcpyHostToDevice);

    ref1.addressMode[0] = cudaAddressModeClamp;
    ref1.addressMode[1] = cudaAddressModeClamp;
    ref1.filterMode = cudaFilterModeLinear;
    ref1.normalized = false;

    ref2.addressMode[0] = cudaAddressModeClamp;
    ref2.addressMode[1] = cudaAddressModeClamp;
    ref2.filterMode = cudaFilterModeLinear;
    ref2.normalized = false;
}

void cuda_motion_copy(CudaMotionData *d, uint16_t *output, int ostride, int width, int height, int component, int xshift, int yshift, struct cudaArray *aref1, struct cudaArray *aref2)
{
/*
    /// Set component data

    _Comp c;
    c.comp = component;
    c.x_shift = xshift;
    c.y_shift = yshift;
    cudaMemcpyToSymbol(comp, &c, sizeof(struct _Comp));
*/

    /// Bind textures
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaBindTextureToArray(ref1, aref1, channelDesc);
    if(aref2)
        cudaBindTextureToArray(ref2, aref2, channelDesc);

    /// Execute the kernel
    dim3 block_size, grid_size;
    int shared_size;

    /// Determine execution parameters
    block_size.x = THREADSX;
    block_size.y = THREADSY;
    block_size.z = 1;

    int blocksX = div_roundup(width,  d->obmc.x_sep>>xshift);
    int blocksY = div_roundup(height, d->obmc.y_sep>>yshift);
    int xB = div_roundup(blocksX * (d->obmc.x_mid>>xshift), WIDTHX);
    int yB = div_roundup(blocksY * (d->obmc.y_mid>>yshift), WIDTHY);
    int xC = div_roundup(blocksX * (d->obmc.x_ramp>>xshift), WIDTHX);
    int yC = div_roundup(blocksY * (d->obmc.y_ramp>>yshift), WIDTHY);
/*
    printf("%i %i : %i %i %i %i %i %i : %i %i %i %i %i %i : %i %i\n", width, height, 
                                  d->obmc.x_sep>>xshift, d->obmc.y_sep>>yshift, d->obmc.x_mid>>xshift, d->obmc.y_mid>>yshift, d->obmc.x_ramp>>xshift, d->obmc.y_ramp>>yshift,
                                  blocksX, blocksY, xB, yB, xC, yC,
                                  xB + xC, yB + yB);
*/
    grid_size.x = xB + xC;
    grid_size.y = yB + yB;

    //grid_size.x = div_roundup((max(d->obmc.x_mid, d->obmc.x_ramp)>>xshift)*div_roundup(width, d->obmc.x_sep>>xshift), WIDTHX);
    //grid_size.y = div_roundup((max(d->obmc.y_mid, d->obmc.y_ramp)>>yshift)*div_roundup(height, d->obmc.y_sep>>yshift), WIDTHY);
    grid_size.z = 1;
    shared_size = 0;

    /// Vector scaling constants for this component
    float sxscale = exp2f(-xshift) * 0.25f;
    float syscale = exp2f(-yshift) * 0.25f;
    
/*
    printf("%ix%i comp %i grid %ix%i scale %fx%f ramp %i %i sep %i %i mid %i %i\n", 
        width, height,
        component, grid_size.x, grid_size.y, sxscale, syscale,
        d->obmc.x_ramp_log2 - xshift, d->obmc.y_ramp_log2 - yshift, 
        d->obmc.x_sep_log2 - xshift,  d->obmc.y_sep_log2 - yshift, 
        d->obmc.x_mid_log2 - xshift,  d->obmc.y_mid_log2 - yshift
    );
*/
    motion_copy_2ref<<<grid_size, block_size, shared_size>>>(
        output, ostride, width, height, xB, yB,
        component*8, sxscale, syscale, 
        d->obmc.x_ramp_log2 - xshift, d->obmc.y_ramp_log2 - yshift, 
        d->obmc.x_sep_log2 - xshift,  d->obmc.y_sep_log2 - yshift, 
        d->obmc.x_mid_log2 - xshift,  d->obmc.y_mid_log2 - yshift
        );
    /// XXX unbind textures?
}


}
