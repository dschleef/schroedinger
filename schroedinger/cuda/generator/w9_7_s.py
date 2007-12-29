from wl import *

s_params["9_7"]="""
/// Boundaries (depends on wavelet)
/// This much is reserved at the sides of the signal
/// Must be even!
#define BLEFT 2
#define BRIGHT 2

/// Initial shift (to keep precision in integer wavelets)
#define INITIAL_SHIFT 1
#define INITIAL_OFFSET 1

#define STAGE1_OFFSET 2047
#define STAGE1_SHIFT 12
#define STAGE1_COEFF (-6497)

#define STAGE2_OFFSET 2047
#define STAGE2_SHIFT 12
#define STAGE2_COEFF (-217)

#define STAGE3_OFFSET 2048
#define STAGE3_SHIFT 12
#define STAGE3_COEFF (3616)

#define STAGE4_OFFSET 2048
#define STAGE4_SHIFT 12
#define STAGE4_COEFF (1817)

/// Vertical pass row management
#define RLEFT 1
#define RRIGHT 0
#define COPYROWS 3
"""

s_transform_h["9_7"] = """
    const int end = BLEFT+(width>>1);

    shared[half+BLEFT-1] = shared[half+BLEFT];

    __syncthreads();

    // Process even
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE4_OFFSET;

        acc += __mul24(STAGE4_COEFF,shared[half+ofs-1]);
        acc += __mul24(STAGE4_COEFF,shared[half+ofs+0]);
        
        shared[ofs] -= acc >> STAGE4_SHIFT;
    }

    // hi[n] = hi[n-1]
    if(ofs == (end-1+BSH))
    {
        shared[BLEFT+(width>>1)] = shared[BLEFT+(width>>1)-1];
    }

    __syncthreads();

    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE3_OFFSET;

        acc += __mul24(STAGE3_COEFF, shared[ofs+0]);
        acc += __mul24(STAGE3_COEFF, shared[ofs+1]);
        
        shared[ofs + half] -= acc >> STAGE3_SHIFT;
    }

    // lo[-1] = hi[0]
    if(tidu16 == 0)
    {
        shared[half+BLEFT-1] = shared[half+BLEFT];
    }

    __syncthreads();

    // Process even
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF,shared[half+ofs-1]);
        acc += __mul24(STAGE2_COEFF,shared[half+ofs+0]);
        
        shared[ofs] -= acc >> STAGE2_SHIFT;
    }

    // hi[n] = hi[n-1]
    if(ofs == (end-1+BSH))
    {
        shared[BLEFT+(width>>1)] = shared[BLEFT+(width>>1)-1];
    }

    __syncthreads();
    
    // Process odd
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF, shared[ofs+0]);
        acc += __mul24(STAGE1_COEFF, shared[ofs+1]);
        
        shared[ofs + half] -= acc >> STAGE1_SHIFT;
    }

    __syncthreads();

"""

s_transform_v["9_7"] = """
__device__ void doTransform(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row

    extern __shared__ DATATYPE shared[];
    int ofs;

    ofs = ((RLEFT+(tidy<<1)+3)<<BCOLS_SHIFT) + tidx;

    /* Phase 4 at +3*BCOLS */    
    {
        int acc = STAGE4_OFFSET;

        acc += __mul24(STAGE4_COEFF,shared[ofs-BCOLS]);
        acc += __mul24(STAGE4_COEFF,shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE4_SHIFT;
    }

    __syncthreads();

    /* Phase 3 at +2*BCOLS */    
    ofs -= BCOLS;
    {
        int acc = STAGE3_OFFSET;

        acc += __mul24(STAGE3_COEFF,shared[ofs-BCOLS]);
        acc += __mul24(STAGE3_COEFF,shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE3_SHIFT;
    }

    __syncthreads();

    /* Phase 2 at +1*BCOLS */    
    ofs -= BCOLS;
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF,shared[ofs-BCOLS]);
        acc += __mul24(STAGE2_COEFF,shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE2_SHIFT;
    }

    __syncthreads();

    /* Phase 1 at +0*BCOLS */
    ofs -= BCOLS;
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF, shared[ofs-BCOLS]);
        acc += __mul24(STAGE1_COEFF, shared[ofs+BCOLS]);

        shared[ofs] -= acc >> STAGE1_SHIFT;
    }        
}

__device__ void doTransformTB(int xofs, unsigned int leftover)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const int minn = (RLEFT<<BCOLS_SHIFT) + tidx;
    const int maxx = leftover-(2<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    int ofs;
    ofs = ((RLEFT+(tidy<<1))<<BCOLS_SHIFT) + tidx;

    /* Phase 4 (even) */
    {
        int acc = STAGE4_OFFSET;

        acc += __mul24(STAGE4_COEFF,shared[max(ofs-BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE4_COEFF,shared[ofs+BCOLS]);

        shared[ofs] -= acc >> STAGE4_SHIFT;
    }
    __syncthreads();
    
    /* Phase 3 (odd) */
    ofs += BCOLS;
    {
        int acc = STAGE3_OFFSET;

        acc += __mul24(STAGE3_COEFF, shared[ofs-BCOLS]);
        acc += __mul24(STAGE3_COEFF, shared[min(ofs+BCOLS,maxx)]);

        shared[ofs] -= acc >> STAGE3_SHIFT;
    }

    __syncthreads();
    
    /* Phase 2 (even) */
    ofs -= BCOLS;
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF,shared[max(ofs-BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE2_COEFF,shared[ofs+BCOLS]);

        shared[ofs] -= acc >> STAGE2_SHIFT;
    }

    __syncthreads();
    
    /* Phase 1 (odd) */
    ofs += BCOLS;
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF, shared[ofs-BCOLS]);
        acc += __mul24(STAGE1_COEFF, shared[min(ofs+BCOLS,maxx)]);

        shared[ofs] -= acc >> STAGE1_SHIFT;
    }
    
}

__device__ void doTransformT(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const int minn = ((RLEFT+SKIPTOP)<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    int ofs;
    ofs = ((SKIPTOP+RLEFT+(tidy<<1))<<BCOLS_SHIFT) + tidx;

    /* Phase 4 (even), offset +0 */
    {
        int acc = STAGE4_OFFSET;

        acc += __mul24(STAGE4_COEFF,shared[max(ofs-BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE4_COEFF,shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE4_SHIFT;
    }

    __syncthreads();

    /* Phase 3 (odd), offset +1 */
    ofs += BCOLS;
    if(tidy != (BSVY-1))
    {
        int acc = STAGE3_OFFSET;

        acc += __mul24(STAGE3_COEFF, shared[ofs-BCOLS]);
        acc += __mul24(STAGE3_COEFF, shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE3_SHIFT;
    }
    
    __syncthreads();

    /* Phase 2 (even), offset +0 */
    ofs -= BCOLS;
    if(tidy != (BSVY-1))
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF,shared[max(ofs-BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE2_COEFF,shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE2_SHIFT;
    }
    
    __syncthreads();

    /* Phase 1 (odd), offset +1 */
    ofs += BCOLS;
    if(tidy < (BSVY-2))
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF, shared[ofs-BCOLS]);
        acc += __mul24(STAGE1_COEFF, shared[ofs+BCOLS]);
        
        shared[ofs] -= acc >> STAGE1_SHIFT;
    }
}

__device__ void doTransformB(int xofs, unsigned int leftover)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const int maxx = leftover-(2<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    int ofs, ofs_t;

    ofs = ((RLEFT+(tidy<<1))<<BCOLS_SHIFT) + tidx;

    for(ofs_t=ofs+3*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE4_OFFSET;

        acc += __mul24(STAGE4_COEFF,shared[ofs_t-BCOLS]);
        acc += __mul24(STAGE4_COEFF,shared[ofs_t+BCOLS]);
        
        shared[ofs_t] -= acc >> STAGE4_SHIFT;
    }

    __syncthreads();
    
    for(ofs_t=ofs+2*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE3_OFFSET;

        acc += __mul24(STAGE3_COEFF,shared[ofs_t-BCOLS]);
        acc += __mul24(STAGE3_COEFF,shared[min(ofs_t+BCOLS,maxx)]);
        
        shared[ofs_t] -= acc >> STAGE3_SHIFT;
    }

    __syncthreads();
    
    for(ofs_t=ofs+1*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF,shared[ofs_t-BCOLS]);
        acc += __mul24(STAGE2_COEFF,shared[ofs_t+BCOLS]);
        
        shared[ofs_t] -= acc >> STAGE2_SHIFT;
    }

    __syncthreads();

    for(ofs_t=ofs+0*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF, shared[ofs_t-BCOLS]);
        acc += __mul24(STAGE1_COEFF, shared[min(ofs_t+BCOLS,maxx)]);
        
        shared[ofs_t] -= acc >> STAGE1_SHIFT;
    }
}

"""

