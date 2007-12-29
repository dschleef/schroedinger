from wl import *

s_params["fidelity"]="""
/// Boundaries (depends on wavelet)
/// This much is reserved at the sides of the signal
/// Must be even!
#define BLEFT 4
#define BRIGHT 4

/// Initial shift (to keep precision in integer wavelets)
#define INITIAL_SHIFT 0
#define INITIAL_OFFSET 0

//  static const int16_t stage1_weights[] = { -8, 21, -46, 161, 161, -46, 21, -8 };
//  static const int16_t stage2_weights[] = { 2, -10, 25, -81, -81, 25, -10, 2 };

#define STAGE1_OFFSET 128
#define STAGE1_SHIFT 8
#define STAGE1_COEFF0 (-8)
#define STAGE1_COEFF1 (21)
#define STAGE1_COEFF2 (-46)
#define STAGE1_COEFF3 (161)
#define STAGE1_COEFF4 (161)
#define STAGE1_COEFF5 (-46)
#define STAGE1_COEFF6 (21)
#define STAGE1_COEFF7 (-8)

#define STAGE2_OFFSET 127
#define STAGE2_SHIFT 8
#define STAGE2_COEFF0 (2)
#define STAGE2_COEFF1 (-10)
#define STAGE2_COEFF2 (25)
#define STAGE2_COEFF3 (-81)
#define STAGE2_COEFF4 (-81)
#define STAGE2_COEFF5 (25)
#define STAGE2_COEFF6 (-10)
#define STAGE2_COEFF7 (2)

/// Vertical pass row management
#define RLEFT 8
#define RRIGHT 7
#define COPYROWS 7
"""

s_transform_h["fidelity"] = """
    const int end = BLEFT+(width>>1);

    if(tidu16<4)
    {
        /*
          hi[-3] = hi[0];
          hi[-2] = hi[0];
          hi[-1] = hi[0];
          hi[n] = hi[n-1];
          hi[n+1] = hi[n-1];
          hi[n+2] = hi[n-1];
          hi[n+3] = hi[n-1];
        */
        shared[BLEFT-4+tidu16] = shared[BLEFT];
        shared[BLEFT+(width>>1)+tidu16] = shared[BLEFT+(width>>1)-1];
    }
    __syncthreads();

    // Process odd
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF0, shared[ofs-3]);
        acc += __mul24(STAGE2_COEFF1, shared[ofs-2]);
        acc += __mul24(STAGE2_COEFF2, shared[ofs-1]);
        acc += __mul24(STAGE2_COEFF3, shared[ofs-0]);
        acc += __mul24(STAGE2_COEFF4, shared[ofs+1]);
        acc += __mul24(STAGE2_COEFF5, shared[ofs+2]);
        acc += __mul24(STAGE2_COEFF6, shared[ofs+3]);
        acc += __mul24(STAGE2_COEFF7, shared[ofs+4]);
        
        shared[ofs + half] -= acc >> STAGE2_SHIFT;
    }
    
    __syncthreads();

    if(tidu16<4)
    {
        /*
          lo[-4] = lo[0];
          lo[-3] = lo[0];
          lo[-2] = lo[0];
          lo[-1] = lo[0];
          lo[n] = lo[n-1];
          lo[n+1] = lo[n-1];
          lo[n+2] = lo[n-1];
        */
        shared[half+BLEFT-4+tidu16] = shared[half+BLEFT];
        shared[half+BLEFT+(width>>1)+tidu16] = shared[half+BLEFT+(width>>1)-1];
    }    
    __syncthreads();
    
    // Process even

    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF0,shared[half+ofs-4]);
        acc += __mul24(STAGE1_COEFF1,shared[half+ofs-3]);
        acc += __mul24(STAGE1_COEFF2,shared[half+ofs-2]);
        acc += __mul24(STAGE1_COEFF3,shared[half+ofs-1]);
        acc += __mul24(STAGE1_COEFF4,shared[half+ofs+0]);
        acc += __mul24(STAGE1_COEFF5,shared[half+ofs+1]);
        acc += __mul24(STAGE1_COEFF6,shared[half+ofs+2]);
        acc += __mul24(STAGE1_COEFF7,shared[half+ofs+3]);
        
        shared[ofs] -= acc >> STAGE1_SHIFT;
    }

    __syncthreads();
"""

s_transform_v["fidelity"] = """
__device__ void doTransform(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row

    extern __shared__ DATATYPE shared[];
    int ofs;

    ofs = ((RLEFT+(tidy<<1)+8)<<BCOLS_SHIFT) + tidx;

    /* Phase 2 (odd) at +1*BCOLS */    
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF0, shared[ofs-7*BCOLS]);
        acc += __mul24(STAGE2_COEFF1, shared[ofs-5*BCOLS]);
        acc += __mul24(STAGE2_COEFF2, shared[ofs-3*BCOLS]);
        acc += __mul24(STAGE2_COEFF3, shared[ofs-1*BCOLS]);
        acc += __mul24(STAGE2_COEFF4, shared[ofs+1*BCOLS]);
        acc += __mul24(STAGE2_COEFF5, shared[ofs+3*BCOLS]);
        acc += __mul24(STAGE2_COEFF6, shared[ofs+5*BCOLS]);
        acc += __mul24(STAGE2_COEFF7, shared[ofs+7*BCOLS]);
        
        shared[ofs] -= acc >> STAGE2_SHIFT;
    }
    
    __syncthreads();

    /* Phase 1 (even) at +8*BCOLS */
    ofs -= 7*BCOLS;
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF0, shared[ofs-7*BCOLS]);
        acc += __mul24(STAGE1_COEFF1, shared[ofs-5*BCOLS]);        
        acc += __mul24(STAGE1_COEFF2, shared[ofs-3*BCOLS]);
        acc += __mul24(STAGE1_COEFF3, shared[ofs-1*BCOLS]);
        acc += __mul24(STAGE1_COEFF4, shared[ofs+1*BCOLS]);
        acc += __mul24(STAGE1_COEFF5, shared[ofs+3*BCOLS]);
        acc += __mul24(STAGE1_COEFF6, shared[ofs+5*BCOLS]);
        acc += __mul24(STAGE1_COEFF7, shared[ofs+7*BCOLS]);

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
    int ofs, ofs_t;

    
    ofs = ((RLEFT+(tidy<<1))<<BCOLS_SHIFT) + tidx;

    /* Phase 2 (odd) */
    for(ofs_t=ofs+1*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF0, shared[max(ofs_t-7*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF1, shared[max(ofs_t-5*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF2, shared[max(ofs_t-3*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF3, shared[ofs_t-1*BCOLS]);
        acc += __mul24(STAGE2_COEFF4, shared[min(ofs_t+1*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF5, shared[min(ofs_t+3*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF6, shared[min(ofs_t+5*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF7, shared[min(ofs_t+7*BCOLS, maxx)]);

        shared[ofs_t] -= acc >> STAGE2_SHIFT;
    }
    __syncthreads();
    
    /* Phase 1 (even) */
    for(ofs_t=ofs+0*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF0, shared[max(ofs_t-7*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF1, shared[max(ofs_t-5*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF2, shared[max(ofs_t-3*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF3, shared[max(ofs_t-1*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF4, shared[ofs_t+1*BCOLS]);
        acc += __mul24(STAGE1_COEFF5, shared[min(ofs_t+3*BCOLS, maxx+BCOLS)]);
        acc += __mul24(STAGE1_COEFF6, shared[min(ofs_t+5*BCOLS, maxx+BCOLS)]);
        acc += __mul24(STAGE1_COEFF7, shared[min(ofs_t+7*BCOLS, maxx+BCOLS)]);

        shared[ofs_t] -= acc >> STAGE1_SHIFT;
    }
    
    
}

__device__ void doTransformT(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const int minn = ((RLEFT+SKIPTOP)<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    int ofs;
    
    ofs = ((SKIPTOP+RLEFT+(tidy<<1)+1)<<BCOLS_SHIFT) + tidx;

    /* Phase 2 (odd), offset +1 */
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF0, shared[max(ofs-7*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF1, shared[max(ofs-5*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF2, shared[max(ofs-3*BCOLS, minn)]);
        acc += __mul24(STAGE2_COEFF3, shared[ofs-1*BCOLS]);
        acc += __mul24(STAGE2_COEFF4, shared[ofs+1*BCOLS]);
        acc += __mul24(STAGE2_COEFF5, shared[ofs+3*BCOLS]);
        acc += __mul24(STAGE2_COEFF6, shared[ofs+5*BCOLS]);
        acc += __mul24(STAGE2_COEFF7, shared[ofs+7*BCOLS]);
        
        shared[ofs] -= acc >> STAGE2_SHIFT;
    }
    
    __syncthreads();

    /* Phase 1 (even), offset +0 */
    ofs -= BCOLS;
    if(tidy<(BSVY-3))
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF0, shared[max(ofs-7*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF1, shared[max(ofs-5*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF2, shared[max(ofs-3*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF3, shared[max(ofs-1*BCOLS, minn+BCOLS)]);
        acc += __mul24(STAGE1_COEFF4, shared[ofs+1*BCOLS]);
        acc += __mul24(STAGE1_COEFF5, shared[ofs+3*BCOLS]);
        acc += __mul24(STAGE1_COEFF6, shared[ofs+5*BCOLS]);
        acc += __mul24(STAGE1_COEFF7, shared[ofs+7*BCOLS]);
        
        shared[ofs] -= acc >> STAGE1_SHIFT;
    }

    
}

// Finish off leftover
__device__ void doTransformB(int xofs, unsigned int leftover)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const int maxx = leftover-(2<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    int ofs, ofs_t;

    ofs = ((RLEFT+(tidy<<1))<<BCOLS_SHIFT) + tidx;
    
    for(ofs_t=ofs+8*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE2_OFFSET;

        acc += __mul24(STAGE2_COEFF0, shared[ofs_t-7*BCOLS]);
        acc += __mul24(STAGE2_COEFF1, shared[ofs_t-5*BCOLS]);
        acc += __mul24(STAGE2_COEFF2, shared[ofs_t-3*BCOLS]);
        acc += __mul24(STAGE2_COEFF3, shared[ofs_t-1*BCOLS]);
        acc += __mul24(STAGE2_COEFF4, shared[min(ofs_t+1*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF5, shared[min(ofs_t+3*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF6, shared[min(ofs_t+5*BCOLS, maxx)]);
        acc += __mul24(STAGE2_COEFF7, shared[min(ofs_t+7*BCOLS, maxx)]);        

        shared[ofs_t] -= acc >> STAGE2_SHIFT;
    }

    __syncthreads();

    for(ofs_t=ofs+1*BCOLS; ofs_t<leftover; ofs_t += BSVY*2*BCOLS)
    {
        int acc = STAGE1_OFFSET;

        acc += __mul24(STAGE1_COEFF0, shared[ofs_t-7*BCOLS]);
        acc += __mul24(STAGE1_COEFF1, shared[ofs_t-5*BCOLS]);
        acc += __mul24(STAGE1_COEFF2, shared[ofs_t-3*BCOLS]);
        acc += __mul24(STAGE1_COEFF3, shared[ofs_t-1*BCOLS]);
        acc += __mul24(STAGE1_COEFF4, shared[ofs_t+1*BCOLS]);
        acc += __mul24(STAGE1_COEFF5, shared[min(ofs_t+3*BCOLS, maxx+BCOLS)]);
        acc += __mul24(STAGE1_COEFF6, shared[min(ofs_t+5*BCOLS, maxx+BCOLS)]);
        acc += __mul24(STAGE1_COEFF7, shared[min(ofs_t+7*BCOLS, maxx+BCOLS)]);        

        shared[ofs_t] -= acc >> STAGE1_SHIFT;
    }
    
}
"""

