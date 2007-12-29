from wl import *
a_params["5_3"]="""
/// This much is reserved at the sides of the signal
#define BLEFT 2
#define BRIGHT 2

/// Initial shift (to keep precision in integer wavelets)
#define INITIAL_OFFSET 1
#define INITIAL_SHIFT 1

/// Shifts for lifting steps
#define STAGE1_OFFSET 0
#define STAGE1_SHIFT 1

#define STAGE2_OFFSET 2
#define STAGE2_SHIFT 2

/// Vertical pass row management
#define RLEFT 1
#define RRIGHT 1
#define COPYROWS 0
"""

a_transform_h["5_3"] = """
    // Duplicate right boundary
    if(tidu16==0)
        shared[BLEFT+(width>>1)] = shared[BLEFT+(width>>1)-1];

    __syncthreads();
    
    // Now apply wavelet lifting to entire line at once
    // Process odd
    const int end = BLEFT+(width>>1);
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        /// Accumulate value
        int acc = STAGE1_OFFSET;

        acc += -1*shared[ofs];
        acc += -1*shared[ofs+1];
        
        shared[ofs + half] += acc >> STAGE1_SHIFT;
    }

    // Duplicate left boundary
    // Do this in thread that wrote shared[half+BLEFT] (tid 0)
    if(tidu16 == 0)
        shared[half+BLEFT-1] = shared[half+BLEFT];

    __syncthreads();

    // Process even
    for(ofs = BLEFT+tidu16; ofs < end; ofs += BSH)
    {
        /// Accumulate value
        int acc = STAGE2_OFFSET;

        acc += 1*shared[half+ofs-1];
        acc += 1*shared[half+ofs];
        
        shared[ofs] += acc >> STAGE2_SHIFT;
    }

    __syncthreads();
"""

a_transform_v["5_3"] = """
__device__ void doTransform(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row

    extern __shared__ DATATYPE shared[];
    unsigned int ofs;

    /// Do the transform
    
    /// Do procesing on shared mem
    /// Treat all columns the same
    /// Process odd rows
    /// shared[ofs-1*BCOLS]
    /// shared[ofs+1*BCOLS]
 
    ofs = ((RLEFT+(tidy<<1)+1)<<BCOLS_SHIFT) + tidx;
    //onst int end = RLEFT+(height>>1);
    //for(ofs = RLEFT+tid; ofs < end; ofs += BSVY)
    {
        /// Accumulate value
        int acc = STAGE1_OFFSET;

        acc += -1*shared[ofs-BCOLS];
        acc += -1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE1_SHIFT;
    }
    
    __syncthreads();
    
    /// Process even rows - we can process one less now at the bottom
    /// shared[ofs-1*BCOLS]
    /// shared[ofs+1*BCOLS]
    ofs -= BCOLS;
    {
        /// Accumulate value
        int acc = STAGE2_OFFSET;

        acc += 1*shared[ofs-BCOLS];
        acc += 1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE2_SHIFT;
    }

}

__device__ void doTransformTB(int xofs, unsigned int leftover)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const unsigned int minn = (RLEFT<<BCOLS_SHIFT) + tidx;
    const unsigned int maxx = leftover-(2<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    unsigned int ofs;

    /// Do the transform
    
    /// Do procesing on shared mem
    /// Treat all columns the same
    /// Process odd rows
    ofs = ((RLEFT+(tidy<<1)+1)<<BCOLS_SHIFT) + tidx;
    {
        /// Accumulate value
        int acc = STAGE1_OFFSET;

        acc += -1*shared[ofs-BCOLS];
        acc += -1*shared[min(ofs+BCOLS, maxx)];
        
        shared[ofs] += acc >> STAGE1_SHIFT;
    }
    
    __syncthreads();
    
    /// Process even rows - we can process one less now at the bottom
    ofs -= BCOLS;
    {
        /// Accumulate value
        int acc = STAGE2_OFFSET;

        acc += 1*shared[max(ofs-BCOLS, minn+BCOLS)];
        acc += 1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE2_SHIFT;
    }
}

__device__ void doTransformT(int xofs)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    const unsigned int minn = (RLEFT<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    unsigned int ofs;

    /// Do the transform
    
    /// Do procesing on shared mem
    /// Treat all columns the same
    /// Process odd rows
    ofs = ((RLEFT+(tidy<<1)+1)<<BCOLS_SHIFT) + tidx;
    {
        /// Accumulate value
        int acc = STAGE1_OFFSET;

        acc += -1*shared[ofs-BCOLS];
        acc += -1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE1_SHIFT;
    }
    
    __syncthreads();
    
    /// Process even rows - we can process one less now at the bottom
    ofs -= BCOLS;
    {
        /// Accumulate value
        int acc = STAGE2_OFFSET;

        acc += 1*shared[max(ofs-BCOLS, minn+BCOLS)];
        acc += 1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE2_SHIFT;
    }
}

__device__ void doTransformB(int xofs, unsigned int leftover)
{
    const int tidx = (threadIdx.x<<1)+xofs;   // column
    const int tidy = threadIdx.y;   // row
    //const unsigned int minn = RLEFT*BCOLS + threadIdx.x;
    const unsigned int maxx = leftover-(2<<BCOLS_SHIFT) + tidx;
    
    extern __shared__ DATATYPE shared[];
    unsigned int ofs;

    /// Do the transform
    
    /// Do procesing on shared mem
    /// Treat all columns the same
    /// Process odd rows
    ofs = ((RLEFT+(tidy<<1)+1)<<BCOLS_SHIFT) + tidx;
    {
        /// Accumulate value
        int acc = STAGE1_OFFSET;

        acc += -1*shared[ofs-BCOLS];
        acc += -1*shared[min(ofs+BCOLS, maxx)];
        
        shared[ofs] += acc >> STAGE1_SHIFT;
    }
    
    __syncthreads();
    
    /// Process even rows - we can process one less now at the bottom
    ofs -= BCOLS;
    {
        /// Accumulate value
        int acc = STAGE2_OFFSET;

        acc += 1*shared[ofs-BCOLS];
        acc += 1*shared[ofs+BCOLS];
        
        shared[ofs] += acc >> STAGE2_SHIFT;
    }
}
"""
