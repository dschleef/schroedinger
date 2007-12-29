param_v = """
#define BROWS (2*BSVY+COPYROWS) /* Rows to process at once */
#define SKIPTOP COPYROWS

#define PAD_ROWS (WRITEBACK-SKIPTOP+RRIGHT+COPYROWS) /* Rows below which to use s_transform_v_pad */
/// tid is BCOLSxBROWS matrix
/// RLEFT+BROWS+RRIGHT rows
#define TOTALROWS (RLEFT+BROWS+RRIGHT)
#define OVERLAP (RLEFT+RRIGHT+COPYROWS)
#define OVERLAP_OFFSET (TOTALROWS-OVERLAP)
#define WRITEBACK (2*BSVY)
"""

transform_v = """

static __global__ void %(dir)s_transform_v( DATATYPE* data, int width, int height, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const unsigned int bid = blockIdx.x;    // slab (BCOLS columns)
    const unsigned int tidx = threadIdx.x<<1;   // column
    const unsigned int tidy = threadIdx.y;   // row    
    const unsigned int swidth = min(width-(bid<<BCOLS_SHIFT), BCOLS); // Width of this slab, usually BCOLS but can be less

    // Element offset in global memory
    //int idata = tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    data += tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    
    const unsigned int istride = __mul24(BSVY, stride);
    const unsigned int sdata = tidx + (tidy<<BCOLS_SHIFT);
    // First read BROWS+RRIGHT
    // After that BROWS
    unsigned int ref = height-(WRITEBACK-SKIPTOP+RRIGHT+COPYROWS);
    unsigned int blocks = ref/WRITEBACK;
    unsigned int leftover = (RLEFT+COPYROWS+RRIGHT+(ref%%WRITEBACK))<<BCOLS_SHIFT;
    
    unsigned int gofs,sofs;

    /// More than one block
    /// Read first block of BROWS+RRIGHT rows
    /// Upper RLEFT rows are left unitialized for now, later they should be copied from top
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + ((RLEFT+SKIPTOP)<<BCOLS_SHIFT);
        for(; sofs < (TOTALROWS<<BCOLS_SHIFT); sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]);
    }
// idata_read = idata_write + __mul24(TOTALROWS-RLEFT, stride)

    __syncthreads();
    
    doTransformT(0);
    doTransformT(1);
    
    __syncthreads();

    /// Write back WRITEBACK rows
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + ((RLEFT+SKIPTOP)<<BCOLS_SHIFT);
        for(; sofs < ((WRITEBACK+RLEFT)<<BCOLS_SHIFT); sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]);
    }
// idata_read = idata_write + __mul24((BROWS+RRIGHT)-WRITEBACK, stride)

// Difference between global mem read and write pointer
#define DATA_READ_DIFF __mul24((BROWS+RRIGHT)-WRITEBACK, stride)
// Advance pointer with this amount after each block
#define DATA_INC __mul24(WRITEBACK, stride)

    data += __mul24(WRITEBACK-SKIPTOP, stride);
    for(unsigned int block=0; block<blocks; ++block)
    {
        __syncthreads();
        /// Move lower rows to top rows
#if OVERLAP <= BSVY 
        if(tidy < OVERLAP)
        {
            unsigned int l = (tidy<<BCOLS_SHIFT)+tidx;
            *((uint32_t*)&shared[l]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+l]);
        }
#else
        for(sofs = (tidy<<BCOLS_SHIFT)+tidx; sofs < (OVERLAP<<BCOLS_SHIFT); sofs += (BSVY<<BCOLS_SHIFT))
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+sofs]);
#endif                
        
        /// Fill shared memory -- read next block of BROWS rows
        /// We can skip RRIGHT rows as we've already copied them for the previous block
        /// and moved them to the top
        if(tidx < swidth)
        {
            gofs = DATA_READ_DIFF;
            sofs = sdata + (OVERLAP<<BCOLS_SHIFT);
            for(; sofs < (TOTALROWS<<BCOLS_SHIFT); sofs += (BCOLS*BSVY), gofs += istride)
                *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]);
        }

        __syncthreads();

        doTransform(0);
        doTransform(1);

        __syncthreads();

        /// Write back BROWS rows
        if(tidx < swidth)
        {
            gofs = 0;
            sofs = sdata + (RLEFT<<BCOLS_SHIFT);
            for(; sofs < ((WRITEBACK+RLEFT)<<BCOLS_SHIFT); sofs += (BCOLS*BSVY), gofs += istride)
                *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]);
        }
        
        data += DATA_INC;
    }
    __syncthreads();

    ///
    /// Handle partial last block
    /// Move lower rows to top rows
#if OVERLAP <= BSVY 
        if(tidy < OVERLAP)
        {
            unsigned int l = (tidy<<BCOLS_SHIFT)+tidx;
            *((uint32_t*)&shared[l]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+l]);
        }
#else
        for(sofs = (tidy<<BCOLS_SHIFT)+tidx; sofs < (OVERLAP<<BCOLS_SHIFT); sofs += (BSVY<<BCOLS_SHIFT))
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+sofs]);
#endif                
    
    /// Fill shared memory -- read next block of BROWS rows
    /// We can skip RRIGHT rows as we've already copied them for the previous block
    /// and moved them to the top
    if(tidx < swidth)
    {
        gofs = DATA_READ_DIFF;
        sofs = sdata + (OVERLAP<<BCOLS_SHIFT);
        for(; sofs < leftover; sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]);
    }

    __syncthreads();

    doTransformB(0, leftover);
    doTransformB(1, leftover);
    
    __syncthreads();
    
    /// Write back leftover
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + (RLEFT<<BCOLS_SHIFT);
        for(; sofs < leftover; sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]);
    }

}

/// Use this if the image is lower than PAD_ROWS
static __global__ void %(dir)s_transform_v_pad( DATATYPE* data, int width, int height, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const unsigned int bid = blockIdx.x;    // slab (BCOLS columns)
    const unsigned int tidx = threadIdx.x<<1;   // column
    const unsigned int tidy = threadIdx.y;   // row
    const unsigned int swidth = min(width-(bid<<BCOLS_SHIFT), BCOLS); // Width of this slab, usually BCOLS but can be less

    // Element offset in global memory
    //int idata = tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    data +=  tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    const unsigned int istride = __mul24(BSVY, stride);
    const unsigned int sdata = tidx + ((tidy+RLEFT)<<BCOLS_SHIFT); // Does this get converted into a shift?
    // First read BROWS+RRIGHT
    // After that BROWS
    unsigned int leftover = (RLEFT+height) << BCOLS_SHIFT; /// How far to fill buffer on last read
    //unsigned int blocks = (height-RRIGHT)/BROWS;
    
    unsigned int gofs, sofs;
    
    /// Fill shared memory -- read next block of BROWS rows
    /// We can skip RRIGHT rows as we've already copied them for the previous block
    /// and moved them to the top
    if(tidx < swidth)
    {
        gofs = 0; // Read from row (cur+RRIGHT)
        sofs = sdata;
        for(; sofs < leftover; sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]);
    }

    __syncthreads();
    
    doTransformTB(0, leftover);
    doTransformTB(1, leftover);
    
    __syncthreads();
    
    /// Write back leftover
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata;
        for(; sofs < leftover; sofs += (BCOLS*BSVY), gofs += istride)
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]);
    }
}
"""

transform_v_unroll = """

#if 0
// Rolled
#define READ_LOOP(rows) \
        for(; sofs < (rows); sofs += (BCOLS*BSVY), gofs += istride) \
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]);

#define WRITE_LOOP(rows) \
        for(; sofs < (rows); sofs += (BCOLS*BSVY), gofs += istride) \
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]);
#endif
#if 1
// Unrolled
#define READ_LOOP_ENTRY(rows) \
        if(sofs < (rows)) { \
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]); \
            sofs += (BCOLS*BSVY); gofs += istride; \
        } else break;
#define READ_LOOP(rows) \
        while(1) { \
            READ_LOOP_ENTRY(rows); \
            READ_LOOP_ENTRY(rows); \
        }

#define WRITE_LOOP_ENTRY(rows) \
        if(sofs < (rows)) { \
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]); \
            sofs += (BCOLS*BSVY); gofs += istride; \
        } else break;
#define WRITE_LOOP(rows) \
        while(1) { \
            WRITE_LOOP_ENTRY(rows); \
            WRITE_LOOP_ENTRY(rows); \
        }

#endif

#if 0
// Unrolled
#define READ_LOOP_ENTRY(rows) \
        if(sofs < (rows)) { \
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&data[gofs]); \
            sofs += (BCOLS*BSVY); gofs += istride; \
        }
#define READ_LOOP(rows) \
        READ_LOOP_ENTRY(rows); \
        READ_LOOP_ENTRY(rows); \
        READ_LOOP_ENTRY(rows); 

#define WRITE_LOOP_ENTRY(rows) \
        if(sofs < (rows)) { \
            *((uint32_t*)&data[gofs]) = *((uint32_t*)&shared[sofs]); \
            sofs += (BCOLS*BSVY); gofs += istride; \
        }

#define WRITE_LOOP(rows) \
        WRITE_LOOP_ENTRY(rows); \
        WRITE_LOOP_ENTRY(rows);  \
        WRITE_LOOP_ENTRY(rows); 

#endif

static __global__ void %(dir)s_transform_v( DATATYPE* data, int width, int height, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const unsigned int bid = blockIdx.x;    // slab (BCOLS columns)
    const unsigned int tidx = threadIdx.x<<1;   // column
    const unsigned int tidy = threadIdx.y;   // row    
    const unsigned int swidth = min(width-(bid<<BCOLS_SHIFT), BCOLS); // Width of this slab, usually BCOLS but can be less

    // Element offset in global memory
    //int idata = tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    data += tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    
    const unsigned int istride = __mul24(BSVY, stride);
    const unsigned int sdata = tidx + (tidy<<BCOLS_SHIFT);
    // First read BROWS+RRIGHT
    // After that BROWS
    unsigned int ref = height-(WRITEBACK-SKIPTOP+RRIGHT+COPYROWS);
    unsigned int blocks = ref/WRITEBACK;
    unsigned int leftover = (RLEFT+COPYROWS+RRIGHT+(ref%%WRITEBACK))<<BCOLS_SHIFT;
    
    unsigned int gofs,sofs;

    /// More than one block
    /// Read first block of BROWS+RRIGHT rows
    /// Upper RLEFT rows are left unitialized for now, later they should be copied from top
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + ((RLEFT+SKIPTOP)<<BCOLS_SHIFT);
        READ_LOOP(TOTALROWS<<BCOLS_SHIFT);
    }
// idata_read = idata_write + __mul24(TOTALROWS-RLEFT, stride)

    __syncthreads();
    
    doTransformT(0);
    doTransformT(1);
    
    __syncthreads();

    /// Write back WRITEBACK rows
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + ((RLEFT+SKIPTOP)<<BCOLS_SHIFT);
        WRITE_LOOP((WRITEBACK+RLEFT)<<BCOLS_SHIFT);
    }
// idata_read = idata_write + __mul24((BROWS+RRIGHT)-WRITEBACK, stride)

// Difference between global mem read and write pointer
#define DATA_READ_DIFF __mul24((BROWS+RRIGHT)-WRITEBACK, stride)
// Advance pointer with this amount after each block
#define DATA_INC __mul24(WRITEBACK, stride)

    data += __mul24(WRITEBACK-SKIPTOP, stride);
    for(unsigned int block=0; block<blocks; ++block)
    {
        __syncthreads();
        /// Move lower rows to top rows
#if OVERLAP <= BSVY 
        if(tidy < OVERLAP)
        {
            unsigned int l = (tidy<<BCOLS_SHIFT)+tidx;
            *((uint32_t*)&shared[l]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+l]);
        }
#else
        for(sofs = (tidy<<BCOLS_SHIFT)+tidx; sofs < (OVERLAP<<BCOLS_SHIFT); sofs += (BSVY<<BCOLS_SHIFT))
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+sofs]);
#endif                
        
        /// Fill shared memory -- read next block of BROWS rows
        /// We can skip RRIGHT rows as we've already copied them for the previous block
        /// and moved them to the top
        if(tidx < swidth)
        {
            gofs = DATA_READ_DIFF;
            sofs = sdata + (OVERLAP<<BCOLS_SHIFT);
            READ_LOOP(TOTALROWS<<BCOLS_SHIFT);
        }

        __syncthreads();

        doTransform(0);
        doTransform(1);

        __syncthreads();

        /// Write back BROWS rows
        if(tidx < swidth)
        {
            gofs = 0;
            sofs = sdata + (RLEFT<<BCOLS_SHIFT);
            WRITE_LOOP((WRITEBACK+RLEFT)<<BCOLS_SHIFT);
        }
        
        data += DATA_INC;
    }
    __syncthreads();

    ///
    /// Handle partial last block
    /// Move lower rows to top rows
#if OVERLAP <= BSVY 
        if(tidy < OVERLAP)
        {
            unsigned int l = (tidy<<BCOLS_SHIFT)+tidx;
            *((uint32_t*)&shared[l]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+l]);
        }
#else
        for(sofs = (tidy<<BCOLS_SHIFT)+tidx; sofs < (OVERLAP<<BCOLS_SHIFT); sofs += (BSVY<<BCOLS_SHIFT))
            *((uint32_t*)&shared[sofs]) = *((uint32_t*)&shared[(WRITEBACK<<BCOLS_SHIFT)+sofs]);
#endif                
    
    /// Fill shared memory -- read next block of BROWS rows
    /// We can skip RRIGHT rows as we've already copied them for the previous block
    /// and moved them to the top
    if(tidx < swidth)
    {
        gofs = DATA_READ_DIFF;
        sofs = sdata + (OVERLAP<<BCOLS_SHIFT);
        READ_LOOP(leftover);
    }

    __syncthreads();

    doTransformB(0, leftover);
    doTransformB(1, leftover);
    
    __syncthreads();
    
    /// Write back leftover
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata + (RLEFT<<BCOLS_SHIFT);
        WRITE_LOOP(leftover);
    }

}

/// Use this if the image is lower than PAD_ROWS
static __global__ void %(dir)s_transform_v_pad( DATATYPE* data, int width, int height, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const unsigned int bid = blockIdx.x;    // slab (BCOLS columns)
    const unsigned int tidx = threadIdx.x<<1;   // column
    const unsigned int tidy = threadIdx.y;   // row
    const unsigned int swidth = min(width-(bid<<BCOLS_SHIFT), BCOLS); // Width of this slab, usually BCOLS but can be less

    // Element offset in global memory
    //int idata = tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    data +=  tidx + (bid<<BCOLS_SHIFT) + __mul24(tidy, stride);
    const unsigned int istride = __mul24(BSVY, stride);
    const unsigned int sdata = tidx + ((tidy+RLEFT)<<BCOLS_SHIFT); // Does this get converted into a shift?
    // First read BROWS+RRIGHT
    // After that BROWS
    unsigned int leftover = (RLEFT+height) << BCOLS_SHIFT; /// How far to fill buffer on last read
    //unsigned int blocks = (height-RRIGHT)/BROWS;
    
    unsigned int gofs, sofs;
    
    /// Fill shared memory -- read next block of BROWS rows
    /// We can skip RRIGHT rows as we've already copied them for the previous block
    /// and moved them to the top
    if(tidx < swidth)
    {
        gofs = 0; // Read from row (cur+RRIGHT)
        sofs = sdata;
        READ_LOOP(leftover);
    }

    __syncthreads();
    
    doTransformTB(0, leftover);
    doTransformTB(1, leftover);
    
    __syncthreads();
    
    /// Write back leftover
    if(tidx < swidth)
    {
        gofs = 0;
        sofs = sdata;
        WRITE_LOOP(leftover);
    }
}
"""

