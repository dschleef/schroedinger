s_transform_h_begin = """
static __global__ void s_transform_h( DATATYPE* data, int width, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const int bid = blockIdx.x;    // row
    const int tid = threadIdx.x;   // thread id within row
    const int tidu16 = ((tid&16)>>4)|((tid&15)<<1)|(tid&~31);
    
    data += __mul24(bid, stride);

    int half = BLEFT+(width>>1)+BRIGHT;

    unsigned int ofs;

    if(width&3) // If width is not a multiple of 4, we need to use the slower method
    {
        /// Left part (even coefficients)
        /// Right part (odd coefficients)
        for(ofs = tid; ofs < (width>>1); ofs += BSH)
        {
            shared[BLEFT+ofs] = data[ofs];
            shared[half+BLEFT+ofs] = data[(width>>1)+ofs];   
        }
    } 
    else
    {
        // Shared memory output offset for this thread
        uint32_t *row = (uint32_t*)data;
        /// Left part (even coefficients)
        for(ofs = tid; ofs < (width>>2); ofs += BSH)
        {
            *((uint32_t*)&shared[BLEFT+(ofs<<1)]) = row[ofs];
            *((uint32_t*)&shared[half+BLEFT+(ofs<<1)]) = row[ofs + (width>>2)];
        }
    }

    __syncthreads();
"""

s_transform_h_begin_unroll = """
static __global__ void s_transform_h( DATATYPE* data, int width, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const int bid = blockIdx.x;    // row
    const int tid = threadIdx.x;   // thread id within row
    const int tidu16 = ((tid&16)>>4)|((tid&15)<<1)|(tid&~31);
    
    data += __mul24(bid, stride);

    int half = BLEFT+(width>>1)+BRIGHT;

    int ofs;
    if(width&3) // If width is not a multiple of 4, we need to use the slower method
    {
        /// Left part (even coefficients)
        /// Right part (odd coefficients)
        int w2 = (width>>1);
        uint16_t *row = (uint16_t*)data;
        uint16_t *row2 = (uint16_t*)&data[w2];
        uint16_t *dest1 = (uint16_t*)&shared[BLEFT];
        uint16_t *dest2 = (uint16_t*)&shared[half+BLEFT];
        
        ofs = tid;
        while(true)
        {
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];   
                ofs += BSH;
            } else break;
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];   
                ofs += BSH;
            } else break;
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];   
                ofs += BSH;
            } else break;
        }
    } 
    else
    {
        // Shared memory output offset for this thread
        int w2 = (width>>2);
        uint32_t *row = (uint32_t*)data;
        uint32_t *row2 = (uint32_t*)&data[width>>1];
        uint32_t *dest1 = (uint32_t*)&shared[BLEFT];
        uint32_t *dest2 = (uint32_t*)&shared[half+BLEFT];

        /// Left part (even coefficients)
        ofs = tid;
        while(true)
        {
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];
                ofs += BSH;
            } else break;
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];
                ofs += BSH;
            } else break;
            if(ofs<w2)
            {
                dest1[ofs] = row[ofs];
                dest2[ofs] = row2[ofs];
                ofs += BSH;
            } else break;
        }
    }

    __syncthreads();
"""

s_transform_h_end = """
    uint32_t *row = (uint32_t*)data;
    for(ofs = tid; ofs < (width>>1); ofs += BSH)
    {
        // 2-way bank conflict
        uint16_t a = (uint16_t)((shared[BLEFT + ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT);
        // 2-way bank conflict
        uint16_t b = (uint16_t)((shared[BLEFT + ofs + half] + INITIAL_OFFSET)>>INITIAL_SHIFT);
        row[ofs] = a|(b<<16);
    }

}

"""
s_transform_h_end_unroll = """
    if(width&3)
    {
        uint32_t *row = (uint32_t*)data;
        int16_t *src1 = (int16_t*)&shared[BLEFT];
        int16_t *src2 = (int16_t*)&shared[half+BLEFT];
        for(ofs = tid; ofs < (width>>1); ofs += BSH)
        {
            int a = (src1[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;
            int b = (src2[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;

            row[ofs] = (a&0xFFFF)|((b&0xFFFF)<<16);
        }

    }
    else
    {
        i16_4 *row = (i16_4*)data;
        i16_2 *src1 = (i16_2*)&shared[BLEFT];
        i16_2 *src2 = (i16_2*)&shared[half+BLEFT];
        for(ofs = tid; ofs < (width>>2); ofs += BSH)
        {
            i16_4 x;
            i16_2 a = src1[ofs];
            x.a = (a.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
            x.c = (a.b + INITIAL_OFFSET)>>INITIAL_SHIFT;
            i16_2 b = src2[ofs];
            x.b = (b.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
            x.d = (b.b + INITIAL_OFFSET)>>INITIAL_SHIFT;

            row[ofs] = x;
        }
    }
}
"""
s_transform_h_end_unroll2 = """
    if(width&3)
    {
        uint32_t *row = (uint32_t*)data;
        int16_t *src1 = (int16_t*)&shared[BLEFT];
        int16_t *src2 = (int16_t*)&shared[half+BLEFT];
        int w2 = (width>>1);
        ofs = tid;
        while(true)
        {
            if(ofs<w2)
            {
                int a = (src1[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;
                int b = (src2[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = (a&0xFFFF)|((b&0xFFFF)<<16);
                
                ofs += BSH;
            } else break;
            if(ofs<w2)
            {
                int a = (src1[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;
                int b = (src2[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = (a&0xFFFF)|((b&0xFFFF)<<16);
                
                ofs += BSH;
            } else break;            
            if(ofs<w2)
            {
                int a = (src1[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;
                int b = (src2[ofs] + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = (a&0xFFFF)|((b&0xFFFF)<<16);
                
                ofs += BSH;
            } else break;
        }

    }
    else
    {
        i16_4 *row = (i16_4*)data;
        i16_2 *src1 = (i16_2*)&shared[BLEFT];
        i16_2 *src2 = (i16_2*)&shared[half+BLEFT];
        int w2 = (width>>2);
        ofs = tid;
        while(true)
        {
            if(ofs < w2)
            {
                i16_4 x;
                i16_2 a = src1[ofs];
                x.a = (a.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.c = (a.b + INITIAL_OFFSET)>>INITIAL_SHIFT;
                i16_2 b = src2[ofs];
                x.b = (b.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.d = (b.b + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = x;
                
                ofs += BSH;
            } else break;
            if(ofs < w2)
            {
                i16_4 x;
                i16_2 a = src1[ofs];
                x.a = (a.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.c = (a.b + INITIAL_OFFSET)>>INITIAL_SHIFT;
                i16_2 b = src2[ofs];
                x.b = (b.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.d = (b.b + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = x;
                
                ofs += BSH;
            } else break;            
            if(ofs < w2)
            {
                i16_4 x;
                i16_2 a = src1[ofs];
                x.a = (a.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.c = (a.b + INITIAL_OFFSET)>>INITIAL_SHIFT;
                i16_2 b = src2[ofs];
                x.b = (b.a + INITIAL_OFFSET)>>INITIAL_SHIFT;
                x.d = (b.b + INITIAL_OFFSET)>>INITIAL_SHIFT;

                row[ofs] = x;
                
                ofs += BSH;
            } else break;
        }
    }
}
"""

a_transform_h_begin = """
static __global__ void a_transform_h( DATATYPE* data, int width, int stride )
{
    extern __shared__ DATATYPE shared[];  

    const int bid = blockIdx.x;    // row
    const int tid = threadIdx.x;   // thread id within row
    const int tidu16 = ((tid&16)>>4)|((tid&15)<<1)|(tid&~31);

    data += __mul24(bid, stride);

    // Load entire line into shared memory
    // Deinterleave right here
    int half = BLEFT+(width>>1)+BRIGHT;

    unsigned int ofs;

    // Shared memory output offset for this thread
    i16_2 *row = (i16_2*)data;
    for(ofs = tid; ofs < (width>>1); ofs += BSH)
    {
        i16_2 val = row[ofs];
        shared[BLEFT + ofs]        = val.a << INITIAL_SHIFT; // even
        shared[BLEFT + ofs + half] = val.b << INITIAL_SHIFT; // odd
    }
    __syncthreads();
"""

a_transform_h_end = """
    /// Write line back to global memory, don't interleave again
    /// Mind the gap between BLEFT+width/2 and half
    if(width&3) // If width is not a multiple of 4, we need to use the slower method
    {
        /// Left part (even coefficients)
        for(ofs = tid; ofs < (width>>1); ofs += BSH)
            data[ofs] = shared[BLEFT+ofs];

        /// Right part (odd coefficients)
        for(ofs = tid; ofs < (width>>1); ofs += BSH)
            data[(width>>1)+ofs] = shared[half+BLEFT+ofs];   
    } 
    else
    {
        /// Left part (even coefficients)
        for(ofs = tid; ofs < (width>>2); ofs += BSH)
            row[ofs] = *((i16_2*)&shared[BLEFT+(ofs<<1)]);
        row += (width>>2);
        /// Right part (odd coefficients)
        for(ofs = tid; ofs < (width>>2); ofs += BSH)
            row[ofs] = *((i16_2*)&shared[half+BLEFT+(ofs<<1)]);
    }
}
"""

