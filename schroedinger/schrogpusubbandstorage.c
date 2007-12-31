#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrogpusubbandstorage.h>
#include <cuda_runtime_api.h>
#include <string.h>

#define OFFSET_S16(ptr,offset) ((int16_t *)(((uint8_t *)(ptr)) + (offset)))

schro_subband_storage* schro_subband_storage_new(SchroParams *params, SchroStream stream)
{
    schro_subband_storage *store;
    int zeroes_length;
    
    store = (schro_subband_storage*)malloc(sizeof(schro_subband_storage));
    memset(store, 0, sizeof(schro_subband_storage));
    
    store->stream = stream;
    
    store->numbands = 3*(params->transform_depth*3+1);
    store->iwt_luma_width = params->iwt_luma_width;
    store->iwt_luma_height = params->iwt_luma_height;
    store->iwt_chroma_width = params->iwt_chroma_width;
    store->iwt_chroma_height = params->iwt_chroma_height;
    store->transform_depth = params->transform_depth;
    
    store->maxsize = params->iwt_luma_width*params->iwt_luma_height+2*params->iwt_chroma_width*params->iwt_chroma_height;
    /// Add place for padding
    store->maxsize += store->numbands*32;
    store->used = 0;
    
    /** Make sure offset table is prefixed to buffer, so everything can be copied in one big transfer */
    //store->buffer = malloc(store->data_offset + store->maxsize*2);
    cudaMallocHost((void**)&store->buffer, store->maxsize*2);
    store->odata = (uint32_t*)store->buffer;
    store->data = (int16_t*)(store->buffer);
    
    /** On GPU, allocate space for both the pointer table and the frame data */
    //cudaMalloc((void**)&store->gbuffer, store->maxsize*2);
    
    /** Size of largest subband for dummy subband */
    zeroes_length = (params->iwt_luma_width>>2)*(params->iwt_luma_height>>2);
    store->zeroes = malloc(zeroes_length*2);
    memset(store->zeroes, 0, zeroes_length*2);
    
    SCHRO_DEBUG("numbands=%i luma_width=%i luma_height=%i chroma_width=%i chroma_height=%i bufsize=%i zeroes_length=%i", 
        store->numbands,
        params->iwt_luma_width, params->iwt_luma_height, params->iwt_chroma_width, params->iwt_chroma_height,
        store->maxsize, zeroes_length);
    
    return store;
}

void schro_subband_storage_free(schro_subband_storage *store)
{
    //free(store->buffer);
    cudaFreeHost(store->buffer);
    free(store->zeroes);
    free(store);
    //cudaFree(store->gbuffer);
}

void schro_subband_storage_to_gpuframe_init(schro_subband_storage *store, SchroGPUFrame *frame)
{
    /** Zero the frame */
    schro_gpuframe_zero(frame);
}
#if 1
void schro_subband_storage_to_gpuframe(schro_subband_storage *store, SchroGPUFrame *frame, int comp, int position, int offset)
{
    int16_t *srcdata;
    int data_width, data_height, data_stride;
    int shift, width, height, dstride, sstride;
    int16_t *dest;
    
    /** Transfer one subband */
    switch(comp)
    {
    case 0:
        dest = frame->components[0].gdata;
        data_width = store->iwt_luma_width;
        data_height = store->iwt_luma_height;
        data_stride = frame->components[0].stride;
        break;
    case 1:
        dest = frame->components[1].gdata;
        data_width = store->iwt_chroma_width;
        data_height = store->iwt_chroma_height;
        data_stride = frame->components[1].stride;
        break;
    case 2:
        dest = frame->components[2].gdata;
        data_width = store->iwt_chroma_width;
        data_height = store->iwt_chroma_height;
        data_stride = frame->components[2].stride;
        break;
    default:
        SCHRO_ASSERT(0);
    }
    
    srcdata = (int16_t*)store->buffer;
    shift = store->transform_depth - SCHRO_SUBBAND_SHIFT(position);
    width = data_width >> shift;
    height = data_height >> shift;
    dstride = data_stride << shift; 
    sstride = width<<1;

    if (position & 2)
       dest = OFFSET_S16(dest, dstride>>1);
    if (position & 1)
       dest += width;

    /// Queue memory copy command
    cudaMemcpy2DAsync(dest, dstride, srcdata + offset, sstride, width<<1, height, cudaMemcpyHostToDevice, store->stream);
    //SCHRO_ERROR("cudaMemcpy2DAsync %p %i %p %i %i %i", dest, dstride, srcdata + offset, sstride, width, height);
}
#endif

#if 0
void schro_subband_storage_to_gpuframe(schro_subband_storage *store, SchroGPUFrame *frame)
{
    int16_t *data[3];
    int16_t *srcdata;
    int data_width[3], data_height[3], data_stride[3];
    int transform_depth;

    int bands_per_component, x, comp, sub;
    SCHRO_DEBUG("to frame %ix%i <- %i", frame->width, frame->height, store->used*2);
    
    /** Zero the frame */
    schro_gpuframe_zero(frame);
    
    data[0] = frame->components[0].gdata;
    data_width[0] = store->iwt_luma_width;
    data_height[0] = store->iwt_luma_height;
    data_stride[0] = frame->components[0].stride;
    
    data[1] = frame->components[1].gdata;
    data_width[1] = store->iwt_chroma_width;
    data_height[1] = store->iwt_chroma_height;
    data_stride[1] = frame->components[1].stride;
    
    data[2] = frame->components[2].gdata;
    data_width[2] = store->iwt_chroma_width;
    data_height[2] = store->iwt_chroma_height;
    data_stride[2] = frame->components[2].stride;
    
    transform_depth = store->transform_depth;
    
    srcdata = (int16_t*)store->buffer;

    bands_per_component = store->transform_depth*3+1;    
    x = 0;
    for(comp=0; comp<3; ++comp)
    {
        for(sub=0; sub<bands_per_component; ++sub)
        {
            if(store->offsets[x] != -1)
            {
                int offset = store->offsets[x];
                int position = schro_subband_get_position(sub);
                
                int shift = transform_depth - SCHRO_SUBBAND_SHIFT(position);
            
                int width = data_width[comp] >> shift;
                int height = data_height[comp] >> shift;
                int dstride = data_stride[comp] << shift; 
                int sstride = width<<1;

                int16_t *dest = data[comp]; // component buffer start
                if (position & 2)
                   dest = OFFSET_S16(dest, dstride>>1);
                if (position & 1)
                   dest += width;

                /// Queue memory copy command
                cudaMemcpy2DAsync(dest, dstride, srcdata + offset, sstride, width<<1, height, cudaMemcpyHostToDevice, store->stream);
                //SCHRO_ERROR("cudaMemcpy2DAsync %p %i %p %i %i %i", dest, dstride, params.buffer + offset, sstride, width, height);
            }
            x++;
        }
    }

}
#endif

