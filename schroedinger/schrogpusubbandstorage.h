#ifndef __SCHRO_SUBBANDSTORAGE_H__
#define __SCHRO_SUBBANDSTORAGE_H__

#include <schroedinger/schrobitstream.h>
#include <schroedinger/schrogpuframe.h>

//3*SCHRO_MAX_SUBBANDS*sizeof(int)

typedef struct _schro_subband_storage schro_subband_storage;

struct _schro_subband_storage
{
    SchroCUDAStream stream;
    int numbands;
    int transform_depth;
    int iwt_luma_width;
    int iwt_luma_height;
    int iwt_chroma_width;
    int iwt_chroma_height;
    
    int offsets[SCHRO_LIMIT_SUBBANDS*3];
    uint32_t *odata;
    int16_t *data;
    
    void *buffer;
    //void *gbuffer;
    
    int maxsize;
    int used;
    int16_t *zeroes;
};


schro_subband_storage* schro_subband_storage_new(SchroParams *params, SchroCUDAStream stream);
void schro_subband_storage_free(schro_subband_storage *store);
//void schro_subband_storage_to_gpuframe(schro_subband_storage *store, SchroFrame *frame);
void schro_subband_storage_to_gpuframe_init(schro_subband_storage *store, SchroFrame *frame);
void schro_subband_storage_to_gpuframe(schro_subband_storage *store, SchroFrame *frame, int comp, int position, int offset);

#endif
