
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <cuda_runtime_api.h>

#include <schroedinger/schrogpuframe.h>

void
schro_cuda_init (void)
{
  int n;
  int i;
  cudaError_t ret;

  ret = cudaGetDeviceCount (&n);
  SCHRO_ERROR("cudaGetDeviceCount returned %d", ret);

  SCHRO_ERROR("CUDA devices %d", n);

  for(i=0;i<n;i++){
    struct cudaDeviceProp prop;

    cudaGetDeviceProperties (&prop, i);
    SCHRO_ERROR ("CUDA props: %d: %d.%d mem=%d %s", i,
        prop.major, prop.minor,
        prop.totalGlobalMem, prop.name);
  }
}


static void *
schro_cuda_alloc (int size)
{
  void *ptr;
  int ret;

  SCHRO_ERROR("domain is %d", schro_async_get_exec_domain ());
  SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);

  ret = cudaMalloc (&ptr, size);

  return ptr;
}

static void
schro_cuda_free (void *ptr, int size)
{
  SCHRO_ERROR("domain is %d", schro_async_get_exec_domain ());
  SCHRO_ASSERT(schro_async_get_exec_domain () == SCHRO_EXEC_DOMAIN_CUDA);

  cudaFree (ptr);
}

SchroMemoryDomain *
schro_memory_domain_new_cuda (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new();
  domain->flags = SCHRO_MEMORY_DOMAIN_CUDA;
  domain->alloc = schro_cuda_alloc;
  domain->free = schro_cuda_free;
  
  return domain;
}


void
schro_motion_render_cuda (SchroMotion *motion, SchroFrame *dest)
{

  SCHRO_ASSERT(0);
}

void
schro_frame_inverse_iwt_transform_cuda (SchroFrame *frame,
    SchroFrame *transform_frame, SchroParams *params)
{
  schro_frame_to_gpu (frame, transform_frame);

  schro_gpuframe_inverse_iwt_transform (frame, params);
}

