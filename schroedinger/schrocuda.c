
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <cuda_runtime_api.h>


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

  cudaMalloc (&ptr, size);

  return NULL;
}

static void
schro_cuda_free (void *ptr, int size)
{
  cudaFree (ptr);
}

SchroMemoryDomain *
schro_memory_domain_new_cuda (void)
{
  SchroMemoryDomain *domain;

  domain = schro_memory_domain_new();
  domain->alloc = schro_cuda_alloc;
  domain->free = schro_cuda_free;
  
  return domain;
}


