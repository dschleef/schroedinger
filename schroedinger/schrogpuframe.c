#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <schroedinger/schro.h>
#include <schroedinger/schrogpuframe.h>
#include <schroedinger/schrooil.h>
#include <liboil/liboil.h>

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include "cudawavelet.h"
#include "cudaframe.h"
#include "cudaupsample.h"
#include "cudamotion.h"
#include <stdio.h>

void schro_gpuframe_setstream(SchroFrame *frame, SchroCUDAStream stream)
{
  SCHRO_ASSERT(frame->is_cuda_frame == TRUE);

  frame->stream = stream;
}

int schro_bpp(int format)
{
  int bytes_pp;
  if(SCHRO_FRAME_IS_PACKED(format))
  {
    switch(format)
    {
    case SCHRO_FRAME_FORMAT_YUYV:
    case SCHRO_FRAME_FORMAT_UYVY:
        bytes_pp = 2;
        break;
    case SCHRO_FRAME_FORMAT_AYUV:
        bytes_pp = 4;
        break;
    default:
        SCHRO_ASSERT(0);
    }
  }
  else
  {
    switch (SCHRO_FRAME_FORMAT_DEPTH(format)) {
      case SCHRO_FRAME_FORMAT_DEPTH_U8:
        bytes_pp = 1;
        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S16:
        bytes_pp = 2;
        break;
      case SCHRO_FRAME_FORMAT_DEPTH_S32:
        bytes_pp = 4;
        break;
      default:
        SCHRO_ASSERT(0);
        bytes_pp = 0;
        break;
    }
  }
  return bytes_pp;
}

int schro_components(int format)
{
  int comp;
  if(SCHRO_FRAME_IS_PACKED(format))
  {
    // packed
    return 1;
  }
  else
  {
    // planar
    return 3;
  }
  return comp;
}

SchroFrame *schro_gpuframe_new_and_alloc (SchroFrameFormat format, int width, int height)
{
  SchroFrame *frame = schro_frame_new();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  
  SCHRO_ASSERT(width > 0);
  SCHRO_ASSERT(height > 0);

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->is_cuda_frame = TRUE;
  frame->is_cuda_shared = FALSE;
  
  SCHRO_DEBUG("schro_gpuframe_new_and_alloc %i %i %i", frame->format, frame->width, frame->height);

  bytes_pp = schro_bpp(format);

  if(!SCHRO_FRAME_IS_PACKED(format))
  {
      h_shift = SCHRO_FRAME_FORMAT_H_SHIFT(format);
      v_shift = SCHRO_FRAME_FORMAT_V_SHIFT(format);
      chroma_width = ROUND_UP_SHIFT(width, h_shift);
      chroma_height = ROUND_UP_SHIFT(height, v_shift);

      frame->components[0].width = width;
      frame->components[0].height = height;
      frame->components[0].stride = ROUND_UP_64(width * bytes_pp);
      frame->components[0].length = 
        frame->components[0].stride * frame->components[0].height;
      frame->components[0].v_shift = 0;
      frame->components[0].h_shift = 0;

      frame->components[1].width = chroma_width;
      frame->components[1].height = chroma_height;
      frame->components[1].stride = ROUND_UP_64(chroma_width * bytes_pp);
      frame->components[1].length = 
        frame->components[1].stride * frame->components[1].height;
      frame->components[1].v_shift = v_shift;
      frame->components[1].h_shift = h_shift;

      frame->components[2].width = chroma_width;
      frame->components[2].height = chroma_height;
      frame->components[2].stride = ROUND_UP_64(chroma_width * bytes_pp);
      frame->components[2].length = 
        frame->components[2].stride * frame->components[2].height;
      frame->components[2].v_shift = v_shift;
      frame->components[2].h_shift = h_shift;

      cudaMalloc((void**)&frame->gregions[0], frame->components[0].length +
          frame->components[1].length + frame->components[2].length);

      frame->components[0].gdata = frame->gregions[0];
      frame->components[1].gdata = frame->components[0].gdata +
        frame->components[0].length;
      frame->components[2].gdata = frame->components[1].gdata +
        frame->components[1].length;
  }
  else
  {
      frame->components[0].width = width;
      frame->components[0].height = height;
      frame->components[0].stride = ROUND_UP_4(width * bytes_pp);
      frame->components[0].length = 
        frame->components[0].stride * frame->components[0].height;
      frame->components[0].v_shift = 0;
      frame->components[0].h_shift = 0;

      cudaMalloc((void**)&frame->gregions[0], frame->components[0].length);

      frame->components[0].gdata = frame->gregions[0];
  }

  return frame;
}

SchroFrame *schro_gpuframe_new_clone (SchroFrame *src)
{
  SchroFrame *frame = schro_frame_new();
  int i, length;
  void *ptr;
  
  SCHRO_ASSERT(src->is_cuda_frame == FALSE);

  frame->format = src->format;
  frame->width = src->width;
  frame->height = src->height;
  frame->is_cuda_frame = TRUE;
  
  length = src->components[0].length + src->components[1].length + src->components[2].length;
  cudaMalloc((void**)&frame->gregions[0], length);
  
  SCHRO_DEBUG("schro_gpuframe_new_clone %i %i %i (%i)", frame->format, frame->width, frame->height, length);

  /** Copy components and allocate space */
  ptr = frame->gregions[0];
  for(i=0; i<3; ++i)
  {
      frame->components[i].width = src->components[i].width;
      frame->components[i].height = src->components[i].height;
      frame->components[i].stride = src->components[i].stride;
      frame->components[i].length = src->components[i].length;
      frame->components[i].v_shift = src->components[i].v_shift;
      frame->components[i].h_shift = src->components[i].h_shift;
      
      if(frame->components[i].length)
      {
          frame->components[i].gdata = ptr;
          cudaMemcpy(ptr, src->components[i].data, frame->components[i].length, cudaMemcpyHostToDevice);
          
          ptr += frame->components[i].length;
      }
  }

  return frame;
}

void _schro_gpuframe_free (SchroFrame *frame)
{
  if (frame->gregions[0]) {
    cudaFree(frame->gregions[0]);
  }
}

void schro_gpuframe_convert (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);
  SCHRO_ASSERT(dest->is_cuda_frame == TRUE);
  SCHRO_ASSERT(src->is_cuda_frame == TRUE);
  
  SCHRO_DEBUG("schro_gpuframe_convert %ix%i(format %i) <- %ix%i(format %i)", dest->width, dest->height, dest->format, src->width, src->height, src->format);

  dest->frame_number = src->frame_number;
  
  if((src->format==SCHRO_FRAME_FORMAT_S16_444 && dest->format==SCHRO_FRAME_FORMAT_U8_444) ||
     (src->format==SCHRO_FRAME_FORMAT_S16_422 && dest->format==SCHRO_FRAME_FORMAT_U8_422) ||
     (src->format==SCHRO_FRAME_FORMAT_S16_420 && dest->format==SCHRO_FRAME_FORMAT_U8_420))
  {
      // S16 to U8
      for(i=0; i<3; ++i)
          cuda_convert_u8_s16(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                              src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                              dest->stream);
  }
  else if((src->format==SCHRO_FRAME_FORMAT_U8_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
          (src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
          (src->format==SCHRO_FRAME_FORMAT_U8_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // U8 to S16
      for(i=0; i<3; ++i)
          cuda_convert_s16_u8(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                              src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                              dest->stream);

  }
  else if((src->format==SCHRO_FRAME_FORMAT_U8_444 && dest->format==SCHRO_FRAME_FORMAT_U8_444) ||
          (src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_U8_422) ||
          (src->format==SCHRO_FRAME_FORMAT_U8_420 && dest->format==SCHRO_FRAME_FORMAT_U8_420))
  {
      // U8 to U8
      for(i=0; i<3; ++i)
          cuda_convert_u8_u8(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                             src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                             dest->stream);
  }
  else if((src->format==SCHRO_FRAME_FORMAT_S16_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // S16 to S16
      for(i=0; i<3; ++i)
          cuda_convert_s16_s16(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                               src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                               dest->stream);
  }
  else if(src->format==SCHRO_FRAME_FORMAT_YUYV && dest->format==SCHRO_FRAME_FORMAT_U8_422)
  {
      // deinterleave YUYV
      cuda_convert_u8_422_yuyv(dest->components[0].gdata, dest->components[0].stride,
                               dest->components[1].gdata, dest->components[1].stride,
                               dest->components[2].gdata, dest->components[2].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->width, src->height,
                               dest->stream);
  }
  else if(src->format==SCHRO_FRAME_FORMAT_UYVY && dest->format==SCHRO_FRAME_FORMAT_U8_422)
  {
      // deinterleave UYVY
      cuda_convert_u8_422_uyvy(dest->components[0].gdata, dest->components[0].stride,
                               dest->components[1].gdata, dest->components[1].stride,
                               dest->components[2].gdata, dest->components[2].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->width, src->height,
                               dest->stream);
  }
  else if(src->format==SCHRO_FRAME_FORMAT_AYUV && dest->format==SCHRO_FRAME_FORMAT_U8_444)
  {
      // deinterleave AYUV
      cuda_convert_u8_444_ayuv(dest->components[0].gdata, dest->components[0].stride,
                               dest->components[1].gdata, dest->components[1].stride,
                               dest->components[2].gdata, dest->components[2].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->width, src->height,
                               dest->stream);

  }
  else if(src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_YUYV)
  {
      // interleave YUYV
      cuda_convert_yuyv_u8_422(dest->components[0].gdata, dest->components[0].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->components[1].gdata, src->components[1].stride,
                               src->components[2].gdata, src->components[2].stride,
                               src->width, src->height,
                               dest->stream);
  }
  else if(src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_UYVY)
  {
      // interleave UYVY
      cuda_convert_uyvy_u8_422(dest->components[0].gdata, dest->components[0].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->components[1].gdata, src->components[1].stride,
                               src->components[2].gdata, src->components[2].stride,
                               src->width, src->height,
                               dest->stream);
  }
  else if(src->format==SCHRO_FRAME_FORMAT_U8_444 && dest->format==SCHRO_FRAME_FORMAT_AYUV)
  {
      // interleave AYUV
      cuda_convert_ayuv_u8_444(dest->components[0].gdata, dest->components[0].stride,
                               dest->width, dest->height,
                               src->components[0].gdata, src->components[0].stride,
                               src->components[1].gdata, src->components[1].stride,
                               src->components[2].gdata, src->components[2].stride,
                               src->width, src->height,
                               dest->stream);

  }
  else
  {
      SCHRO_ERROR("conversion unimplemented");
      SCHRO_ASSERT(0);
  }
}

void schro_gpuframe_add (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);
  SCHRO_ASSERT(dest->is_cuda_frame == TRUE);
  SCHRO_ASSERT(src->is_cuda_frame == TRUE);
  
  SCHRO_DEBUG("schro_gpuframe_add %ix%i(format %i) <- %ix%i(format %i)", dest->width, dest->height, dest->format, src->width, src->height, src->format);
  
  if((src->format==SCHRO_FRAME_FORMAT_U8_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
     (src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
     (src->format==SCHRO_FRAME_FORMAT_U8_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // U8 to S16
      for(i=0; i<3; ++i)
          cuda_add_s16_u8(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                          src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height, 
                          dest->stream);
  }
  else if((src->format==SCHRO_FRAME_FORMAT_S16_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // S16 to S16
      for(i=0; i<3; ++i)
          cuda_add_s16_s16(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                           src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                           dest->stream);
  }
  else 
  {
      SCHRO_ERROR("add function unimplemented");
      SCHRO_ASSERT(0);
  }
}

void schro_gpuframe_subtract (SchroFrame *dest, SchroFrame *src)
{
  int i;

  SCHRO_ASSERT(dest != NULL);
  SCHRO_ASSERT(src != NULL);
  SCHRO_ASSERT(dest->is_cuda_frame == TRUE);
  SCHRO_ASSERT(src->is_cuda_frame == TRUE);
  
  SCHRO_DEBUG("schro_gpuframe_subtract %ix%i(format %i) <- %ix%i(format %i)", dest->width, dest->height, dest->format, src->width, src->height, src->format);
  
  if((src->format==SCHRO_FRAME_FORMAT_U8_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
     (src->format==SCHRO_FRAME_FORMAT_U8_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
     (src->format==SCHRO_FRAME_FORMAT_U8_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // U8 to S16
      for(i=0; i<3; ++i)
          cuda_subtract_s16_u8(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                          src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                          dest->stream);
  }
  else if((src->format==SCHRO_FRAME_FORMAT_S16_444 && dest->format==SCHRO_FRAME_FORMAT_S16_444) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_422 && dest->format==SCHRO_FRAME_FORMAT_S16_422) ||
          (src->format==SCHRO_FRAME_FORMAT_S16_420 && dest->format==SCHRO_FRAME_FORMAT_S16_420))
  {
      // S16 to S16
      for(i=0; i<3; ++i)
          cuda_subtract_s16_s16(dest->components[i].gdata, dest->components[i].stride, dest->components[i].width, dest->components[i].height,
                           src->components[i].gdata, src->components[i].stride, src->components[i].width, src->components[i].height,
                           dest->stream);
  }
  else 
  {
      SCHRO_ERROR("add function unimplemented");
      SCHRO_ASSERT(0);
  }
}


void schro_gpuframe_iwt_transform (SchroFrame *frame, SchroParams *params)
{
  int16_t *frame_data;
  int component;
  int width;
  int height;
  int level;
  SCHRO_DEBUG("schro_gpuframe_iwt_transform %ix%i (%i levels)", frame->width, frame->height, params->transform_depth);
  
  SCHRO_ASSERT(frame->is_cuda_frame == TRUE);

  for(component=0;component<3;component++){
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)comp->gdata;
    for(level=0;level<params->transform_depth;level++) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;

      cuda_wavelet_transform_2d (params->wavelet_filter_index, frame_data, stride, w, h, frame->stream);
    }
  }
}

void schro_gpuframe_inverse_iwt_transform (SchroFrame *frame, SchroParams *params)
{
  int16_t *frame_data;
  int width;
  int height;
  int level;
  int component;
  SCHRO_DEBUG("schro_gpuframe_inverse_iwt_transform %ix%i, filter %i, %i levels", frame->width, frame->height, params->wavelet_filter_index, params->transform_depth);
#ifdef TEST
  int16_t *c_data;
  int x;
    
  c_data = malloc(frame->components[0].stride * params->iwt_luma_height);
#endif

  SCHRO_ASSERT(frame->is_cuda_frame == TRUE);

  for(component=0;component<3;component++){
    SchroFrameData *comp = &frame->components[component];

    if (component == 0) {
      width = params->iwt_luma_width;
      height = params->iwt_luma_height;
    } else {
      width = params->iwt_chroma_width;
      height = params->iwt_chroma_height;
    }
    
    frame_data = (int16_t *)comp->gdata;
#ifdef TEST
    /// Copy frame from GPU
    cudaMemcpy(c_data, frame_data, height * comp->stride, cudaMemcpyDeviceToHost);
    for(x=0; x<width; ++x)
    {
        fprintf(stderr, "%i ", c_data[x]);
    }
    fprintf(stderr, "\n");
#endif
    
    for(level=params->transform_depth-1; level >=0;level--) {
      int w;
      int h;
      int stride;

      w = width >> level;
      h = height >> level;
      stride = comp->stride << level;
      
      cuda_wavelet_inverse_transform_2d (params->wavelet_filter_index, frame_data, stride, w, h, frame->stream);
    }
#ifdef TEST
    /// Copy frame from GPU
    cudaMemcpy(c_data, frame_data, height * comp->stride, cudaMemcpyDeviceToHost);
    for(x=0; x<width; ++x)
    {
        fprintf(stderr, "%i ", c_data[x]);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "---------------------\n");
#endif
  }
#ifdef TEST
  free(c_data);
#endif
}

void schro_gpuframe_to_cpu (SchroFrame *dest, SchroFrame *src)
{
    int i;
    
    SCHRO_ASSERT(src->is_cuda_frame == TRUE);
    SCHRO_ASSERT(dest->is_cuda_frame == FALSE);

    dest->frame_number = src->frame_number;
    int bpp = schro_bpp(dest->format);    
    SCHRO_DEBUG("schro_gpuframe_to_cpu %ix%i (%i)", dest->width, dest->height, dest->frame_number);
    
    /** Format, components and dimensions must match exactly */
    SCHRO_ASSERT(src->format == dest->format);
    for(i=0; i<3; ++i)
    {
        if(src->components[i].gdata)
        {
            SCHRO_ASSERT(dest->components[i].data);
            //SCHRO_ASSERT(dest->components[i].stride==src->components[i].stride && dest->components[i].length==src->components[i].length);
            SCHRO_ASSERT(dest->components[i].width==src->components[i].width && dest->components[i].height==src->components[i].height);
        }
    }
    /** If the buffer is consecutive, move it in one pass */
    if(src->components[1].gdata == (src->components[0].gdata + src->components[0].length) &&
       src->components[2].gdata == (src->components[1].gdata + src->components[1].length) &&
       dest->components[1].data == (dest->components[0].data + dest->components[0].length) &&
       dest->components[2].data == (dest->components[1].data + dest->components[1].length) &&
       src->components[0].length == dest->components[0].length &&
       src->components[1].length == dest->components[1].length &&
       src->components[2].length == dest->components[2].length)
    {
        SCHRO_DEBUG("consecutive %i+%i+%i", src->components[0].length, src->components[1].length, src->components[2].length);
        cudaMemcpy(dest->components[0].data, src->components[0].gdata, src->components[0].length+src->components[1].length+src->components[2].length, cudaMemcpyDeviceToHost);
    }
    else
    {
        for(i=0; i<3; ++i)
        {
            if(src->components[i].gdata)
            {
                SCHRO_DEBUG("component %i: %p %i %p %i %i %i",
                i,
                dest->components[i].data, dest->components[i].stride, 
                             src->components[i].gdata, src->components[i].stride,
                             src->components[i].width*bpp, src->components[i].height
                );

//                cudaMemcpy(dest->components[i].data, src->components[i].gdata, src->components[i].length, cudaMemcpyDeviceToHost);
                cudaMemcpy2D(dest->components[i].data, dest->components[i].stride, 
                             src->components[i].gdata, src->components[i].stride,
                             src->components[i].width*bpp, src->components[i].height,
                             cudaMemcpyDeviceToHost);
            }
        }
    }
}

void schro_frame_to_gpu (SchroFrame *dest, SchroFrame *src)
{
    int i;
    
    SCHRO_ASSERT(src->is_cuda_frame == FALSE);
    SCHRO_ASSERT(dest->is_cuda_frame == TRUE);

    dest->frame_number = src->frame_number;
    int bpp = schro_bpp(dest->format);
    SCHRO_DEBUG("schro_frame_to_gpu %ix%i (%i)", dest->width, dest->height, dest->frame_number);

    /** Format, components and dimensions must match exactly */
    SCHRO_ASSERT(src->format == dest->format);
    for(i=0; i<3; ++i)
    {
        if(src->components[i].data)
        {
            SCHRO_ASSERT(dest->components[i].gdata);
            //SCHRO_ASSERT(dest->components[i].stride==src->components[i].stride && dest->components[i].length==src->components[i].length);
            SCHRO_ASSERT(dest->components[i].width==src->components[i].width && dest->components[i].height==src->components[i].height);
        }
    }
    /** If the buffer is consecutive, move it in one pass */
    if(src->components[1].data == (src->components[0].data + src->components[0].length) &&
       src->components[2].data == (src->components[1].data + src->components[1].length) &&
       dest->components[1].gdata == (dest->components[0].gdata + dest->components[0].length) &&
       dest->components[2].gdata == (dest->components[1].gdata + dest->components[1].length) &&
       src->components[0].length == dest->components[0].length &&
       src->components[1].length == dest->components[1].length &&
       src->components[2].length == dest->components[2].length)
    {
        SCHRO_DEBUG("consecutive %i+%i+%i", src->components[0].length, src->components[1].length, src->components[2].length);
        cudaMemcpy(dest->components[0].gdata, src->components[0].data, src->components[0].length+src->components[1].length+src->components[2].length, cudaMemcpyHostToDevice);
    }
    else
    {
        for(i=0; i<3; ++i)
        {
            if(src->components[i].data)
            {
                //cudaMemcpy(dest->components[i].gdata, src->components[i].data, src->components[i].length, cudaMemcpyHostToDevice);
                SCHRO_DEBUG("component %i: %p %i %p %i %i %i",
                i,
                dest->components[i].gdata, dest->components[i].stride, 
                             src->components[i].data, src->components[i].stride,
                             src->components[i].width*bpp, src->components[i].height
                );
                cudaMemcpy2D(dest->components[i].gdata, dest->components[i].stride, 
                             src->components[i].data, src->components[i].stride,
                             src->components[i].width*bpp, src->components[i].height,
                             cudaMemcpyHostToDevice);
            }
        }
    }
}

void schro_gpuframe_compare (SchroFrame *a, SchroFrame *b)
{
    void *temp;
    int i, bpp;
    SCHRO_ASSERT(a->format == b->format);
    /// Temp buffer
    temp = malloc(b->components[0].length);
    bpp = schro_bpp(a->format);  
    
    SCHRO_ASSERT(b->is_cuda_frame == FALSE);
    SCHRO_ASSERT(a->is_cuda_frame == TRUE);

    SCHRO_DEBUG("schro_gpuframe_compare %ix%ix%i", a->width, a->height, bpp);
    for(i=0; i<3; ++i)
    {
        int y;
        if(a->components[i].gdata == NULL)
            continue;
        SCHRO_ASSERT(a->components[i].length <= a->components[0].length);
        SCHRO_ASSERT(a->components[i].length==b->components[i].length && a->components[i].width==b->components[i].width);
        
        cudaMemcpy(temp, a->components[i].gdata, a->components[i].length, cudaMemcpyDeviceToHost);
        
        for(y=0; y<a->components[i].height; ++y)
        {
            void *bofs = b->components[i].data + y*b->components[i].stride;
            void *aofs = temp + y*a->components[i].stride;
            int diff = memcmp(bofs, aofs, a->components[i].width*bpp);
            if(diff!=0)
            {
                int x;
                for(x=0; x<a->components[i].width; ++x)
                {
                    fprintf(stderr, "%i ", ((int16_t*)aofs)[x]);
                }
                fprintf(stderr, "\n");
                for(x=0; x<a->components[i].width; ++x)
                {
                    fprintf(stderr, "%i ", ((int16_t*)bofs)[x]);
                }
                fprintf(stderr, "\n");

                SCHRO_ERROR("Error on line %i of component %i", y, i);
            }
            SCHRO_ASSERT(diff==0);
        }
    }
    
    free(temp);
}

void schro_gpuframe_zero (SchroFrame *dest)
{
    int i;

    SCHRO_ASSERT(dest->is_cuda_frame == TRUE);

    /** If the buffer is consecutive, fill it in one pass */
    if(dest->components[1].gdata == (dest->components[0].gdata + dest->components[0].length) &&
       dest->components[2].gdata == (dest->components[1].gdata + dest->components[1].length))
    {
        cudaMemset(dest->components[0].gdata, 0, dest->components[0].length+dest->components[1].length+dest->components[2].length);
    }
    else
    {
        /** Otherwise, fill per component */
        for(i=0; i<3; ++i)
        {
            if(dest->components[i].gdata)
                cudaMemset(dest->components[i].gdata, 0, dest->components[i].length);
        }
    }
}


void schro_gpuframe_upsample(SchroFrame *dst, SchroFrame *src)
{
    int i;

    SCHRO_ASSERT(dst->is_cuda_frame == TRUE);
    SCHRO_ASSERT(src->is_cuda_frame == TRUE);
    SCHRO_ASSERT(dst->width == src->width*2 && dst->height == src->height*2);
    SCHRO_ASSERT(SCHRO_FRAME_FORMAT_DEPTH(src->format) == SCHRO_FRAME_FORMAT_DEPTH_U8);
    SCHRO_ASSERT(src->format == dst->format);

    for(i=0; i<3; ++i)
    {
        uint8_t *dst_data = (uint8_t*)dst->components[i].gdata;
        int dst_stride = dst->components[i].stride;
        uint8_t *src_data = (uint8_t*)src->components[i].gdata;
        int src_stride = src->components[i].stride;
        int width = src->components[i].width;
        int height = src->components[i].height;

        cuda_upsample_horizontal(dst_data, dst_stride*2, src_data, src_stride, width, height, dst->stream);
        cuda_upsample_vertical(dst_data+dst_stride, dst_stride*2, dst_data, dst_stride*2, width*2, height, dst->stream);
    }
}

SchroUpsampledFrame *schro_upsampled_gpuframe_new(SchroVideoFormat *fmt)
{
    struct cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    SchroUpsampledFrame *rv;

    SCHRO_DEBUG("schro_upsampled_gpuframe_new");
    rv = (SchroUpsampledFrame*)malloc(sizeof(SchroUpsampledFrame));
    memset((void*)rv, 0, sizeof(SchroUpsampledFrame));

    /** Make an 8 bit texture for each component */
    cudaMallocArray(&rv->components[0], &channelDesc, fmt->width*2, fmt->height*2);
    cudaMallocArray(&rv->components[1], &channelDesc, fmt->chroma_width*2, fmt->chroma_height*2);
    cudaMallocArray(&rv->components[2], &channelDesc, fmt->chroma_width*2, fmt->chroma_height*2);

    return rv;
}

void schro_upsampled_gpuframe_upsample(SchroUpsampledFrame *rv, SchroFrame *temp_f, SchroFrame *src, SchroVideoFormat *fmt)
{
    SCHRO_ASSERT(temp_f->is_cuda_frame == TRUE);
    SCHRO_ASSERT(src->is_cuda_frame == TRUE);

    SCHRO_DEBUG("schro_upsampled_gpuframe_upsample %ix%i <- %ix%i", temp_f->width, temp_f->height, src->width, src->height);

    /** Temporary texture must have two times the size of a frame in each dimension */
    schro_gpuframe_upsample(temp_f, src);
    //schro_gpuframe_convert(temp_f, src);
    
    /** Copy data to texture */
    cudaMemcpy2DToArray(rv->components[0], 0, 0, temp_f->components[0].gdata, temp_f->components[0].stride, fmt->width*2, fmt->height*2, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(rv->components[1], 0, 0, temp_f->components[1].gdata, temp_f->components[1].stride, fmt->chroma_width*2, fmt->chroma_height*2, cudaMemcpyDeviceToDevice);    
    cudaMemcpy2DToArray(rv->components[2], 0, 0, temp_f->components[2].gdata, temp_f->components[2].stride, fmt->chroma_width*2, fmt->chroma_height*2, cudaMemcpyDeviceToDevice);
    
    //cudaMemcpy2DToArray(rv->components[0], 0, 0, src->components[0].gdata, src->components[0].stride, fmt->width, fmt->height, cudaMemcpyDeviceToDevice);
    //cudaMemcpy2DToArray(rv->components[1], 0, 0, src->components[1].gdata, src->components[1].stride, fmt->chroma_width, fmt->chroma_height, cudaMemcpyDeviceToDevice);    
    //cudaMemcpy2DToArray(rv->components[2], 0, 0, src->components[2].gdata, src->components[2].stride, fmt->chroma_width, fmt->chroma_height, cudaMemcpyDeviceToDevice);
        
}

void schro_upsampled_gpuframe_free(SchroUpsampledFrame *x)
{
    int i;

    SCHRO_DEBUG("schro_upsampled_gpuframe_free -- freed");
    for(i=0; i<3; ++i)
        cudaFreeArray(x->components[i]);
        
    //active --;
    free(x);
    //SCHRO_DEBUG("active is now %i", active);
}

#if 0
SchroFrame *
schro_frame_new_and_alloc_locked (SchroFrameFormat format, int width, int height)
{
  SchroFrame *frame = schro_frame_new();
  int bytes_pp;
  int h_shift, v_shift;
  int chroma_width;
  int chroma_height;
  
  SCHRO_ASSERT(width > 0);
  SCHRO_ASSERT(height > 0);

  /* FIXME this function allocates with cudaMallocHost() but doesn't
   * set the free() function, which means it will be freed using free(). */

  frame->format = format;
  frame->width = width;
  frame->height = height;
  frame->is_cuda_frame = FALSE;
  frame->is_cuda_shared = TRUE;

  switch (SCHRO_FRAME_FORMAT_DEPTH(format)) {
    case SCHRO_FRAME_FORMAT_DEPTH_U8:
      bytes_pp = 1;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S16:
      bytes_pp = 2;
      break;
    case SCHRO_FRAME_FORMAT_DEPTH_S32:
      bytes_pp = 4;
      break;
    default:
      SCHRO_ASSERT(0);
      bytes_pp = 0;
      break;
  }

  h_shift = SCHRO_FRAME_FORMAT_H_SHIFT(format);
  v_shift = SCHRO_FRAME_FORMAT_V_SHIFT(format);
  chroma_width = ROUND_UP_SHIFT(width, h_shift);
  chroma_height = ROUND_UP_SHIFT(height, v_shift);

  frame->components[0].width = width;
  frame->components[0].height = height;
  frame->components[0].stride = ROUND_UP_64(width * bytes_pp);
  frame->components[0].length = 
    frame->components[0].stride * frame->components[0].height;
  frame->components[0].v_shift = 0;
  frame->components[0].h_shift = 0;

  frame->components[1].width = chroma_width;
  frame->components[1].height = chroma_height;
  frame->components[1].stride = ROUND_UP_64(chroma_width * bytes_pp);
  frame->components[1].length = 
    frame->components[1].stride * frame->components[1].height;
  frame->components[1].v_shift = v_shift;
  frame->components[1].h_shift = h_shift;

  frame->components[2].width = chroma_width;
  frame->components[2].height = chroma_height;
  frame->components[2].stride = ROUND_UP_64(chroma_width * bytes_pp);
  frame->components[2].length = 
    frame->components[2].stride * frame->components[2].height;
  frame->components[2].v_shift = v_shift;
  frame->components[2].h_shift = h_shift;

  cudaMallocHost((void**)&frame->regions[0], frame->components[0].length + frame->components[1].length + frame->components[2].length);
  //frame->regions[0] = malloc (frame->components[0].length +
  //    frame->components[1].length + frame->components[2].length);

  frame->components[0].data = frame->regions[0];
  frame->components[1].data = frame->components[0].data +
    frame->components[0].length;
  frame->components[2].data = frame->components[1].data +
    frame->components[1].length;

  return frame;
}
#endif

