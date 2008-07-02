
#ifndef __SCHRO_VIRT_FRAME_H__
#define __SCHRO_VIRT_FRAME_H__

#include <schroedinger/schroutils.h>
#include <schroedinger/schroframe.h>

SCHRO_BEGIN_DECLS

SchroFrame *schro_frame_new_virtual (SchroMemoryDomain *domain,
    SchroFrameFormat format, int width, int height);

void *schro_virt_frame_get_line (SchroFrame *frame, int component, int i);
void schro_virt_frame_render_line (SchroFrame *frame, void *dest,
    int component, int i);

void schro_virt_frame_render (SchroFrame *frame, SchroFrame *dest);

SchroFrame *schro_virt_frame_new_horiz_downsample (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_vert_downsample (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_unpack (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_YUY2 (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_UYVY (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_AYUV (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_color_matrix (SchroFrame *vf);

SCHRO_END_DECLS

#endif

