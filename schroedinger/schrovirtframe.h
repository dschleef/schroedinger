
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

SchroFrame *schro_virt_frame_new_horiz_downsample (SchroFrame *vf, int cosite);
SchroFrame *schro_virt_frame_new_vert_downsample (SchroFrame *vf, int cosite);
SchroFrame *schro_virt_frame_new_vert_resample (SchroFrame *vf, int height);
SchroFrame *schro_virt_frame_new_horiz_resample (SchroFrame *vf, int width);
SchroFrame *schro_virt_frame_new_unpack (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_YUY2 (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_UYVY (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_AYUV (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_v216 (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_v210 (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_RGB (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_color_matrix (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_subsample (SchroFrame *vf, SchroFrameFormat format);

SchroFrame *schro_virt_frame_new_horiz_downsample_take (SchroFrame *vf, int cosite);
SchroFrame *schro_virt_frame_new_vert_downsample_take (SchroFrame *vf, int cosite);
SchroFrame *schro_virt_frame_new_vert_resample_take (SchroFrame *vf, int height);
SchroFrame *schro_virt_frame_new_horiz_resample_take (SchroFrame *vf, int width);
SchroFrame *schro_virt_frame_new_unpack_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_YUY2_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_UYVY_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_AYUV_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_v216_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_v210_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_pack_RGB_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_color_matrix_take (SchroFrame *vf);
SchroFrame *schro_virt_frame_new_subsample_take (SchroFrame *vf, SchroFrameFormat format);

SchroFrame * schro_virt_frame_new_convert_u8_take (SchroFrame *vf);
SchroFrame * schro_virt_frame_new_convert_s16_take (SchroFrame *vf);
SchroFrame * schro_virt_frame_new_crop_take (SchroFrame *vf, int width, int height);
SchroFrame * schro_virt_frame_new_edgeextend_take (SchroFrame *vf, int width, int height);

SCHRO_END_DECLS

#endif

