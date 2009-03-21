
#include <schroedinger/schro.h>
#include <stdio.h>


int
main (int argc, char *argv[])
{
#define X(sym) printf("%s: %p\n", #sym, sym);

  /* schro.h */
  X(schro_init);

  /* schrobuffer.h */
  X(schro_buffer_new);
  X(schro_buffer_new_and_alloc);
  X(schro_buffer_new_with_data);
  X(schro_buffer_new_subbuffer);
  X(schro_buffer_dup);
  X(schro_buffer_ref);
  X(schro_buffer_unref);
  X(schro_tag_new);
  X(schro_tag_free);

  /* schrodebug.h */
  X(schro_debug_log);
  X(schro_debug_set_level);
  X(schro_debug_get_level);
  X(schro_debug_set_log_function);

  /* schrodecoder.h */
  X(schro_decoder_new);
  X(schro_decoder_free);
  X(schro_decoder_reset);
  X(schro_decoder_get_video_format);
  X(schro_decoder_add_output_picture);
  X(schro_decoder_push_ready);
  X(schro_decoder_push);
  /* deprecated.  Don't know how to check this without triggering warning */
  //X(schro_decoder_set_flushing);
  X(schro_decoder_set_picture_order);
  X(schro_decoder_push_end_of_stream);
  X(schro_decoder_pull);
  X(schro_decoder_wait);
  X(schro_decoder_set_earliest_frame);
  X(schro_decoder_set_skip_ratio);
  X(schro_decoder_get_picture_number);
  X(schro_decoder_need_output_frame);
  X(schro_decoder_autoparse_wait);
  X(schro_decoder_autoparse_push);
  X(schro_decoder_autoparse_push_end_of_sequence);
  X(schro_decoder_get_picture_tag);

  /* schroencoder.h */
  X(schro_encoder_new);
  X(schro_encoder_free);
  X(schro_encoder_get_video_format);
  X(schro_encoder_set_video_format);
  X(schro_encoder_end_of_stream);
  X(schro_encoder_push_ready);
  X(schro_encoder_push_frame);
  X(schro_encoder_push_frame_full);
  X(schro_encoder_force_sequence_header);
  X(schro_encoder_encode_auxiliary_data);
  X(schro_encoder_encode_parse_info);
  X(schro_encoder_insert_buffer);
  X(schro_encoder_frame_insert_buffer);
  X(schro_encoder_start);
  X(schro_encoder_set_packet_assembly);
  X(schro_encoder_wait);
  X(schro_encoder_pull);
  X(schro_encoder_pull_full);
  X(schro_encoder_encode_sequence_header);
  X(schro_encoder_get_n_settings);
  X(schro_encoder_get_setting_info);
  X(schro_encoder_setting_set_double);
  X(schro_encoder_setting_get_double);

  /* schroframe.h */
  X(schro_frame_new);
  X(schro_frame_new_and_alloc);
  X(schro_frame_new_from_data_I420);
  X(schro_frame_new_from_data_YV12);
  X(schro_frame_new_from_data_YUY2);
  X(schro_frame_new_from_data_UYVY);
  X(schro_frame_new_from_data_UYVY_full);
  X(schro_frame_new_from_data_AYUV);
  X(schro_frame_new_from_data_v216);
  X(schro_frame_new_from_data_v210);
  X(schro_frame_set_free_callback);
  X(schro_frame_unref);
  X(schro_frame_ref);
  X(schro_frame_dup);
  X(schro_frame_clone);
  X(schro_frame_convert);
  X(schro_frame_add);
  X(schro_frame_subtract);
  X(schro_frame_shift_left);
  X(schro_frame_shift_right);
  X(schro_frame_downsample);
  X(schro_frame_upsample_horiz);
  X(schro_frame_upsample_vert);
  X(schro_frame_calculate_average_luma);
  X(schro_frame_convert_to_444);
  X(schro_frame_md5);
  
  /* schroparse */
  X(schro_parse_decode_sequence_header);

  /* schrovideoformat.h */
  X(schro_video_format_validate);
  X(schro_video_format_set_std_video_format);
  X(schro_video_format_get_std_video_format);
  X(schro_video_format_set_std_frame_rate);
  X(schro_video_format_get_std_frame_rate);
  X(schro_video_format_set_std_aspect_ratio);
  X(schro_video_format_get_std_aspect_ratio);
  X(schro_video_format_set_std_signal_range);
  X(schro_video_format_get_std_signal_range);
  X(schro_video_format_set_std_colour_spec);
  X(schro_video_format_get_std_colour_spec);

  return 0;
}

