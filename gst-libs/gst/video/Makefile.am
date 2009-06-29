
if ENABLE_VIDEO_COMPAT
noinst_LTLIBRARIES = libgstvideo.la
endif


libgstvideo_la_SOURCES = \
	gstbasevideocodec.c \
	gstbasevideocodec.h \
	gstbasevideoencoder.c \
	gstbasevideoencoder.h \
	gstbasevideoparse.c \
	gstbasevideoparse.h \
	gstbasevideoutils.c \
	gstbasevideoutils.h \
	gstvideocompat.c \
	gstvideocompat.h \
	gstbasevideodecoder.c \
	gstbasevideodecoder.h

fixbaseclasses = \
	-DGstBaseVideoCodec=SchroBaseVideoCodec \
	-DGstBaseVideoCodecClass=SchroBaseVideoCodecClass \
	-DGstBaseVideoEncoder=SchroBaseVideoEncoder \
	-DGstBaseVideoEncoderClass=SchroBaseVideoEncoderClass \
	-DGstBaseVideoDecoder=SchroBaseVideoDecoder \
	-DGstBaseVideoDecoderClass=SchroBaseVideoDecoderClass \
	-DGstBaseVideoParse=SchroBaseVideoParse \
	-DGstBaseVideoParseClass=SchroBaseVideoParseClass

libgstvideo_la_CFLAGS = $(GST_CFLAGS) $(SCHRO_CFLAGS)
if ENABLE_VIDEO_COMPAT
libgstvideo_la_CFLAGS += $(fixbaseclasses) -I$(top_srcdir)/gst-libs
endif

libgstvideo_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstvideo_la_LIBADD = $(GST_LIBS) $(GST_BASE_LIBS)

