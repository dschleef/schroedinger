
plugin_LTLIBRARIES = libgstschro.la


libgstschro_la_SOURCES = \
	gstschro.c \
	gstschrodec.c \
	gstschroparse.c \
	gstschroutils.c \
	gstschroutils.h

if ENABLE_ENCODER
libgstschro_la_SOURCES += \
	gstschroenc.c
endif

if ENABLE_VIDEO_COMPAT
fixbaseclasses = \
	-DGstBaseVideoCodec=SchroBaseVideoCodec \
	-DGstBaseVideoCodecClass=SchroBaseVideoCodecClass \
	-DGstBaseVideoEncoder=SchroBaseVideoEncoder \
	-DGstBaseVideoEncoderClass=SchroBaseVideoEncoderClass \
	-DGstBaseVideoDecoder=SchroBaseVideoDecoder \
	-DGstBaseVideoDecoderClass=SchroBaseVideoDecoderClass \
	-DGstBaseVideoParse=SchroBaseVideoParse \
	-DGstBaseVideoParseClass=SchroBaseVideoParseClass

basevideolibs = $(top_builddir)/gst-libs/gst/video/libgstvideo.la
endif

libgstschro_la_CFLAGS = $(GST_CFLAGS) $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(fixbaseclasses) -I$(top_srcdir)/gst-libs
libgstschro_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstschro_la_LIBADD = $(basevideolibs) $(GST_LIBS) $(GST_BASE_LIBS) $(SCHRO_LIBS) $(LIBOIL_LIBS)

