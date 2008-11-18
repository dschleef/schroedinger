
plugin_LTLIBRARIES = libgstschro.la


libgstschro_la_SOURCES = \
	gstschro.c \
	gstschrodec.c \
	gstschroparse.c \
	gstschroscale.c \
	gstschrocolorspace.c \
	gstlogoinsert.c \
	gstbasevideodecoder.c \
	gstbasevideodecoder.h \
	gstbasevideoencoder.c \
	gstbasevideoencoder.h \
	gstbasevideoparse.c \
	gstbasevideoparse.h \
	gstbasevideoutils.c \
	gstbasevideoutils.h \
	gstframestore.c \
	gstschroutils.c \
	gstschroutils.h \
	gstadapter2.c \
	gstadapter2.h

if ENABLE_ENCODER
libgstschro_la_SOURCES += \
	gstschroenc.c \
	gstschrodownsample.c \
	gstschrofilter.c
endif

libgstschro_la_CFLAGS = $(GST_CFLAGS) $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
libgstschro_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstschro_la_LIBADD = $(GST_LIBS) $(GST_BASE_LIBS) $(SCHRO_LIBS) $(LIBOIL_LIBS) -lpng

