
plugin_LTLIBRARIES = libgstschro.la


libgstschro_la_SOURCES = \
	gstschro.c \
	gstschrodec.c \
	gstschroparse.c \
	gstschroscale.c \
	gstbasevideocoder.c \
	gstbasevideocoder.h \
	gstframestore.c \
	gstschrocolorspace.c

if ENABLE_ENCODER
libgstschro_la_SOURCES += \
	gstschroenc.c \
	gstschrodownsample.c \
	gstschrofilter.c
endif

libgstschro_la_CFLAGS = $(GST_CFLAGS) $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
libgstschro_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstschro_la_LIBADD = $(GST_LIBS) $(GST_BASE_LIBS) $(SCHRO_LIBS) $(LIBOIL_LIBS)

