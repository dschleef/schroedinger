
plugin_LTLIBRARIES = libgstschro.la


libgstschro_la_SOURCES = \
	gstschro.c \
	gstschroenc.c \
	gstschrodec.c \
	gstschroparse.c \
	gstschrotoy.c \
	gstschrofilter.c \
	gstschrodownsample.c \
	gstschroscale.c \
	gstframestore.c \
	vs_image.c \
	vs_image.h \
	vs_4tap.c \
	vs_4tap.h \
	vs_scanline.c \
	vs_scanline.h

libgstschro_la_CFLAGS = $(GST_CFLAGS) $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
libgstschro_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstschro_la_LIBADD = $(GST_LIBS) $(GST_BASE_LIBS) $(SCHRO_LIBS) $(LIBOIL_LIBS)

