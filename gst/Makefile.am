
plugindir = $(libdir)/gstreamer-1.0

plugin_LTLIBRARIES = libgstschro.la

noinst_HEADERS = \
	gstschroutils.h

libgstschro_la_SOURCES = \
	gstschro.c \
	gstschrodec.c \
	gstschroenc.c \
	gstschroutils.c
libgstschro_la_CFLAGS = \
	$(GST_CFLAGS) \
	$(SCHRO_CFLAGS)
libgstschro_la_LIBADD = \
	$(GST_LIBS) -lgstvideo-1.0 \
	$(SCHRO_LIBS)
libgstschro2_la_LDFLAGS = -module -avoid-version -no-undefined
libgstschro2_la_LIBTOOLFLAGS = --tag=disable-static



