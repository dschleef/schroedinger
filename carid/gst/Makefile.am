
plugin_LTLIBRARIES = libgstcarid.la


libgstcarid_la_SOURCES = gstcarid.c gstcaridenc.c gstcariddec.c
libgstcarid_la_CFLAGS = $(GST_CFLAGS) $(GST_BASE_CFLAGS) $(CARID_CFLAGS) $(LIBOIL_CFLAGS)
libgstcarid_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)
libgstcarid_la_LIBADD = $(GST_LIBS) $(GST_BASE_LIBS) $(CARID_LIBS) $(LIBOIL_LIBS)

