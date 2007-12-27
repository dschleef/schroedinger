
EXTRA_DIST = drc-to-png

if HAVE_GST
noinst_PROGRAMS = dump_packets dump_gop
else
noinst_PROGRAMS =
endif


AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) $(GST_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = $(SCHRO_LIBS) $(GST_LIBS)

