
EXTRA_DIST = drc-to-png

if HAVE_GST
noinst_PROGRAMS = dump_packets
else
noinst_PROGRAMS =
endif


AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) $(GST_CFLAGS)
LDADD = $(SCHRO_LIBS) $(GST_LIBS)

