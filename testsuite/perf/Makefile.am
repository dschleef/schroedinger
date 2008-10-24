

noinst_PROGRAMS = wavelet_speed upsample_speed report \
	quantise_speed dequantise_speed

# Doesn't compile on windows
#noinst_PROGRAMS = wavelet

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)

wavelet_speed_SOURCES = wavelet_speed.c

report_SOURCES = report.c

