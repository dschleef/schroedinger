

noinst_PROGRAMS = wavelet_speed upsample_speed report \
	quantise_speed dequantise_speed

# Doesn't compile on windows
#noinst_PROGRAMS = wavelet

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = ../libschrotest.la

wavelet_speed_SOURCES = wavelet_speed.c

report_SOURCES = report.c

