
ORC_SOURCE=schroorc
include $(top_srcdir)/orc.mak

noinst_PROGRAMS = wavelet_speed upsample_speed report \
	quantise_speed dequantise_speed unpack_speed

# Doesn't compile on windows
#noinst_PROGRAMS = wavelet

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = ../libschrotest.la $(SCHRO_LIBS) -lorc-test-0.4 $(ORC_LIBS) $(LIBM)

wavelet_speed_SOURCES = wavelet_speed.c

dequantise_speed_SOURCES = \
	dequantise_speed.c
nodist_dequantise_speed_SOURCES = \
	$(ORC_NODIST_SOURCES)

report_SOURCES = report.c

