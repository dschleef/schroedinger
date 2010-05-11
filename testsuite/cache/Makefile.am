

noinst_PROGRAMS = wavelet_speed upsample_speed

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = ../libschrotest.la $(SCHRO_LIBS) -lorc-test-0.4 $(ORC_LIBS) $(LIBM)

wavelet_speed_SOURCES = wavelet_speed.c



