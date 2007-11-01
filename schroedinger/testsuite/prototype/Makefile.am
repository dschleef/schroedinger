
if HAVE_I386
noinst_PROGRAMS = arith_speed
endif

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = $(SCHRO_LIBS)

arith_speed_SOURCES = arith_speed.c

