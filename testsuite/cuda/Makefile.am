
noinst_PROGRAMS = cuda

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API -I$(srcdir)/..
LDADD = $(SCHRO_LIBS) ../libschrotest.la


cuda_SOURCES = cuda.c

