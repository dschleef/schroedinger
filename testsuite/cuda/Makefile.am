
if HAVE_CUDA
noinst_PROGRAMS = cuda
endif

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API -I$(srcdir)/..
LDADD = $(SCHRO_LIBS) ../libschrotest.la


cuda_SOURCES = cuda.c

