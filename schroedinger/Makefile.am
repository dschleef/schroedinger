
if HAVE_CUDA
SUBDIRS = cuda
endif

if HAVE_OPENGL
SUBDIRS = opengl
endif

pkgincludedir = $(includedir)/schroedinger-@SCHRO_MAJORMINOR@/schroedinger

lib_LTLIBRARIES = libschroedinger-@SCHRO_MAJORMINOR@.la

BUILT_SOURCES = \
	schroversion.h

DISTCLEANFILES = schroversion.h

EXTRA_DIST = schroarith-i386.c \
	schroasync-none.c \
	schroasync-pthread.c \
	schroasync-gthread.c \
	schroasync-win32.c

pkginclude_HEADERS = \
	schro.h \
	schrobuffer.h \
	schrobufferlist.h \
	schrocog.h \
	schrodecoder.h \
	schroencoder.h \
	schroengine.h \
	schrofilter.h \
	schroframe.h \
	schro-stdint.h \
	schroarith.h \
	schroanalysis.h \
	schroasync.h \
	schrobitstream.h \
	schrodebug.h \
	schrodomain.h \
	schrofft.h \
	schrohistogram.h \
	schrolimits.h \
	schrolist.h \
	schrometric.h \
	schromotion.h \
	schromotionest.h \
	schropack.h \
	schroparams.h \
	schroparse.h \
	schrophasecorrelation.h \
	schroqueue.h \
	schrossim.h \
	schrotables.h \
	schrounpack.h \
	schroutils.h \
	schrovideoformat.h \
	schrovirtframe.h \
	schrowavelet.h \
	schrobitstream.h \
	schromotion.h

nodist_pkginclude_HEADERS = \
	schroversion.h

noinst_HEADERS = \
	schrooil.h \
	schroorc.h \
	schrogpuframe.h

if HAVE_CUDA
cuda_sources = \
	schrocuda.c \
	schrocuda.h \
	schrogpuframe.c \
	schrogpumotion.c \
	dummy.cc
cuda_libs = cuda/libdcuda.la
endif

if HAVE_OPENGL
opengl_libs = opengl/libopengl.la
endif

libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES = \
	schro.c \
	schroarith.c \
	schrocog.c \
	schrodecoder.c \
	schrodebug.c \
	schrodomain.c \
	schroframe.c \
	schrohistogram.c \
	schrobuffer.c \
	schrobufferlist.c \
	schrolowdelay.c \
	schromotion.c \
	schrooil.c \
	schroparams.c \
	schroparse.c \
	schroroughmotion.c \
	schroqueue.c \
	schrotables.c \
	schrotablesinit.c \
	schrounpack.c \
	schrounpacktables.c \
	schroutils.c \
	schrovideoformat.c \
	schrovirtframe.c \
	$(cuda_sources) \
	schroasync-@THREAD_IMPL@.c

if ENABLE_ENCODER
libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES += \
	schroanalysis.c \
	schroencoder.c \
	schroengine.c \
	schrofft.c \
	schrofilter.c \
	schroglobalest.c \
	schrolist.c \
	schrometric.c \
	schromotionest.c \
	schropack.c \
	schrophasecorrelation.c \
	schroquantiser.c \
	schrossim.c \
	schrowavelettables.c
endif

if HAVE_ORC
libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES += \
	schroorc.c schrowaveletorc.c
else
libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES += schrowavelet.c
endif



libschroedinger_@SCHRO_MAJORMINOR@_la_CFLAGS = \
	$(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(ORC_CFLAGS) $(ERROR_CFLAGS) \
	-DSCHRO_ENABLE_UNSTABLE_API \
	$(GTHREAD_CFLAGS) \
	$(CUDA_CFLAGS) -I$(srcdir)/cuda -I$(srcdir)/opengl

libschroedinger_@SCHRO_MAJORMINOR@_la_LIBADD = \
	$(cuda_libs) \
	$(CUDA_LIBS) \
	$(opengl_libs) \
	$(OPENGL_LIBS) \
	$(LIBOIL_LIBS) \
	$(ORC_LIBS) \
	$(PTHREAD_LIBS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^schro_'

