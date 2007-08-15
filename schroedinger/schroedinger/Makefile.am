
pkgincludedir = $(includedir)/schroedinger-@SCHRO_MAJORMINOR@/schroedinger

lib_LTLIBRARIES = libschroedinger-@SCHRO_MAJORMINOR@.la

DISTCLEANFILES = schro-stdint.h

EXTRA_DIST = schroarith-i386.c

pkginclude_HEADERS = \
	schro.h \
	schrobuffer.h \
	schrodecoder.h \
	schroencoder.h \
	schroengine.h \
	schrofilter.h \
	schroframe.h \
	schro-stdint.h \
	schroarith.h \
	schroanalysis.h \
	schroasync.h \
	schrobits.h \
	schrobitstream.h \
	schrodebug.h \
	schrohistogram.h \
	schrometric.h \
	schromotion.h \
	schroparams.h \
	schrophasecorrelation.h \
	schropredict.h \
	schroqueue.h \
	schrossim.h \
	schrotables.h \
	schroutils.h \
	schrowavelet.h \
	schrobitstream.h \
	schromotion.h

noinst_HEADERS = \
	schrooffsets.h \
	schrooil.h

libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES = \
	schro.c \
	schroanalysis.c \
	schroarith.c \
	schroasync-pthread.c \
	schrobits.c \
	schrodecoder.c \
	schrodebug.c \
	schroencoder.c \
	schroengine.c \
	schrofilter.c \
	schroframe.c \
	schrohistogram.c \
	schrobuffer.c \
	schrometric.c \
	schromotion.c \
	schrooil.c \
	schroparams.c \
	schrophasecorrelation.c \
	schropredict.c \
	schroquantiser.c \
	schroqueue.c \
	schrossim.c \
	schrotables.c \
	schroutils.c \
	schrowavelet.c

libschroedinger_@SCHRO_MAJORMINOR@_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(ERROR_CFLAGS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^(oil|schro)_'

