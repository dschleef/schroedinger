
pkgincludedir = $(includedir)/schroedinger-@SCHRO_MAJORMINOR@/schroedinger

lib_LTLIBRARIES = libschroedinger-@SCHRO_MAJORMINOR@.la

DISTCLEANFILES = schro-stdint.h

EXTRA_DIST = schroarith-i386.c schrointernal.h

pkginclude_HEADERS = \
	schro.h \
	schrobuffer.h \
	schrodecoder.h \
	schroencoder.h \
	schroengine.h \
	schroframe.h \
	schro-stdint.h \
	schroarith.h \
	schroanalysis.h \
	schroasync.h \
	schrobits.h \
	schrobitstream.h \
	schrodebug.h \
	schrometric.h \
	schromotion.h \
	schrooil.h \
	schroparams.h \
	schropredict.h \
	schrotables.h \
	schroutils.h \
	schrowavelet.h \
	schrobitstream.h \
	schromotion.h

noinst_HEADERS = \
	schrooffsets.h \
	schrointernal.h

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
	schroframe.c \
	schrobuffer.c \
	schrometric.c \
	schromotion.c \
	schrooil.c \
	schroparams.c \
	schropredict.c \
	schrotables.c \
	schroutils.c \
	schrowavelet.c

libschroedinger_@SCHRO_MAJORMINOR@_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(ERROR_CFLAGS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^(oil|schro)_'

