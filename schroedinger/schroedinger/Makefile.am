
pkgincludedir = $(includedir)/schroedinger-@SCHRO_MAJORMINOR@/schro

lib_LTLIBRARIES = libschroedinger-@SCHRO_MAJORMINOR@.la

DISTCLEANFILES = schro-stdint.h

EXTRA_DIST = schroarith-i386.c schrointernal.h

libschroedinger_@SCHRO_MAJORMINOR@_la_SOURCES = \
	schro-stdint.h \
	schro.h \
	schro.c \
	schroarith.c \
	schroarith.h \
	schrobits.c \
	schrobits.h \
	schrobitstream.h \
	schrodecoder.c \
	schrodecoder.h \
	schrodebug.c \
	schrodebug.h \
	schroencoder.c \
	schroencoder.h \
	schroframe.c \
	schroframe.h \
	schrobuffer.c \
	schrobuffer.h \
	schrometric.c \
	schrometric.h \
	schromotion.c \
	schromotion.h \
	schrooil.h \
	schrooil.c \
	schroparams.c \
	schroparams.h \
	schropredict.c \
	schropredict.h \
	schrooffsets.h \
	schrotables.c \
	schrotables.h \
	schroutils.c \
	schroutils.h \
	schrowavelet.c \
	schrowavelet.h \
	schrobitstream.h \
	schromotion.h	\
	schrointernal.h

libschroedinger_@SCHRO_MAJORMINOR@_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libschroedinger_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^schro_'

