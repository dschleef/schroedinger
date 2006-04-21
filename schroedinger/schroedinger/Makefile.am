
pkgincludedir = $(includedir)/schro-@SCHRO_MAJORMINOR@/schro

lib_LTLIBRARIES = libschro-@SCHRO_MAJORMINOR@.la

DISTCLEANFILES = schro-stdint.h

libschro_@SCHRO_MAJORMINOR@_la_SOURCES = \
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
	schromotion.c \
	schromotion.h \
	schroparams.c \
	schroparams.h \
	schrotables.c \
	schrotables.h \
	schrowavelet.c \
	schrowavelet.h \
	schrobitstream.h

libschro_@SCHRO_MAJORMINOR@_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
libschro_@SCHRO_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libschro_@SCHRO_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(SCHRO_LIBVERSION) \
	-export-symbols-regex '^schro_'

