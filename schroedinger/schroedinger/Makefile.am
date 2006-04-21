
pkgincludedir = $(includedir)/carid-@CARID_MAJORMINOR@/carid

lib_LTLIBRARIES = libcarid-@CARID_MAJORMINOR@.la

DISTCLEANFILES = carid-stdint.h

libcarid_@CARID_MAJORMINOR@_la_SOURCES = \
	carid-stdint.h \
	carid.h \
	carid.c \
	caridarith.c \
	caridarith.h \
	caridbits.c \
	caridbits.h \
	caridbitstream.h \
	cariddecoder.c \
	cariddecoder.h \
	cariddebug.c \
	cariddebug.h \
	caridencoder.c \
	caridencoder.h \
	caridframe.c \
	caridframe.h \
	caridbuffer.c \
	caridbuffer.h \
	caridmotion.c \
	caridmotion.h \
	caridparams.c \
	caridparams.h \
	caridtables.c \
	caridtables.h \
	caridwavelet.c \
	caridwavelet.h \
	caridbitstream.h

libcarid_@CARID_MAJORMINOR@_la_CFLAGS = $(CARID_CFLAGS) $(LIBOIL_CFLAGS)
libcarid_@CARID_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libcarid_@CARID_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(CARID_LIBVERSION) \
	-export-symbols-regex '^carid_'

