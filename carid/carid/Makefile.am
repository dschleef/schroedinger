
pkgincludedir = $(includedir)/carid-@CARID_MAJORMINOR@/carid

lib_LTLIBRARIES = libcarid-@CARID_MAJORMINOR@.la

libcarid_@CARID_MAJORMINOR@_la_SOURCES = \
	carid-stdint.h \
	carid.h \
	carid.c \
	caridarith.c \
	caridarith.h \
	caridbits.c \
	caridbits.h \
	cariddecoder.c \
	cariddecoder.h \
	caridencoder.c \
	caridencoder.h \
	caridbuffer.c \
	caridbuffer.h \
	caridwavelet.c \
	caridwavelet.h

libcarid_@CARID_MAJORMINOR@_la_CFLAGS = $(CARID_CFLAGS) $(LIBOIL_CFLAGS)
libcarid_@CARID_MAJORMINOR@_la_LIBADD = $(LIBOIL_LIBS)
libcarid_@CARID_MAJORMINOR@_la_LDFLAGS = \
	-no-undefined \
	-version-info $(CARID_LIBVERSION) \
	-export-symbols-regex '^carid_'

