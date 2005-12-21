
AUTOMAKE_OPTIONS = foreign

SUBDIRS = doc carid gst

EXTRA_DIST = COPYING autogen.sh gtk-doc.make

pkgconfig_DATA = carid-$(CARID_MAJORMINOR).pc

carid-$(CARID_MAJORMINOR).pc: carid.pc
	cp carid.pc carid-$(CARID_MAJORMINOR).pc

carid-$(CARID_MAJORMINOR)-uninstalled.pc: carid-uninstalled.pc
	cp carid-uninstalled.pc carid-$(CARID_MAJORMINOR)-uninstalled.pc

BUILT_SOURCES=carid-$(CARID_MAJORMINOR)-uninstalled.pc

CLEANFILES = carid-$(CARID_MAJORMINOR).pc carid-$(CARID_MAJORMINOR)-uninstalled.pc

ACLOCAL_FLAGS = -I m4

