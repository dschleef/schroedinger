
AUTOMAKE_OPTIONS = foreign

SUBDIRS = doc carid gst testsuite

EXTRA_DIST = COPYING COPYING.GPL COPYING.LGPL COPYING.MIT COPYING.MPL autogen.sh libcarid.spec

pkgconfig_DATA = carid-$(CARID_MAJORMINOR).pc

carid-$(CARID_MAJORMINOR).pc: carid.pc
	cp carid.pc carid-$(CARID_MAJORMINOR).pc

carid-$(CARID_MAJORMINOR)-uninstalled.pc: carid-uninstalled.pc
	cp carid-uninstalled.pc carid-$(CARID_MAJORMINOR)-uninstalled.pc

BUILT_SOURCES=carid-$(CARID_MAJORMINOR)-uninstalled.pc

CLEANFILES = carid-$(CARID_MAJORMINOR).pc carid-$(CARID_MAJORMINOR)-uninstalled.pc

ACLOCAL_FLAGS = -I m4

