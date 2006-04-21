
AUTOMAKE_OPTIONS = foreign

SUBDIRS = doc schro gst testsuite

EXTRA_DIST = COPYING COPYING.GPL COPYING.LGPL COPYING.MIT COPYING.MPL autogen.sh libschro.spec

pkgconfig_DATA = schro-$(SCHRO_MAJORMINOR).pc

schro-$(SCHRO_MAJORMINOR).pc: schro.pc
	cp schro.pc schro-$(SCHRO_MAJORMINOR).pc

schro-$(SCHRO_MAJORMINOR)-uninstalled.pc: schro-uninstalled.pc
	cp schro-uninstalled.pc schro-$(SCHRO_MAJORMINOR)-uninstalled.pc

BUILT_SOURCES=schro-$(SCHRO_MAJORMINOR)-uninstalled.pc

CLEANFILES = schro-$(SCHRO_MAJORMINOR).pc schro-$(SCHRO_MAJORMINOR)-uninstalled.pc

ACLOCAL_FLAGS = -I m4

