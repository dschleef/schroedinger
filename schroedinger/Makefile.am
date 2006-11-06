
AUTOMAKE_OPTIONS = foreign

SUBDIRS = doc schroedinger gst testsuite

EXTRA_DIST = COPYING COPYING.GPL COPYING.LGPL COPYING.MIT COPYING.MPL \
	     autogen.sh libschroedinger.spec TODO NEWS

pkgconfig_DATA = schroedinger-$(SCHRO_MAJORMINOR).pc

schroedinger-$(SCHRO_MAJORMINOR).pc: schroedinger.pc
	cp schroedinger.pc schroedinger-$(SCHRO_MAJORMINOR).pc

schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc: schroedinger-uninstalled.pc
	cp schroedinger-uninstalled.pc schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

BUILT_SOURCES=schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

CLEANFILES = schroedinger-$(SCHRO_MAJORMINOR).pc schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

ACLOCAL_FLAGS = -I m4

