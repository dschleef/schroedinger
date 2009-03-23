AUTOMAKE_OPTIONS = foreign

SUBDIRS = schroedinger doc tools testsuite

DISTCHECK_CONFIGURE_FLAGS=--enable-gtk-doc

if HAVE_GST
SUBDIRS += gst-libs/gst/video gst
endif

DIST_SUBDIRS = schroedinger doc tools gst testsuite gst-libs/gst/video

EXTRA_DIST = COPYING COPYING.GPL COPYING.LGPL COPYING.MIT COPYING.MPL \
	     autogen.sh schroedinger.spec TODO NEWS

pkgconfig_DATA = schroedinger-$(SCHRO_MAJORMINOR).pc

schroedinger-$(SCHRO_MAJORMINOR).pc: schroedinger.pc
	cp schroedinger.pc schroedinger-$(SCHRO_MAJORMINOR).pc

schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc: schroedinger-uninstalled.pc
	cp schroedinger-uninstalled.pc schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

BUILT_SOURCES=schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

CLEANFILES = schroedinger-$(SCHRO_MAJORMINOR).pc schroedinger-$(SCHRO_MAJORMINOR)-uninstalled.pc

ACLOCAL_FLAGS = -I m4

