AC_PREREQ([2.58])
AC_INIT(carid,0.1.0.1)

AS_NANO(CARID_CVS=no,CARID_CVS=no)

AC_CANONICAL_HOST([])
AM_INIT_AUTOMAKE(1.6)
AM_MAINTAINER_MODE

CARID_MAJORMINOR=0.1
AC_SUBST(CARID_MAJORMINOR)

AM_CONFIG_HEADER(config.h)

dnl CURRENT, REVISION, AGE
dnl - library source changed -> increment REVISION
dnl - interfaces added/removed/changed -> increment CURRENT, REVISION = 0
dnl - interfaces added -> increment AGE
dnl - interfaces removed -> AGE = 0
CARID_LIBVERSION="0:0:0"
AC_SUBST(CARID_LIBVERSION)
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL
AM_PROG_AS

dnl allow for different autotools
AS_AUTOTOOLS_ALTERNATE

AC_CONFIG_SRCDIR([carid/carid.h])

ACLOCAL_AMFLAGS="-I m4 $ACLOCAL_AMFLAGS"
AC_SUBST(ACLOCAL_AMFLAGS)

dnl set up gettext
dnl the version check needs to stay here because autopoint greps for it
#AM_GNU_GETTEXT_VERSION([0.11.5])
#AM_GNU_GETTEXT([external])
#GST_GETTEXT([carid-$CARID_MAJORMINOR])

dnl liboil is required
PKG_CHECK_MODULES(LIBOIL, liboil-0.3 >= 0.3.2, HAVE_LIBOIL=yes, HAVE_LIBOIL=no)
if test "x${HAVE_LIBOIL}" != xyes ; then
  AC_ERROR([liboil-0.3 is required])
fi

##################################################
# Check for gtk-doc.
##################################################

GTK_DOC_CHECK([1.0])

AX_CREATE_STDINT_H([carid/carid-stdint.h])

AS_COMPILER_FLAG(-Wall, CARID_CFLAGS="$CARID_CFLAGS -Wall")
if test "x$CARID_CVS" = "xyes"
then
  AS_COMPILER_FLAG(-Werror, CARID_CFLAGS="$CARID_CFLAGS -Werror")
fi

CARID_CFLAGS="$CARID_CFLAGS -I\$(top_srcdir)"
AC_SUBST(CARID_CFLAGS)

CARID_LIBS="\$(top_builddir)/carid/libcarid-$CARID_MAJORMINOR.la"
AC_SUBST(CARID_LIBS)

########################
# GStreamer stuff
########################

GSTREAMER_MAJORMINOR=0.10

dnl *** check for arguments to configure ***

GST_ARG_DEBUG
GST_ARG_PROFILING
GST_ARG_VALGRIND
GST_ARG_GCOV

GST_ARG_EXAMPLES

GST_ARG_WITH_PKG_CONFIG_PATH
GST_ARG_WITH_PACKAGE_NAME([GStreamer Base Plug-ins])
GST_ARG_WITH_PACKAGE_ORIGIN

dnl checks for gstreamer
dnl uninstalled is selected preferentially -- see pkg-config(1)
GST_MAJORMINOR=0.10
GST_REQ=0.9.7
GST_CHECK_GST($GST_MAJORMINOR, [$GST_REQ])
GST_CHECK_GST_BASE($GST_MAJORMINOR, [$GST_REQ])
GST_CHECK_GST_GDP($GST_MAJORMINOR, [$GST_REQ])
GST_CHECK_GST_CONTROLLER($GST_MAJORMINOR, [$GST_REQ])
GST_CHECK_GST_CHECK($GST_MAJORMINOR, [$GST_REQ], no)

dnl FIXME: get rid of this by making sure gstreamer-check brings it in
dnl check for "check", unit testing library/header
AM_PATH_CHECK(0.9.2, HAVE_CHECK=yes, HAVE_CHECK=no)
AM_CONDITIONAL(HAVE_CHECK, test "x$HAVE_CHECK" = "xyes")

dnl *** set variables based on configure arguments ***

dnl set license and copyright notice
GST_LICENSE="LGPL"
AC_DEFINE_UNQUOTED(GST_LICENSE, "$GST_LICENSE", [GStreamer license])
AC_SUBST(GST_LICENSE)

dnl set location of plugin directory
GST_SET_PLUGINDIR

dnl define an ERROR_CFLAGS Makefile variable
GST_SET_ERROR_CFLAGS($GST_CVS)

dnl define correct level for debugging messages
GST_SET_LEVEL_DEFAULT($GST_CVS)

dnl *** finalize CFLAGS, LDFLAGS, LIBS

dnl Overview:
dnl GST_OPTION_CFLAGS:  common flags for profiling, debugging, errors, ...
dnl GST_*:              flags shared by built objects to link against GStreamer
dnl GST_ALL_LDFLAGS:    linker flags shared by all
dnl GST_LIB_LDFLAGS:    additional linker flags for all libaries
dnl GST_LT_LDFLAGS:     library versioning of our libraries
dnl GST_PLUGIN_LDFLAGS: flags to be used for all plugins

dnl GST_OPTION_CFLAGS
if test "x$USE_DEBUG" = xyes; then
   PROFILE_CFLAGS="-g"
fi
AC_SUBST(PROFILE_CFLAGS)

DEPRECATED_CFLAGS="-DGST_DISABLE_DEPRECATED"
AC_SUBST(DEPRECATED_CFLAGS)

dnl every flag in GST_OPTION_CFLAGS can be overridden at make time
GST_OPTION_CFLAGS="\$(ERROR_CFLAGS) \$(DEBUG_CFLAGS) \$(PROFILE_CFLAGS) \$(GCOV_CFLAGS) \$(OPT_CFLAGS) \$(DEPRECATED_CFLAGS)"
AC_SUBST(GST_OPTION_CFLAGS)

dnl our libraries need to be versioned correctly
AC_SUBST(GST_LT_LDFLAGS)

dnl FIXME: do we want to rename to GST_ALL_* ?
dnl prefer internal headers to already installed ones
dnl also add builddir include for enumtypes and marshal
dnl add GST_OPTION_CFLAGS, but overridable
GST_CFLAGS="-I\$(top_srcdir)/gst-libs -I\$(top_builddir)/gst-libs $GST_CFLAGS \$(GST_OPTION_CFLAGS)"
AC_SUBST(GST_CFLAGS)
AC_SUBST(GST_LIBS)

dnl LDFLAGS really should only contain flags, not libs - they get added before
dnl whatevertarget_LIBS and -L flags here affect the rest of the linking
GST_ALL_LDFLAGS="-no-undefined"
AC_SUBST(GST_ALL_LDFLAGS)

dnl GST_LIB_LDFLAGS
dnl linker flags shared by all libraries
dnl LDFLAGS modifier defining exported symbols from built libraries
GST_LIB_LDFLAGS="-export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*"
AC_SUBST(GST_LIB_LDFLAGS)

dnl this really should only contain flags, not libs - they get added before
dnl whatevertarget_LIBS and -L flags here affect the rest of the linking
GST_PLUGIN_LDFLAGS="-module -avoid-version -export-symbols-regex '^[_]*gst_plugin_desc\$\$' $GST_ALL_LDFLAGS"
AC_SUBST(GST_PLUGIN_LDFLAGS)

PKG_CHECK_MODULES(GST, gstreamer-0.10 gstreamer-base-0.10,
  HAVE_GST=yes, HAVE_GST=no)
dnl Give error and exit if we don't have gstreamer
if test "x$HAVE_GST" = "xno"; then
  AC_MSG_ERROR([you need gstreamer development packages installed !])
fi
AC_SUBST(GST_CFLAGS)
AC_SUBST(GST_LIBS)

plugindir="\$(libdir)/gstreamer-$GST_MAJORMINOR"
AC_SUBST(plugindir)

GST_ALL_LDFLAGS="-no-undefined"
AC_SUBST(GST_ALL_LDFLAGS)

GST_PLUGIN_LDFLAGS="-module -avoid-version -export-symbols-regex '^[_]*gst_plugin_desc\$\$' $GST_ALL_LDFLAGS"
AC_SUBST(GST_PLUGIN_LDFLAGS)

pkgconfigdir="\$(libdir)/pkgconfig"
AC_SUBST(pkgconfigdir)

AC_CONFIG_FILES([
Makefile 
doc/Makefile
carid/Makefile
gst/Makefile
testsuite/Makefile
carid-uninstalled.pc
carid.pc
libcarid.spec
])
AC_OUTPUT


