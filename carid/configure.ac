AC_PREREQ([2.58])
AC_INIT(carid,0.1.0.1)

AS_NANO(CARID_CVS=no,CARID_CVS=yes)

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
CARID_LIBVERSION="1:0:1"
AC_SUBST(CARID_LIBVERSION)
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL
AM_PROG_AS

AC_CONFIG_SRCDIR([carid/carid.h])

ACLOCAL_AMFLAGS="-I m4 $ACLOCAL_AMFLAGS"
AC_SUBST(ACLOCAL_AMFLAGS)

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

CARID_LIBS="\$(top_builddir)/carid/carid-$CARID_MAJORMINOR.la"
AC_SUBST(CARID_LIBS)

pkgconfigdir="\$(libdir)/pkgconfig"
AC_SUBST(pkgconfigdir)

AC_CONFIG_FILES([
Makefile 
doc/Makefile
carid/Makefile
carid-uninstalled.pc
carid.pc
])
AC_OUTPUT



