AC_INIT

dnl the fourth (nano) number should be 0 for a release, 1 for trunk,
dnl and 2... for a prerelease
dnl releases only do Wall, cvs and prerelease does Werror too
AS_VERSION(libtsmux, TSMUX_VERSION, 0, 1, 1, 0,
    TSMUX_TRUNK="no", TSMUX_TRUNK="yes")

dnl AM_MAINTAINER_MODE provides the option to enable maintainer mode
AM_MAINTAINER_MODE

AM_INIT_AUTOMAKE($PACKAGE, $VERSION)

AM_CONFIG_HEADER(config.h)

dnl make aclocal work in maintainer mode
AC_SUBST(ACLOCAL_AMFLAGS, "-I m4")

dnl check for tools
AC_PROG_CC
AC_PROG_LIBTOOL

dnl Check for pkgconfig first
AC_CHECK_PROG(HAVE_PKGCONFIG, pkg-config, yes, no)

dnl Give error and exit if we don't have pkgconfig
if test "x$HAVE_PKGCONFIG" = "xno"; then
  AC_MSG_ERROR(you need to have pkgconfig installed !)
fi

dnl Grab Glib flags separately
PKG_CHECK_MODULES(GLIB, glib-2.0 gthread-2.0)
AC_SUBST(GLIB_LIBS)
AC_SUBST(GLIB_CFLAGS)

AC_OUTPUT(
Makefile
libtsmux.spec
tsmux/Makefile
tests/Makefile
common/Makefile
common/m4/Makefile
pkgconfig/Makefile
pkgconfig/tsmux.pc
pkgconfig/tsmux-uninstalled.pc
)
