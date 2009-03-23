AC_PREREQ([2.58])
dnl PLEASE remove the nano when doing a release
AC_INIT(schroedinger,1.0.6)

AS_VERSION

AS_NANO(SCHRO_CVS=no,SCHRO_CVS=yes)

AC_CANONICAL_HOST([])
AM_INIT_AUTOMAKE(1.6)

SCHRO_MAJORMINOR=1.0
AC_SUBST(SCHRO_MAJORMINOR)

AM_CONFIG_HEADER(config.h)

dnl CURRENT, REVISION, AGE
dnl - library source changed -> increment REVISION
dnl - interfaces added/removed/changed -> increment CURRENT, REVISION = 0
dnl - interfaces added -> increment AGE
dnl - interfaces removed -> AGE = 0
SCHRO_LIBVERSION="2:0:2"
AC_SUBST(SCHRO_LIBVERSION)
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL
AC_PROG_CXX
AM_PROG_AS
AM_PROG_CC_C_O
AC_SYS_LARGEFILE

dnl allow for different autotools
AS_AUTOTOOLS_ALTERNATE

AC_CONFIG_SRCDIR([schroedinger/schro.h])

AC_CONFIG_MACRO_DIR([m4])
ACLOCAL_AMFLAGS="-I m4 $ACLOCAL_AMFLAGS"
AC_SUBST(ACLOCAL_AMFLAGS)

dnl set up gettext
dnl the version check needs to stay here because autopoint greps for it
#AM_GNU_GETTEXT_VERSION([0.11.5])
#AM_GNU_GETTEXT([external])

AC_CHECK_LIBM
AC_SUBST(LIBM)

dnl liboil is required
OIL_VER="0.3.16"
PKG_CHECK_MODULES(LIBOIL, liboil-0.3 >= $OIL_VER, HAVE_LIBOIL=yes, HAVE_LIBOIL=no)
if test "x${HAVE_LIBOIL}" != xyes ; then
  AC_ERROR([liboil-0.3 >= $OIL_VER is required])
fi
SCHRO_PKG_DEPS="liboil-0.3 >= $OIL_VER"

AC_ARG_ENABLE(orc, AC_HELP_STRING([--enable-orc],
              [enable orc support (default=disabled)]))
if test "x$enable_orc" = xyes
then
  ORC_VER="0.3.13"
  PKG_CHECK_MODULES(ORC, orc-0.3 >= $ORC_VER, HAVE_ORC=yes, HAVE_ORC=no)
else
  HAVE_ORC=no
fi
if test "x${HAVE_ORC}" = xyes ; then
  AC_DEFINE(HAVE_ORC, 1, [Whether or not orc is enabled])
  SCHRO_PKG_DEPS="$SCHRO_PKG_DEPS orc-0.3 >= $ORC_VER"
fi
AM_CONDITIONAL(HAVE_ORC, test "x$HAVE_ORC" = "xyes")

AC_ARG_WITH([thread],
    AS_HELP_STRING([--with-thread=IMPL],[use threading library {auto,pthread,gthread,win32,none} [[default=auto]]]),
    [],
    [with_thread=auto])

AC_MSG_CHECKING([threading implementation to use])
if test "x${with_thread}" = "xauto" ; then
  case $host_os in
    mingw* | pw32*)
      with_thread=win32
      ;;
    *)
      with_thread=pthread
      ;;
  esac
fi
AC_MSG_RESULT([$with_thread])

THREAD_IMPL=${with_thread}
AC_SUBST(THREAD_IMPL)

if test "x${with_thread}" = "xpthread" ; then
  case "$host_os" in 
    *)
      PTHREAD_CFLAGS=
      PTHREAD_LIBS=-lpthread
      ;;
  esac
  AC_SUBST(PTHREAD_CFLAGS)
  AC_SUBST(PTHREAD_LIBS)
fi

if test "x${with_thread}" = "xgthread" ; then
  PKG_CHECK_MODULES(GTHREAD, gthread-2.0, HAVE_GTHREAD=yes, HAVE_GTHREAD=no)
  AC_SUBST(GTHREAD_LIBS)
  AC_SUBST(GTHREAD_CFLAGS)
  SCHRO_PKG_DEPS="$SCHRO_PKG_DEPS gthread-2.0"
fi

AC_ARG_WITH([opengl],
    AS_HELP_STRING([--with-opengl=PATH],[prefix where OpenGL is installed [[default=no]]]),
    [],
    [with_opengl=no])

AC_ARG_ENABLE(encoder, AC_HELP_STRING([--disable-encoder],
              [disable encoder (default=enabled)]))
AM_CONDITIONAL(ENABLE_ENCODER, test "x$enable_encoder" != "xno")
if [ test "x$enable_encoder" != "xno" ] ; then
  AC_DEFINE(ENABLE_ENCODER, 1, [Enable encoder])
fi
AC_ARG_ENABLE(motion-ref, AC_HELP_STRING([--disable-motion-ref],
              [disable motion reference code (default=enabled)]))
AM_CONDITIONAL(ENABLE_MOTION_REF, test "x$enable_motion_ref" != "xno")
if [ test "x$enable_motion_ref" != "xno" ] ; then
  AC_DEFINE(ENABLE_MOTION_REF, 1, [Enable motion reference code])
fi

##################################################
# Check for gtk-doc.
##################################################

if test "x$cross_compiling" = "xyes" ; then
  enable_gtk_doc=no
fi
GTK_DOC_CHECK([1.0])

AS_COMPILER_FLAG(-Wall, SCHRO_CFLAGS="$SCHRO_CFLAGS -Wall")
if test "x$SCHRO_CVS" = "xyes"
then
  AS_COMPILER_FLAG(-Werror, SCHRO_CFLAGS="$SCHRO_CFLAGS -Werror")
fi

AS_COMPILER_FLAG(-O3, CFLAGS=`echo $CFLAGS | sed s/O2/O3/`)

# Enable severe pain
#AS_COMPILER_FLAGS(COMPANY_CFLAGS,"-Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wold-style-definition -Wdeclaration-after-statement -Wmissing-declarations -Wmissing-prototypes -Wredundant-decls -Wmissing-noreturn -Wshadow -Wpointer-arith -Wcast-align -Wwrite-strings -Winline -Wformat-nonliteral -Wformat-security -Wswitch-enum -Wswitch-default")
#COMPANY_CFLAGS="-Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wold-style-definition -Wdeclaration-after-statement -Wmissing-declarations -Wmissing-prototypes -Wredundant-decls -Wmissing-noreturn -Wshadow -Wpointer-arith -Wcast-align -Wwrite-strings -Winline -Wformat-nonliteral -Wformat-security -Wswitch-enum -Wswitch-default"
#SCHRO_CFLAGS="$SCHRO_CFLAGS $COMPANY_CFLAGS"

SCHRO_CFLAGS="$SCHRO_CFLAGS -I\$(top_srcdir)"
AC_SUBST(SCHRO_CFLAGS)

########################
# GStreamer stuff
########################

GST_MAJORMINOR=0.10

dnl *** check for arguments to configure ***

if test "x$with_opengl" = xyes; then
  GST_GL_DEP="gst-plugins-gl-0.10"
fi

AC_ARG_ENABLE(gstreamer, AC_HELP_STRING([--disable-gstreamer],
              [disable gstreamer support (default=enabled)]))
if test "x$enable_gstreamer" != xno
then
  PKG_CHECK_MODULES(GST, gstreamer-0.10 >= 0.10.19 gstreamer-base-0.10 >= 0.10.19 gstreamer-video-0.10 >= 0.10.19 ${GST_GL_DEP},
      HAVE_GST=yes, HAVE_GST=no)
else
  HAVE_GST=no
fi

AM_CONDITIONAL(HAVE_GST, test "x$HAVE_GST" = "xyes")
AC_SUBST(GST_CFLAGS)
AC_SUBST(GST_LIBS)

plugindir="\$(libdir)/gstreamer-$GST_MAJORMINOR"
AC_SUBST(plugindir)

GST_ALL_LDFLAGS="-no-undefined"
AC_SUBST(GST_ALL_LDFLAGS)

GST_PLUGIN_LDFLAGS="-module -avoid-version -export-symbols-regex '^[_]*gst_plugin_desc\$\$' $GST_ALL_LDFLAGS"
AC_SUBST(GST_PLUGIN_LDFLAGS)

AC_DEFINE(GST_PACKAGE_ORIGIN, "unknown", [Package origin in GStreamer plugin])

AC_ARG_ENABLE(video-compat, AC_HELP_STRING([--disable-video-compat],
    [disable gstvideo compatibility (default=enabled)]))
AM_CONDITIONAL(ENABLE_VIDEO_COMPAT, test "x$enable_video_compat" != "xno")


########################
# Check for CUDA
########################
AC_ARG_WITH([cuda],
    AS_HELP_STRING([--with-cuda=PATH],[prefix where cuda is installed [[default=no]]]),
    [],
    [with_cuda=no])
if test "x$with_cuda" = xyes; then
  AC_PATH_PROG([NVCC], [nvcc], no, [$PATH:/usr/local/cuda/bin])
  if test "x$NVCC" = xno ; then
    AC_MSG_FAILURE(["--with-cuda was give, but nvcc was not found"])
  fi
  with_cuda=`echo $NVCC|sed 's,bin/nvcc,,'`
fi

if test "x$with_cuda" != xno; then
  CUDA_CFLAGS="-I$with_cuda/include"
  CUDA_LIBS="-L$with_cuda/lib -lcudart"
  NVCC="$with_cuda/bin/nvcc"
  NVCCFLAGS="-O2 -use_fast_math"
  HAVE_CUDA=yes
  AC_DEFINE(HAVE_CUDA, 1, [Compile in CUDA rendering backend])
else
  CUDA_CFLAGS=
  CUDA_LIBS=
  NVCC=
  NVCCFLAGS=
  HAVE_CUDA=no
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(NVCCFLAGS)
AM_CONDITIONAL(HAVE_CUDA, test "x$HAVE_CUDA" = "xyes")

########################
# Check for OpenGL/GLEW
########################
if test "x$with_opengl" = xyes; then
  save_CPPFLAGS="$CPPFLAGS"
  save_LIBS="$LIBS"
  CPPFLAGS="$CPPFLAGS $X_CFLAGS"
  LIBS="$LIBS $X_LIBS"
  AC_CHECK_LIB(GL, glTexImage2D, [
    OPENGL_CFLAGS=
    OPENGL_LIBS=-lGL
    HAVE_OPENGL=yes
    AC_DEFINE(HAVE_OPENGL, 1, [Compile in OpenGL rendering backend])
  ], [HAVE_OPENGL=no])
  if test "x${HAVE_OPENGL}" = xyes ; then
    AC_CHECK_LIB(GLEW, glewInit, [
      OPENGL_CFLAGS=
      OPENGL_LIBS="-lGL -lGLEW"
      HAVE_LIBGLEW=yes
    ], [HAVE_LIBGLEW=no])
    if test "x${HAVE_LIBGLEW}" != xyes ; then
      AC_ERROR([libglew1.5 is required])
    fi
  fi
  CPPFLAGS="$save_CPPFLAGS"
  LIBS="$save_LIBS"
fi
AC_SUBST(OPENGL_CFLAGS)
AC_SUBST(OPENGL_LIBS)
AM_CONDITIONAL(HAVE_OPENGL, test "x$HAVE_OPENGL" = "xyes")


AM_CONDITIONAL(BUILD_BROKEN_TESTS, false)

SCHRO_LIBS="\$(top_builddir)/schroedinger/libschroedinger-$SCHRO_MAJORMINOR.la"
# hack around ld bug
SCHRO_LIBS="$SCHRO_LIBS $LIBOIL_LIBS $CUDA_LIBS $OPENGL_LIBS $PTHREAD_LIBS $GTHREAD_LIBS $LIBM"
#SCHRO_LIBS="$SCHRO_LIBS -lprofiler"
AC_SUBST(SCHRO_LIBS)

AC_SUBST(SCHRO_PKG_DEPS)
AS_HOST_DEFINES()
	
pkgconfigdir="\$(libdir)/pkgconfig"
AC_SUBST(pkgconfigdir)

AC_CONFIG_FILES([
Makefile 
doc/Makefile
schroedinger/Makefile
schroedinger/cuda/Makefile
schroedinger/opengl/Makefile
gst-libs/gst/video/Makefile
gst/Makefile
testsuite/Makefile
testsuite/cache/Makefile
testsuite/cuda/Makefile
testsuite/perf/Makefile
testsuite/coder/Makefile
testsuite/dirac/Makefile
testsuite/installation/Makefile
testsuite/manual/Makefile
testsuite/opengl/Makefile
testsuite/prototype/Makefile
testsuite/streams/Makefile
testsuite/tables2/Makefile
tools/Makefile
schroedinger/schroversion.h
schroedinger-uninstalled.pc
schroedinger.pc
schroedinger.spec
])
AC_OUTPUT

