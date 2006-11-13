lib_LTLIBRARIES = libtsmux.la

libtsmux_la_SOURCES =		\
		tsmux.c		\
		tsmuxstream.c

libtsmux_la_CFLAGS = $(GLIB_CFLAGS) -I\$(top_srcdir) -I\$(top_builddir)
libtsmux_la_LIBADD = $(GLIB_LIBS)
libtsmux_la_LDFLAGS = 

# headers we need but don't want installed
noinst_HEADERS = crc.h

# headers we want installed
tsmuxincludedir = $(includedir)/tsmux
tsmuxinclude_HEADERS = tsmux.h tsmuxcommon.h tsmuxstream.h
