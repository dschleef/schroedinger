noinst_PROGRAMS = testmux write-empty

testmux_SOURCES = 		\
		testmux.c

write_empty_SOURCES = 		\
		write-empty.c

testmux_CFLAGS = $(GLIB_CFLAGS) -I$(top_srcdir)
testmux_LDADD = $(GLIB_LIBS) $(top_builddir)/tsmux/libtsmux.la
testmux_LDFLAGS = 

write_empty_CFLAGS = $(GLIB_CFLAGS) -I$(top_srcdir)
write_empty_LDADD = $(GLIB_LIBS) $(top_builddir)/tsmux/libtsmux.la
write_empty_LDFLAGS = 
