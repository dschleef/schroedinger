
noinst_PROGRAMS = opengl

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API -I$(srcdir)/..
LDADD = $(SCHRO_LIBS) ../libschrotest.la

opengl_SOURCES = \
	opengl.c \
	opengl_combine.c \
	opengl_pushpull.c \
	opengl_util.c
