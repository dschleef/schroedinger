
noinst_LTLIBRARIES = libopengl.la

libopengl_la_SOURCES = \
	schroopengl.c \
	schroopenglextensions.c \
	schroopenglframe.c \
	schroopenglframe_convert.c \
	schroopenglframe_pull.c \
	schroopenglframe_push.c \
	schroopenglshader.c

libopengl_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(OPENGL_CFLAGS) $(ERROR_CFLAGS) \
	-DSCHRO_ENABLE_UNSTABLE_API

libopengl_la_LIBADD = $(OPENGL_LIBS)


