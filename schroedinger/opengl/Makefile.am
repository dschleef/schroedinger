
noinst_LTLIBRARIES = libopengl.la

libopengl_la_SOURCES = \
	schroopengl.c \
	schroopenglframe.c \
	schroopenglframe_combine.c \
	schroopenglframe_convert.c \
	schroopenglframe_pull.c \
	schroopenglframe_push.c \
	schroopenglmotion.c \
	schroopenglshader.c \
	schroopenglwavelet.c

libopengl_la_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS) $(OPENGL_CFLAGS) $(ERROR_CFLAGS) \
	-DSCHRO_ENABLE_UNSTABLE_API -D_GNU_SOURCE

libopengl_la_LIBADD = $(OPENGL_LIBS)


