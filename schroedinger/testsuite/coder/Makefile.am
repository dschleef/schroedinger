
noinst_PROGRAMS = coding_efficiency speed speed2 decode

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS)
LDADD = $(SCHRO_LIBS)

CODERS = arith_qm.c arith_exp.c arith_dirac.c arith_bit.c arith_dirac_byte.c arith_dirac_stats.c arith_dirac_both.c

EXTRA_DIST = arith.h

coding_efficiency_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
coding_efficiency_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
coding_efficiency_SOURCES = coding_efficiency.c $(CODERS)

speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
speed_SOURCES = speed.c $(CODERS)

speed2_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
speed2_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
speed2_SOURCES = speed2.c $(CODERS)

decode_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
decode_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
decode_SOURCES = decode.c $(CODERS)

plots:
	./speed >s
	./plot_speed
	./coding_efficiency >e
	./plot_efficiency

