
noinst_PROGRAMS = coding_efficiency speed speed2 decode

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS)
LDADD = ../libschrotest.la $(SCHRO_LIBS) -lorc-test-0.4 $(ORC_LIBS) $(LIBM)

CODERS = arith_qm.c arith_exp.c arith_dirac.c arith_bit.c arith_dirac_byte.c arith_dirac_stats.c arith_dirac_both.c

EXTRA_DIST = arith.h

coding_efficiency_SOURCES = coding_efficiency.c $(CODERS)

speed_SOURCES = speed.c $(CODERS)

speed2_SOURCES = speed2.c $(CODERS)

decode_SOURCES = decode.c $(CODERS)

plots:
	./speed >s
	./plot_speed
	./coding_efficiency >e
	./plot_efficiency

