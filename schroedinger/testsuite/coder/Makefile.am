
noinst_PROGRAMS = coding_efficiency speed

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS)
LDADD = $(SCHRO_LIBS)

coding_efficiency_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
coding_efficiency_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
coding_efficiency_SOURCES = coding_efficiency.c arith_qm.c arith_qm.h arith_exp.c arith_dirac.c arith_bit.c arith_dirac_byte.c

speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
speed_SOURCES = speed.c arith_qm.c arith_qm.h arith_exp.c arith_dirac.c arith_bit.c arith_dirac_byte.c


