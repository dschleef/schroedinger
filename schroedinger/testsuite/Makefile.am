
SUBDIRS = check perf

noinst_PROGRAMS = arith arith_encode bits bits2 \
	 encode wavelet tables wavelet_2d arith_speed \
	 obmc phasecorrelation motion \
	 arith_fixup_shift quant struct_offsets \
	 global_motion upsample encode_big qm \
	 coding_efficiency qm_speed exp_speed

TESTS = arith arith_fixup_shift bits bits2 encode wavelet tables wavelet_2d

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS)
LDADD = $(SCHRO_LIBS)

arith_CFLAGS = $(SCHRO_CFLAGS)
arith_LDADD = $(SCHRO_LIBS)
arith_SOURCES = arith.c

arith_encode_CFLAGS = $(SCHRO_CFLAGS)
arith_encode_LDADD = $(SCHRO_LIBS)
arith_encode_SOURCES = arith_encode.c

# arith_decode_CFLAGS = $(SCHRO_CFLAGS)
# arith_decode_LDADD = $(SCHRO_LIBS)
# arith_decode_SOURCES = arith_decode.c

arith_speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
arith_speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
arith_speed_SOURCES = arith_speed.c

arith_fixup_shift_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
arith_fixup_shift_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
arith_fixup_shift_SOURCES = arith_fixup_shift.c

bits_CFLAGS = $(SCHRO_CFLAGS)
bits_LDADD = $(SCHRO_LIBS)
bits_SOURCES = bits.c

bits2_CFLAGS = $(SCHRO_CFLAGS)
bits2_LDADD = $(SCHRO_LIBS)
bits2_SOURCES = bits2.c

#codes_CFLAGS = $(SCHRO_CFLAGS)
#codes_LDADD = $(SCHRO_LIBS)
#codes_SOURCES = codes.c

encode_CFLAGS = $(SCHRO_CFLAGS)
encode_LDADD = $(SCHRO_LIBS)
encode_SOURCES = encode.c

encode_big_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
encode_big_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
encode_big_SOURCES = encode_big.c

wavelet_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
wavelet_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
wavelet_SOURCES = wavelet.c

#wavelet2_CFLAGS = $(SCHRO_CFLAGS)
#wavelet2_LDADD = $(SCHRO_LIBS)
#wavelet2_SOURCES = wavelet2.c

wavelet_2d_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
wavelet_2d_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
wavelet_2d_SOURCES = wavelet_2d.c

tables_CFLAGS = 
tables_LDADD = -lm
tables_SOURCES = tables.c

obmc_CFLAGS = $(SCHRO_CFLAGS)
obmc_LDADD = $(SCHRO_LIBS)
obmc_SOURCES = obmc.c

phasecorrelation_CFLAGS = $(SCHRO_CFLAGS)
phasecorrelation_LDADD = $(SCHRO_LIBS) -lm
phasecorrelation_SOURCES = phasecorrelation.c

motion_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
motion_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
motion_SOURCES = motion.c

quant_CFLAGS = 
quant_LDADD = -lm
quant_SOURCES = quant.c

qm_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
qm_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
qm_SOURCES = qm.c arith_qm.c arith_qm.h

qm_speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
qm_speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
qm_speed_SOURCES = qm_speed.c arith_qm.c arith_qm.h

coding_efficiency_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
coding_efficiency_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
coding_efficiency_SOURCES = coding_efficiency.c arith_qm.c arith_qm.h arith_exp.c

exp_speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
exp_speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
exp_speed_SOURCES = exp_speed.c arith_exp.c arith_exp.h


struct_offsets_SOURCES = struct_offsets.c

