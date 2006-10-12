

noinst_PROGRAMS = arith arith_encode arith_decode bits bits2 \
	 encode wavelet wavelet2 tables wavelet_2d arith_speed utils \
	 obmc predict dump_packets phasecorrelation motion

TESTS = arith bits bits2 encode wavelet wavelet2 tables wavelet_2d utils


arith_CFLAGS = $(SCHRO_CFLAGS)
arith_LDADD = $(SCHRO_LIBS)
arith_SOURCES = arith.c

arith_encode_CFLAGS = $(SCHRO_CFLAGS)
arith_encode_LDADD = $(SCHRO_LIBS)
arith_encode_SOURCES = arith_encode.c

arith_decode_CFLAGS = $(SCHRO_CFLAGS)
arith_decode_LDADD = $(SCHRO_LIBS)
arith_decode_SOURCES = arith_decode.c

arith_speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
arith_speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
arith_speed_SOURCES = arith_speed.c

bits_CFLAGS = $(SCHRO_CFLAGS)
bits_LDADD = $(SCHRO_LIBS)
bits_SOURCES = bits.c

bits2_CFLAGS = $(SCHRO_CFLAGS)
bits2_LDADD = $(SCHRO_LIBS)
bits2_SOURCES = bits2.c

#codes_CFLAGS = $(SCHRO_CFLAGS)
#codes_LDADD = $(SCHRO_LIBS)
#codes_SOURCES = codes.c

dump_packets_CFLAGS = $(SCHRO_CFLAGS) $(GST_CFLAGS)
dump_packets_LDADD = $(SCHRO_LIBS) $(GST_LIBS)
dump_packets_SOURCES = dump_packets.c

encode_CFLAGS = $(SCHRO_CFLAGS)
encode_LDADD = $(SCHRO_LIBS)
encode_SOURCES = encode.c

wavelet_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
wavelet_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
wavelet_SOURCES = wavelet.c

wavelet2_CFLAGS = $(SCHRO_CFLAGS)
wavelet2_LDADD = $(SCHRO_LIBS)
wavelet2_SOURCES = wavelet2.c

wavelet_2d_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
wavelet_2d_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
wavelet_2d_SOURCES = wavelet_2d.c

tables_CFLAGS = 
tables_LDADD = -lm
tables_SOURCES = tables.c

utils_CFLAGS = $(SCHRO_CFLAGS)
utils_LDADD = $(SCHRO_LIBS)
utils_SOURCES = utils.c ../schroedinger/schroutils.c

obmc_CFLAGS = $(SCHRO_CFLAGS)
obmc_LDADD = $(SCHRO_LIBS)
obmc_SOURCES = obmc.c

predict_CFLAGS = $(SCHRO_CFLAGS)
predict_LDADD = $(SCHRO_LIBS)
predict_SOURCES = predict.c

phasecorrelation_CFLAGS = $(SCHRO_CFLAGS)
phasecorrelation_LDADD = $(SCHRO_LIBS) -lm
phasecorrelation_SOURCES = phasecorrelation.c

motion_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
motion_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
motion_SOURCES = motion.c


