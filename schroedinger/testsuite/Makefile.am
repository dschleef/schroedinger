
SUBDIRS = check perf coder installation

noinst_PROGRAMS = arith arith_encode bits bits2 \
	 encode wavelet tables wavelet_2d \
	 obmc phasecorrelation motion \
	 quant decode \
	 global_motion upsample \
	 motion2 sort filter ssim gaussian

TESTS = arith bits bits2 encode wavelet tables wavelet_2d

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

# arith_speed_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
# arith_speed_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
# arith_speed_SOURCES = arith_speed.c

# arith_fixup_shift_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
# arith_fixup_shift_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
# arith_fixup_shift_SOURCES = arith_fixup_shift.c

bits_CFLAGS = $(SCHRO_CFLAGS)
bits_LDADD = $(SCHRO_LIBS)
bits_SOURCES = bits.c

bits2_CFLAGS = $(SCHRO_CFLAGS)
bits2_LDADD = $(SCHRO_LIBS)
bits2_SOURCES = bits2.c

#codes_CFLAGS = $(SCHRO_CFLAGS)
#codes_LDADD = $(SCHRO_LIBS)
#codes_SOURCES = codes.c

encode_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
encode_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
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

motion2_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
motion2_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
motion2_SOURCES = motion2.c

quant_CFLAGS = 
quant_LDADD = -lm
quant_SOURCES = quant.c

sort_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
sort_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
sort_SOURCES = sort.c

filter_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
filter_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
filter_SOURCES = filter.c

ssim_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
ssim_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
ssim_SOURCES = ssim.c

gaussian_CFLAGS = $(SCHRO_CFLAGS) $(LIBOIL_CFLAGS)
gaussian_LDADD = $(SCHRO_LIBS) $(LIBOIL_LIBS)
gaussian_SOURCES = gaussian.c


#struct_offsets_SOURCES = struct_offsets.c


zzuf_test:
	zzuf -A -s0:10000 -r0.001 -c -v ./decode test.drc

zzuf_one:
	zzuf -A -s9 -r0.001 -c -v cat test.drc >test-n.drc

