
SUBDIRS = . perf coder installation prototype streams tables2 dirac \
	  manual cache cuda opengl

buildnoinstdir = $(libdir)
buildnoinst_LTLIBRARIES = libschrotest.la


noinst_PROGRAMS = arith arith_encode \
	 wavelet_2d \
	 phasecorrelation motion \
	 quant decode decode_coded_order decode_loop divide \
	 upsample \
	 motion2 sort fft \
	 wavelet_noise_gain \
	 wavelet_max_gain \
	 downsample taps ramp \
	 colorspace filter_coeff utils extension

if HAVE_ORCC
noinst_PROGRAMS += orc_test
endif

if ENABLE_ENCODER
noinst_PROGRAMS += encode ssim wavelet_gain \
	wavelet_gain_curve perceptual_weight perceptual_curves \
	wavelet_noise_2d unpack abi
endif

noinst_HEADERS = common.h

if BUILD_BROKEN_TESTS
noinst_PROGRAMS += bits bits2 bits3
endif

TESTS = arith encode wavelet_2d unpack downsample

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = libschrotest.la $(SCHRO_LIBS) -lorc-test-0.4 $(ORC_LIBS) $(LIBM)

libschrotest_la_SOURCES = common.c
libschrotest_la_LIBADD = $(SCHRO_LIBS) $(ORC_LIBS) -lorc-test-0.4 $(LIBM)

install-buildnoinstLTLIBRARIES:

#arith_CFLAGS = $(SCHRO_CFLAGS)
#arith_LDADD = $(SCHRO_LIBS)
arith_SOURCES = arith.c

#arith_encode_CFLAGS = $(SCHRO_CFLAGS)
#arith_encode_LDADD = $(SCHRO_LIBS)
arith_encode_SOURCES = arith_encode.c

# arith_decode_CFLAGS = $(SCHRO_CFLAGS)
# arith_decode_LDADD = $(SCHRO_LIBS)
# arith_decode_SOURCES = arith_decode.c

# arith_fixup_shift_CFLAGS = $(SCHRO_CFLAGS)
# arith_fixup_shift_LDADD = $(SCHRO_LIBS)
# arith_fixup_shift_SOURCES = arith_fixup_shift.c

#bits_CFLAGS = $(SCHRO_CFLAGS)
#bits_LDADD = $(SCHRO_LIBS)
bits_SOURCES = bits.c

#bits2_CFLAGS = $(SCHRO_CFLAGS)
#bits2_LDADD = $(SCHRO_LIBS)
bits2_SOURCES = bits2.c

#codes_CFLAGS = $(SCHRO_CFLAGS)
#codes_LDADD = $(SCHRO_LIBS)
#codes_SOURCES = codes.c

#encode_CFLAGS = $(SCHRO_CFLAGS)
#encode_LDADD = $(SCHRO_LIBS)
encode_SOURCES = encode.c

#encode_big_CFLAGS = $(SCHRO_CFLAGS)
#encode_big_LDADD = $(SCHRO_LIBS)
#encode_big_SOURCES = encode_big.c

#wavelet_CFLAGS = $(SCHRO_CFLAGS)
#wavelet_LDADD = $(SCHRO_LIBS)
#wavelet_SOURCES = wavelet.c

wavelet_gain_SOURCES = wavelet_gain.c

wavelet_noise_gain_SOURCES = wavelet_noise_gain.c

wavelet_gain_curve_SOURCES = wavelet_gain_curve.c

#wavelet2_CFLAGS = $(SCHRO_CFLAGS)
#wavelet2_LDADD = $(SCHRO_LIBS)
#wavelet2_SOURCES = wavelet2.c

#wavelet_2d_CFLAGS = $(SCHRO_CFLAGS)
#wavelet_2d_LDADD = $(SCHRO_LIBS)
wavelet_2d_SOURCES = wavelet_2d.c
nodist_wavelet_2d_SOURCES = $(top_srcdir)/schroedinger/tmp-orc.c

wavelet_max_gain_SOURCES = wavelet_max_gain.c

wavelet_noise_2d_SOURCES = wavelet_noise_2d.c

#phasecorrelation_CFLAGS = $(SCHRO_CFLAGS)
#phasecorrelation_LDADD = $(SCHRO_LIBS) -lm
phasecorrelation_SOURCES = phasecorrelation.c

#motion_CFLAGS = $(SCHRO_CFLAGS)
#motion_LDADD = $(SCHRO_LIBS)
motion_SOURCES = motion.c

#motion2_CFLAGS = $(SCHRO_CFLAGS)
#motion2_LDADD = $(SCHRO_LIBS)
motion2_SOURCES = motion2.c

#quant_CFLAGS = 
#quant_LDADD = -lm
quant_SOURCES = quant.c
nodist_quant_SOURCES = $(top_srcdir)/schroedinger/tmp-orc.c

#sort_CFLAGS = $(SCHRO_CFLAGS)
#sort_LDADD = $(SCHRO_LIBS)
sort_SOURCES = sort.c

#ssim_CFLAGS = $(SCHRO_CFLAGS)
#ssim_LDADD = $(SCHRO_LIBS)
ssim_SOURCES = ssim.c

#fft_CFLAGS = $(SCHRO_CFLAGS)
#fft_LDADD = $(SCHRO_LIBS)
fft_SOURCES = fft.c

perceptual_weight_SOURCES = perceptual_weight.c

#perceptual_curves_CFLAGS = $(SCHRO_CFLAGS)
#perceptual_curves_LDADD = $(SCHRO_LIBS)
perceptual_curves_SOURCES = perceptual_curves.c

orc_test_CFLAGS = $(ORC_CFLAGS)
orc_test_LDADD = $(ORC_LIBS) -lorc-test-0.4


orc_test.c: $(top_srcdir)/schroedinger/schroorc.orc
	$(ORCC) --test -o $@ $<

zzuf_test:
	zzuf -A -s0:10000 -r0.00001 -c -v ./decode test_stream.drc

test-%.drc:
	num=`echo $@ | sed 's/test-\(.*\)\.drc/\1/'`; \
	zzuf -A -s$$num -r0.00001 -c -v cat test_stream.drc >$@

zzuf_files:
	n=0; \
	while [ $$n -lt 100 ] ; do \
	  zzuf -A -s$$n -r0.00001 -c -v cat test_stream.drc >test-$$n.drc; \
	  n=$$(($$n+1)); \
	done

