.cu.lo:
	$(top_builddir)/cudalt.py $@ $(NVCC) -c $(NVCCFLAGS) $<

noinst_LTLIBRARIES = libdcuda.la

CCLD = g++

#pkginclude_HEADERS = 

#noinst_HEADERS = 

libdcuda_la_SOURCES = \
	dummy.c \
	init.cu cudawavelet.cu \
        haar_iwt.cu haar_iiwt.cu \
        frame.cu upsample.cu motion_tex.cu \
        iiwt5_3.cu iwt5_3.cu iiwt9_3.cu iwt9_3.cu iiwt13_5.cu iwt13_5.cu \
        iiwtfidelity.cu iwtfidelity.cu iiwt9_7.cu iwt9_7.cu

libdcuda_la_CFLAGS = $(SCHRO_CFLAGS) $(CUDA_CFLAGS) $(ERROR_CFLAGS)
libdcuda_la_LIBADD = $(CUDA_LIBS)
libdcuda_la_LDFLAGS = 
#	-no-undefined \
#	-version-info $(SCHRO_LIBVERSION) \
#	-export-symbols-regex '^(oil|schro)_'

