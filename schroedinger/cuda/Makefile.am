.cu.lo:
	$(top_builddir)/cudalt.py $@ $(NVCC) -c $(NVCCFLAGS) $<

noinst_LTLIBRARIES = libdcuda.la

CCLD = g++

#pkginclude_HEADERS = 

#noinst_HEADERS = 

libdcuda_la_SOURCES = init.cu cudawavelet.cu \
        haar_iwt.cu haar_iiwt.cu \
        format/frame.cu upsample.cu motion_tex.cu \
        generated/iiwt5_3.cu generated/iwt5_3.cu generated/iiwt9_3.cu generated/iwt9_3.cu  \
        generated/iiwt13_5.cu generated/iwt13_5.cu generated/iiwtfidelity.cu generated/iwtfidelity.cu \
        generated/iiwt9_7.cu generated/iwt9_7.cu

#        iiwt5_3.cu iwt5_3.cu iiwt9_3.cu iwt9_3.cu iiwt13_5.cu iwt13_5.cu \
#        iiwtfidelity.cu iwtfidelity.cu iiwt9_7.cu iwt9_7.cu

libdcuda_la_CFLAGS = $(SCHRO_CFLAGS) $(CUDA_CFLAGS) $(ERROR_CFLAGS)
libdcuda_la_LIBADD = $(CUDA_LIBS)
libdcuda_la_LDFLAGS = 
#	-no-undefined \
#	-version-info $(SCHRO_LIBVERSION) \
#	-export-symbols-regex '^(oil|schro)_'

