
noinst_PROGRAMS = \
	lowdelay_tables \
	quant_tables \
	unpack_tables

if ENABLE_ENCODER
noinst_PROGRAMS += \
	wavelet_tables
endif

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(ORC_CFLAGS) -I$(srcdir)/.. \
	    -DSCHRO_ENABLE_UNSTABLE_API
LDADD = ../libschrotest.la $(SCHRO_LIBS) $(LIBM)

lowdelay_tables_SOURCES = lowdelay_tables.c

wavelet_tables_SOURCES = wavelet_tables.c

quant_tables_SOURCES = quant_tables.c

unpack_tables_SOURCES = unpack_tables.c


