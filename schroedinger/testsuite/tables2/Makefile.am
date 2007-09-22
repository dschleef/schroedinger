
noinst_PROGRAMS = \
	lowdelay_tables \
	quant_tables \
	unpack_tables \
	wavelet_tables

AM_CFLAGS = $(SCHRO_CFLAGS) $(ERROR_CFLAGS) $(LIBOIL_CFLAGS) -I..
LDADD = $(SCHRO_LIBS) ../libschrotest.la

lowdelay_tables_SOURCES = lowdelay_tables.c

wavelet_tables_SOURCES = wavelet_tables.c

quant_tables_SOURCES = quant_tables.c

unpack_tables_SOURCES = unpack_tables.c


