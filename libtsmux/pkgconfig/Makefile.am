pcfiles = tsmux.pc
pcfiles_uninstalled = tsmux-uninstalled.pc

all-local: $(pcfiles) $(pcfiles_uninstalled)

### how to generate pc files
%.pc: %.pc
	cp $< $@
%-uninstalled.pc: %-uninstalled.pc
	cp $< $@

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = $(pcfiles)

EXTRA_DIST =							\
	tsmux.pc.in						\
	tsmux-uninstalled.pc.in

CLEANFILES = $(pcfiles) $(pcfiles_uninstalled)
