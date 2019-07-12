if HAVE_DOXYGEN
docs/doxygen: doxyfile-stamp

doxyfile-stamp: Doxyfile
	$(DOXYGEN) $^
	date > $@

all-local: doxyfile-stamp

clean-local:
	-rm -rf docs/doxygen
endif

EXTRA_DIST = mainpage.dox

CLEANFILES = *~ *.bak doxyfile-stamp
