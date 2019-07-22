if HAVE_DOXYGEN
docs/doxygen: doxyfile-stamp

doxyfile-stamp: Doxyfile
	$(DOXYGEN) $^
	date > $@

clean:
	-rm -rf doxygen doxyfile-stamp Doxyfile
endif

EXTRA_DIST = mainpage.dox

CLEANFILES = *~ *.bak doxyfile-stamp Doxyfile
