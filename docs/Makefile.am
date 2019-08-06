if HAVE_DOXYGEN
docs/doxygen: doxyfile-stamp

doxyfile-stamp: Doxyfile ../README
	cp ../README README.md
	$(DOXYGEN) $^
	date > $@

clean:
	-rm -rf doxygen doxyfile-stamp Doxyfile README.md
endif

EXTRA_DIST = Doxyfile.in

CLEANFILES = *~ *.bak doxyfile-stamp Doxyfile
