if HAVE_DOXYGEN
docs/doxygen: doxyfile-stamp

doxyfile-stamp: Doxyfile ../README
	cp $(abs_top_srcdir)/README $(abs_top_builddir)/docs/README.md
	$(DOXYGEN) $^
	date > $@

clean:
	-rm -rf doxygen doxyfile-stamp Doxyfile README.md
endif

EXTRA_DIST = Doxyfile.in

CLEANFILES = *~ *.bak doxyfile-stamp Doxyfile
