# INCLUDES
ACLOCAL_AMFLAGS = -I config

# TARGETS
SUBDIRS = src

if ENABLE_DEBUG
SUBDIRS += tests
endif

# DOCS and TUTORIAL
if HAVE_DOXYGEN
docs:
	make -C docs docs/doxygen
endif

tutorial:
	@if [[ $(abs_top_builddir) != $(abs_top_srcdir) ]]; then\
		rsync -ar --exclude='$(abs_top_srcdir)/tutorial/code/Makefile' $(abs_top_srcdir)/tutorial/code/ $(abs_top_builddir)/tutorial/code/;\
	fi
	make -C tutorial/code

.PHONY: docs tutorial defaults

# FOR DISTRIBUTION
EXTRA_DIST = configure.ac.in config.site autogen.sh bin tutorial/* \
	tests/defs tests/runtests.sh \
	docs/Doxyfile.in docs/Makefile.am docs/Makefile.in docs/mainpage.dox docs/new_arch.png

# TESTING
if ENABLE_DEBUG
check:
	cd tests/; ./runtests.sh $(GPI2_RUNTEST_OPTIONS)
endif

# INSTALL RUN SCRIPTS
if WITH_LOADLEVELER
install-exec-hook:
	cp $(abs_top_srcdir)/bin/gaspi_run.poe $(DESTDIR)$(bindir)/gaspi_run
	head -n 18 $(abs_top_srcdir)/bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
if WITH_SLURM
install-exec-hook:
	cp $(abs_top_srcdir)/bin/gaspi_run.slurm $(DESTDIR)$(bindir)/gaspi_run
	cp $(abs_top_srcdir)/bin/slurm.env $(DESTDIR)$(bindir)/
	head -n 18 $(abs_top_srcdir)/bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
bin_SCRIPTS = bin/ssh.spawner bin/gaspi_cleanup
install-exec-hook:
	cp $(abs_top_srcdir)/bin/gaspi_run.ssh $(DESTDIR)$(bindir)/gaspi_run
endif
endif

# CLEAN
MOSTLYCLEANFILES = $(DX_CLEANFILES)
CLEANFILES = *~ *.bak m4/*~

clean-tutorial:
	make -C tutorial/code clean

.PHONY: clean-tutorial

if HAVE_DOXYGEN
clean-docs:
	make -C docs clean

.PHONY +: clean-docs

clean-local:
	make -C docs clean
	make -C tutorial/code clean
else
clean-local:
	make -C tutorial/code clean
endif

uninstall-hook:
	-rm $(DESTDIR)$(bindir)/gaspi_run
	-rm $(DESTDIR)$(bindir)/ssh.spawner
	-rm $(DESTDIR)$(bindir)/slurm.env
	-rm $(DESTDIR)$(bindir)/gaspi_cleanup
	-rm -fr $(prefix)/tests/
