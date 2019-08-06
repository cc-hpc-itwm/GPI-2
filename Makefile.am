# INCLUDES
ACLOCAL_AMFLAGS = -I config

# TARGETS
SUBDIRS = src
if ENABLE_DEBUG
SUBDIRS += tests
endif

# DOCS and TUTORIAL
if HAVE_DOXYGEN
docs :
	make -C docs docs/doxygen
endif

tutorial:
	@if [[ $(abs_top_builddir) != $(abs_top_srcdir) ]]; then\
		rsync -ar $(abs_top_srcdir)/tutorial/code/ tutorial/code/;\
	fi
	make -C tutorial/code

.PHONY : docs tutorial

# FOR DISTRIBUTION
EXTRA_DIST = config.site autogen.sh bin tutorial/* \
	tests/defs tests/runtests.sh \
	docs/Doxyfile.in docs/Makefile.am docs/Makefile.in docs/mainpage.dox docs/new_arch.png

# TESTING
check :
	cd tests/; ./runtests.sh

# INSTALL RUN SCRIPTS
bin_SCRIPTS = bin/gaspi_logger
if WITH_LOADLEVELER
install-exec-hook :
	cp $(abs_top_srcdir)/bin/gaspi_run.poe $(DESTDIR)$(bindir)/gaspi_run
	head -n 18 $(abs_top_srcdir)/bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
if WITH_SLURM
install-exec-hook :
	cp $(abs_top_srcdir)/bin/gaspi_run.slurm $(DESTDIR)$(bindir)/gaspi_run
	cp $(abs_top_srcdir)/bin/slurm.env $(DESTDIR)$(bindir)/
	head -n 18 $(abs_top_srcdir)/bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
bin_SCRIPTS += bin/ssh.spawner bin/gaspi_cleanup
install-exec-hook :
	cp $(abs_top_srcdir)/bin/gaspi_run.ssh $(DESTDIR)$(bindir)/gaspi_run
endif
endif

# INSTALL TESTS
if ENABLE_DEBUG
install-data-hook :
	mkdir -p $(prefix)/tests/
	rsync -ar --exclude='.deps' --exclude='*.o' --exclude='*.log' --exclude='machines*' $(abs_top_srcdir)/tests/ $(prefix)/tests/
	@if [[ $(abs_top_builddir) != $(abs_top_srcdir) ]]; then\
		rsync -ar --exclude='.deps' --exclude='*.o' --exclude='*.log' --exclude='machines*' $(abs_top_builddir)/tests/ $(prefix)/tests/;\
	fi
endif

# CLEAN
MOSTLYCLEANFILES = $(DX_CLEANFILES)
CLEANFILES = *~ *.bak m4/*~

clean-tutorial :
	make -C tutorial/code clean

.PHONY : clean-tutorial

if HAVE_DOXYGEN
clean-docs :
	make -C docs clean

.PHONY +: clean-docs

clean-local :
	make -C docs clean
	make -C tutorial/code clean
else
clean-local :
	make -C tutorial/code clean
endif

uninstall-hook :
	-rm $(DESTDIR)$(bindir)/gaspi_run
	-rm $(DESTDIR)$(bindir)/ssh.spawner
	-rm $(DESTDIR)$(bindir)/slurm.env
	-rm $(DESTDIR)$(bindir)/gaspi_cleanup
	-rm -fr $(prefix)/tests/
