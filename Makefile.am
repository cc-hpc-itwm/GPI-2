# INCLUDES
ACLOCAL_AMFLAGS = -I config

# TARGETS
SUBDIRS = src tests

docs :
	make -C docs docs/doxygen

tutorial:
	make -C tutorial/code

.PHONY : docs tutorial

# FOR DISTRIBUTION
EXTRA_DIST = autogen.sh tutorial bin/*

# TESTING
check :
	cd tests/; ./runtests.sh

# INSTALL RUN SCRIPTS
bin_SCRIPTS = bin/gaspi_logger
if WITH_LOADLEVER
install-exec-hook :
	cp bin/gaspi_run.poe $(DESTDIR)$(bindir)/gaspi_run
	head -n 18 bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
if WITH_SLURM
install-exec-hook :
	cp bin/gaspi_run.slurm $(DESTDIR)$(bindir)/gaspi_run
	cp bin/slurm.env $(DESTDIR)$(bindir)/
	head -n 18 bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
bin_SCRIPTS += bin/ssh.spawner bin/gaspi_cleanup
install-exec-hook :
	cp bin/gaspi_run.ssh $(DESTDIR)$(bindir)/gaspi_run
endif
endif

# INSTALL TESTS
install-data-hook:
	mkdir -p $(prefix)/tests/
	rsync -ar --exclude='.deps' --exclude='*.o' tests/ $(prefix)/tests/

# CLEAN
MOSTLYCLEANFILES = $(DX_CLEANFILES)
CLEANFILES = *~ *.bak m4/*~

clean-docs :
	make -C docs clean

clean-tutorial :
	make -C tutorial/code clean

.PHONY +: clean-docs clean-tutorial

clean-local :
	make -C docs clean
	make -C tutorial/code clean
