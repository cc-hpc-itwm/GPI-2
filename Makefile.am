# INCLUDES
include $(top_srcdir)/amdoxygen.am
ACLOCAL_AMFLAGS = -I m4

# TARGETS
SUBDIRS = src tests docs

# FOR DISTRIBUTION
EXTRA_DIST = autogen.sh tutorial $(DX_CONFIG) bin/*

# TESTING
check :
	cd tests/; ./runtests.sh

# RUN SCRIPTS
bin_SCRIPTS = bin/gaspi_logger
if WITH_LOADLEVER
install-exec-hook:
	cp bin/gaspi_run.poe $(DESTDIR)$(bindir)/gaspi_run
	head -n 18 bin/gaspi_cleanup > $(DESTDIR)$(bindir)/gaspi_cleanup
	chmod +x $(DESTDIR)$(bindir)/gaspi_cleanup
else
bin_SCRIPTS += bin/ssh.spawner bin/gaspi_cleanup
install-exec-hook:
	cp bin/gaspi_run.ssh $(DESTDIR)$(bindir)/gaspi_run
endif

# CLEAN
MOSTLYCLEANFILES = $(DX_CLEANFILES)
CLEANFILES = *~ *.bak m4/*~
