SUBDIRS = atomic cm collectives comm config error_vec extensions

if WITH_FORTRAN
SUBDIRS += fortran
endif

SUBDIRS += groups init lists

if WITH_MPI
SUBDIRS += mpi
endif

SUBDIRS += notifications passive segments utils

CLEANFILES = *~ *.bak *.bin fortran/*.mod

# INSTALL DATA
test_pkgdatadir=$(prefix)/tests/
test_pkgdata_DATA = $(abs_top_srcdir)/tests/README

test_defs_pkgdatadir=$(prefix)/tests/defs
test_defs_pkgdata_DATA = $(abs_top_srcdir)/tests/defs/barrier_recover.def \
		$(abs_top_srcdir)/tests/defs/barrier_recover_ping.def \
		$(abs_top_srcdir)/tests/defs/cmd_line_args.def \
		$(abs_top_srcdir)/tests/defs/default_ib.def \
		$(abs_top_srcdir)/tests/defs/default_ofi.def \
		$(abs_top_srcdir)/tests/defs/default_tcp.def \
		$(abs_top_srcdir)/tests/defs/kill_procs.def \
		$(abs_top_srcdir)/tests/defs/one_dies.def \
		$(abs_top_srcdir)/tests/defs/one_dies_passive_atomic.def \
		$(abs_top_srcdir)/tests/defs/passive_send_dead.def \
		$(abs_top_builddir)/tests/defs/active_device

# INSTALL RUN SCRIPT
test_pkgdata_SCRIPTS = $(abs_top_srcdir)/tests/runtests.sh

# COPY DEVICE DEFAULTS
if WITH_INFINIBAND
defaults:
	cp $(prefix)/tests/defs/default_ib.def $(prefix)/tests/defs/default.def
else
if WITH_OFI
defaults:
	cp $(prefix)/tests/defs/default_ofi.def $(prefix)/tests/defs/default.def
else
defaults:
	cp $(prefix)/tests/defs/default_tcp.def $(prefix)/tests/defs/default.def
endif
endif
install-data-hook:
	$(MAKE) defaults
