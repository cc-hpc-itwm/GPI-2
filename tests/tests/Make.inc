AM_CFLAGS = -g
AM_CPPFLAGS = -I$(abs_top_srcdir)/src/include -I$(abs_top_srcdir)/tests/tests

LDADD = $(abs_top_builddir)/src/.libs/libGPI2-dbg.la

LDFLAGS = -pthread
if WITH_INFINIBAND
LDFLAGS += @ac_lib_infiniband@
endif

if WITH_FORTRAN
AM_FCFLAGS = -I$(abs_top_builddir)/src @FCFLAGS_TESTS@
endif

if WITH_MPI
AM_CFLAGS += -DGPI2_WITH_MPI
AM_CFLAGS += @ac_inc_mpi@
LDADD += @ac_lib_mpi@ @mpi_extra_flags@
if WITH_FORTRAN
AM_FCFLAGS += -DGPI2_WITH_MPI
endif
endif

AM_CXXFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS)

# aux TESTS
auxSRCS = ../test_utils.c ../test_utils.h

# Installation path
test_pkglibexecdir=$(prefix)/tests/bin/
