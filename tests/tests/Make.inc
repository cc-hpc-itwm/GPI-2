AM_CFLAGS = -g
AM_CPPFLAGS = -I$(top_builddir)/src/include -I../

LDADD = -L$(top_builddir)/src/.libs -lGPI2-dbg -lpthread
if WITH_INFINIBAND
LDADD += @ac_lib_infiniband@ -libverbs
endif

if WITH_FORTRAN
AM_FCFLAGS = -I$(top_builddir)/src @FCFLAGS_TESTS@
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

