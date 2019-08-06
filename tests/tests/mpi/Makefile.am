include ../Make.inc

## MPI
mpi_tests = ../../bin/mpi_allreduce.bin \
      ../../bin/mpi_barrier.bin \
      ../../bin/with_mpi.bin

______bin_mpi_allreduce_bin_SOURCES = mpi_allreduce.c   $(auxSRCS)
______bin_mpi_barrier_bin_SOURCES = mpi_barrier.c $(auxSRCS)
______bin_with_mpi_bin_SOURCES = with_mpi.c  $(auxSRCS)

noinst_PROGRAMS = $(mpi_tests)
