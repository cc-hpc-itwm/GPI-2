if WITH_INFINIBAND
if HAVE_CINTEL
if WITH_MPI
all:
	$(MAKE) -C basic NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C ring-1d/gaspi NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C ring-1d/mpi
	$(MAKE) -C advanced NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
clean:
	$(MAKE) -C basic clean
	$(MAKE) -C ring-1d/gaspi clean
	$(MAKE) -C ring-1d/mpi clean
	$(MAKE) -C advanced clean
endif
else
all:
	$(MAKE) -C basic NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C ring-1d/gaspi NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C advanced NON_MPI_CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
clean:
	$(MAKE) -C basic clean
	$(MAKE) -C ring-1d/gaspi clean
	$(MAKE) -C advanced clean
endif
endif

CLEANFILES: *~
