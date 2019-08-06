if WITH_INFINIBAND
if WITH_MPI
if HAVE_CINTEL
all :
	$(MAKE) -C ring-1d/mpi
endif
else
all +:
	$(MAKE) -C basic CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C ring-1d/gaspi CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
	$(MAKE) -C advanced CC=@NON_MPI_CC@ srcdir=$(abs_top_srcdir)
endif
endif

clean :
	-$(MAKE) -C basic clean
	-$(MAKE) -C ring-1d/gaspi clean
	-$(MAKE) -C ring-1d/mpi clean
	-$(MAKE) -C advanced clean

CLEANFILES : *~
