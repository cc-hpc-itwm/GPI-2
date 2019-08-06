SUBDIRS = atomic cm collectives comm config error_vec extensions

if WITH_FORTRAN
SUBDIRS += fortran
endif

SUBDIRS += groups init lists

if WITH_MPI
SUBDIRS += mpi
endif

SUBDIRS += notifications passive segments threads utils

CLEANFILES = *~ *.bak *.bin fortran/*.mod
