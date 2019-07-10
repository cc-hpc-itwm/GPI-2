################################################
# Check and set MPI compiler
# ----------------------------------
AC_DEFUN([ACX_MPI],[
	HAVE_MPI=1
	if test "x$with_mpi" != xno; then
	   if test "x$with_mpi" != xyes; then
	      HAVE_MPI_USER=1
	      ac_inc_mpi=$with_mpi/include/openmpi-x86_64
	      AC_CHECK_FILE($ac_inc_mpi/mpi.h,[AC_SUBST([ac_inc_mpi])],[HAVE_MPI=0])
	      ac_lib_mpi=$with_mpi/lib64/openmpi/lib
	      AC_CHECK_FILE($ac_lib_mpi/libmpi.so,[AC_SUBST([ac_lib_mpi])],[HAVE_MPI=0])
	   else
    	      HAVE_MPI_USER=0
	      AX_PROG_CC_MPI([test "x$with_mpi" = xyes],[],[HAVE_MPI=0])
	   fi
	   if test x${HAVE_MPI} = x0; then
	      AC_MSG_FAILURE([MPI compiler requested, but could not use it])
	   fi
	fi
	])
