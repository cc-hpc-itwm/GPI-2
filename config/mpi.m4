################################################
# Check and set MPI compiler
# ----------------------------------
AC_DEFUN([ACX_MPI],[
	   if test "x$with_mpi" != xyes; then
	      # Check in "standard" paths
	      inc_path=include
	      ac_inc_mpi=$with_mpi/$inc_path
	      AC_CHECK_FILE($ac_inc_mpi/mpi.h,[HAVE_MPI_HEADER=1],[HAVE_MPI_HEADER=0])
	      for lib_path in lib lib64; do
	      	  for mpi_lib in libmpi.a libmpi.so libmpich.a libmpich.so; do
	      	      ac_lib_mpi=$with_mpi/$lib_path
	      	      AC_CHECK_FILE($ac_lib_mpi/$mpi_lib,[HAVE_MPI_LIB=1],[HAVE_MPI_LIB=0])
	      	      if test ${HAVE_MPI_LIB} = 1; then
		      	 break
		      fi
		  done
	      	  if test ${HAVE_MPI_LIB} = 1; then
		     break
		  fi
	      done
	   else
	      # Use autoconf macros
	      AC_CHECK_HEADER([mpi.h],[HAVE_MPI_HEADER=1],[HAVE_MPI_HEADER=0])
	      for mpi_lib in mpi mpich; do
      	      	  AC_CHECK_LIB([$mpi_lib],[MPI_Init],[HAVE_MPI_LIB=1],[HAVE_MPI_LIB=0])
	      	  if test ${HAVE_MPI_LIB} = 1; then
		     break
		  fi
	      done
	   fi
	   if test ${HAVE_MPI_HEADER} = 1 -a ${HAVE_MPI_LIB} = 1; then
	      HAVE_MPI=1
	      if test ! -z $ac_inc_mpi; then
	      	 AC_SUBST([ac_inc_mpi],[-I$ac_inc_mpi])
	      fi
	      if test "x$with_mpi_extra_flags" = xno; then
	      	 if test -z $ac_lib_mpi; then
		    AC_SUBST([ac_lib_mpi],["-l$mpi_lib"])
		 else
		    libvar=${mpi_lib:3}
		    AC_SUBST([ac_lib_mpi],["-L$ac_lib_mpi -l${libvar%.*}"])
		 fi
	      fi
	   else
	      HAVE_MPI=0
	      AC_MSG_ERROR([MPI compiler requested, but could not use it])
	   fi
	])
