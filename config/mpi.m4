################################################
# Check and set MPI compiler
# ----------------------------------
AC_DEFUN([ACX_MPI],[
	   if test "x$with_mpi" != xyes; then
	      # Check "standard" openmpi
	      for inc_path in include/ include/openmpi-x86_64; do
	      	  ac_inc_mpi=$with_mpi/$inc_path
	      	  AC_CHECK_FILE($ac_inc_mpi/mpi.h,[HAVE_MPI=1],[HAVE_MPI=0])
	      	  if test ${HAVE_MPI} = 1; then
		     AC_SUBST([ac_inc_mpi],[-I$ac_inc_mpi])
		     break
		  fi
	      done
	      for lib_path in lib lib64 lib64/openmpi/lib; do
	      	  for mpi_lib in libmpi.a libmpi.so; do
	      	      ac_lib_mpi=$with_mpi/$lib_path
	      	      AC_CHECK_FILE($ac_lib_mpi/$mpi_lib,[HAVE_MPI=1],[HAVE_MPI=0])
	      	      if test ${HAVE_MPI} = 1; then
		      	 if test "x$with_mpi_extra_flags" = xno; then
	      	     	    AC_SUBST([ac_lib_mpi],["-L$ac_lib_mpi -lmpi"])
	      	     	 fi
		         break
		      fi
		  done
	      done
	      if test ${HAVE_MPI} = 0; then
	      # Check "standard" mpich
	      for inc_path in include/ include/mpich-3.2-x86_64; do
	      	  ac_inc_mpi=$with_mpi/$inc_path
	      	  AC_CHECK_FILE($ac_inc_mpi/mpi.h,[HAVE_MPI=1],[HAVE_MPI=0])
	      	  if test ${HAVE_MPI} = 1; then
	     	     AC_SUBST([ac_inc_mpi],[-I$ac_inc_mpi])
	      	     break
	      	  fi
	      done
	      for lib_path in lib lib64 lib64/mpich-3.2/lib; do
	      	  for mpi_lib in libmpich.a libmpich.so; do
	      	      ac_lib_mpi=$with_mpi/$lib_path
	      	      AC_CHECK_FILE($ac_lib_mpi/$mpi_lib,[HAVE_MPI=1],[HAVE_MPI=0])
	      	      if test ${HAVE_MPI} = 1; then
	      	      	 if test "x$with_mpi_extra_flags" = xno; then
	      		    AC_SUBST([ac_lib_mpi],["-L$ac_lib_mpi -lmpich"])
	      		 fi
	      		 break
	      	      fi
	      	  done
	      done
	      fi
	   else
	      # Use autoconf macros
	      AC_CHECK_HEADER([mpi.h],[HAVE_MPI=1],[HAVE_MPI=0])
	      for mpi_lib in mpi mpich; do
      	      	  AC_CHECK_LIB([$mpi_lib],[MPI_Init],[HAVE_MPI=1],[HAVE_MPI=0])
	      	  if test ${HAVE_MPI} = 1; then
		     if test "x$with_mpi_extra_flags" = xno; then
	      	     	AC_SUBST([ac_lib_mpi],[-l$mpi_lib])
	      	     fi
		     break
		  fi
	      done
	   fi
	   if test x${HAVE_MPI} = x0; then
	      AC_MSG_NOTICE([MPI compiler requested, but could not use it])
	   fi
	])
