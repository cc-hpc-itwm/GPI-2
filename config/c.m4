################################################
# Set default C std flag
# ----------------------------------
AC_DEFUN([ACX_C],[
	CCLOC=`basename ${CC}`
	case "${CCLOC}" in
	     gcc|clang|mpicc)
		AS_IF([test "x${CCLOC}" = xmpicc],AC_SUBST([NON_MPI_CC],`mpicc --version | head -n 1 | { read first rest ; echo $first ; }`))
		CFLAGS+=" -std=gnu99 -Wall"
	  	;;
	     icc|mpiicc)
	     	AC_SUBST([NON_MPI_CC],icc)
		AS_IF([test "x${CCLOC}" = xmpiicc],with_mpi=yes)
	        CFLAGS+=" -std=gnu99 -Wall"
      		;;
    	     *)
		CFLAGS+=" -std=c99"
       		;;
  	esac
        AS_IF([test "x${CCLOC}" = xmpicc -o "x${CCLOC}" = xmpiicc],[with_mpi=yes])
        AC_MSG_NOTICE([Using CFLAGS="$CFLAGS"])

	# USED FOR tutorial/code
	AM_CONDITIONAL([HAVE_CGNU],[test x${NON_MPI_CC} = xgcc -o x${NON_MPI_CC} = xclang])
	AM_CONDITIONAL([HAVE_CINTEL],[test x${NON_MPI_CC} = xicc])

	])
