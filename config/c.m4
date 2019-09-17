################################################
# Set default C std flag
# ----------------------------------
AC_DEFUN([ACX_C],[
	if test -z "${CFLAGS}"; then
	   CCLOC=`basename ${CC}`
	   case "${CCLOC}" in
	     gcc|mpicc)
		AS_IF([test "x${CCLOC}" = xmpicc],[with_mpi=yes])
		AS_IF([test "x${CCLOC}" = xmpicc],AC_SUBST([NON_MPI_CC],`mpicc --version | head -n 1 | { read first rest ; echo $first ; }`))
		CFLAGS+=" -std=gnu99"
	  	;;
	     icc|mpiicc)
	     	AC_SUBST([NON_MPI_CC],icc)
		AS_IF([test "x${CCLOC}" = xmpiicc],with_mpi=yes)
	        CFLAGS+=" -std=gnu99"
      		;;
    	     *)
		CFLAGS+=" -std=c99"
       		;;
  	   esac
	fi
	AC_MSG_NOTICE([Using std CFLAGS="$CFLAGS"])

	# USED FOR tutorial/code
	AM_CONDITIONAL([HAVE_CGNU],[test x${NON_MPI_CC} = xgcc])
	AM_CONDITIONAL([HAVE_CINTEL],[test x${NON_MPI_CC} = xicc])

	])
