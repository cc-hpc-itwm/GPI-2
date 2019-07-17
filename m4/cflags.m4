################################################
# Set default C std flag
# ----------------------------------
AC_DEFUN([ACX_CFLAGS],[
	if test -z "${CFLAGS}"; then
	   CCLOC=`basename ${CC}`
	   case "${CCLOC}" in
	     gcc|mpicc)
		AC_SUBST([NON_MPI_CC],gcc)
		CFLAGS+=" -std=gnu99"
	  	;;
	     icc|mpiicc)
	     	AC_SUBST([NON_MPI_CC],icc)
	        CFLAGS+=" -std=gnu99"
      		;;
    	     *)
		CFLAGS+=" -std=c99"
       		;;
  	   esac
	fi
	AC_MSG_NOTICE([Using std CFLAGS="$CFLAGS"])

	AM_CONDITIONAL([HAVE_CGNU],[test ${NON_MPI_CC} = gcc])
	AM_CONDITIONAL([HAVE_CINTEL],[test ${NON_MPI_CC} = icc])

	])
