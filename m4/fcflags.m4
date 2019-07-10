################################################
# Set default FCFLAGS for tests
# ----------------------------------
AC_DEFUN([ACX_FCFLAGS],[
	if test -z "${FCFLAGS}"; then
	   FCLOC=`basename ${FC}`
  	   case "${FCLOC}" in
    	     gfortran|mpif90*)
		#FCFLAGS+=" -O2"
      		AC_SUBST([FCFLAGS_TESTS],["-fno-range-check"])
      	     	;;
    	     ifort*|mpiifort*)
		#FCFLAGS+=" -O2"
      		AC_SUBST([FCFLAGS_TESTS],["-mcmodel=medium"])
	     	;;
    	     *)
		#FCFLAGS+=" -O2"
       		AC_SUBST([FCFLAGS_TESTS],[" "])
       		;;
  	   esac
	fi
	AC_MSG_NOTICE([Using test FCFLAGS="$FCFLAGS_TESTS"])
	])
