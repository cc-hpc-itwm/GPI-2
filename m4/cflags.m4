################################################
# Set default C std flag
# ----------------------------------
AC_DEFUN([ACX_CFLAGS],[
	if test -z "${CFLAGS}"; then
	   CCLOC=`basename ${CC}`
	   case "${CCLOC}" in
	     gcc|mpicc*)
		CFLAGS+=" "
	  	;;
	     icc*|mpiicc*)
	        CFLAGS+=" -std=gnu99"
      		;;
    	     *)
		CFLAGS+=" -std=c99"
       		;;
  	   esac
	fi
	AC_MSG_NOTICE([Using std CFLAGS="$CFLAGS"])
	])
