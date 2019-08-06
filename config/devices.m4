################################################
# Check and select device
# ----------------------------------
AC_DEFUN([ACX_USABLE_DEVICE],[
        if test x${with_infiniband} != xno -a x${with_ethernet} != xno; then
           TITLE([Checking for device(s):])
           AC_MSG_ERROR([Concurrently Infiniband and Ethernet is not supported])
        elif test x${with_infiniband} != xno -a x${with_ethernet} = xno; then
           TITLE([Checking for Infiniband])
           ACX_INFINIBAND
           if test x${HAVE_INFINIBAND} = x0; then
              AC_MSG_ERROR([Infiniband requested, but can not use it])
           fi
        elif test x${with_infiniband} = xno -a x${with_ethernet} != xno; then
           TITLE([Checking for Ethernet])
           ACX_ETHERNET
           if test x${HAVE_TCP} = x0; then
              AC_MSG_ERROR([Ethernet requested, but can not use it])
           fi
        else
	   TITLE([Infiniband or Ethernet is required, checking for Infiniband...])
           with_infiniband=yes
           ACX_INFINIBAND
           if test x${HAVE_INFINIBAND} = x0; then
	      AC_MSG_NOTICE([Infiniband can not be used])
              TITLE([Checking for Ethernet])
              ACX_ETHERNET
              if test x${HAVE_TCP} = x0; then
              	 AC_MSG_ERROR([Neither Infiniband nor Ethernet are usable])
              fi
           fi
        fi

        # COPY DEFAULT FILES FOR TESTING
        AM_CONDITIONAL([WITH_ETHERNET], test x${HAVE_TCP} = x1)
        if [test x${HAVE_TCP} = x1]; then
           cp tests/defs/default_tcp.def tests/defs/default.def
           options="$options Ethernet"
        fi
        AM_CONDITIONAL([WITH_INFINIBAND],[test x${HAVE_INFINIBAND} = x1])
 	AM_CONDITIONAL([WITH_INFINIBAND_EXP],[test x${HAVE_INFINIBAND_EXP} = x1 -a x$exp_infiniband != xno])
        if [test x${HAVE_INFINIBAND} = x1]; then
           cp tests/defs/default_ib.def tests/defs/default.def
           if [test x${HAVE_INFINIBAND_EXP} = x1 -a x$exp_infiniband != xno]; then
              options="$options Infiniband Experimental"
	   else
	      options="$options Infiniband"
	   fi
        fi
	])

################################################
# Check and set INFINIBAND path
# ----------------------------------
AC_DEFUN([ACX_INFINIBAND],[
	if test "x$with_infiniband" != xno; then
   	   if test "x$with_infiniband" != xyes; then
	      # User specifies path(s)
      	      ac_inc_infiniband=$with_infiniband/include/infiniband
	      ACX_IBVERBS_VERSION([$ac_inc_infiniband],[HAVE_INF_HEADER=1],[HAVE_INF_HEADER=0])
	      AC_CHECK_FILE($ac_inc_infiniband/verbs_exp.h,
			    [HAVE_INFINIBAND_EXP=1],[HAVE_INFINIBAND_EXP=0])
	      for iblib in libibverbs.so libibverbs.a; do
      	      	  ac_lib_infiniband=$with_infiniband/lib64
		  AC_CHECK_FILE($ac_lib_infiniband/$iblib,[HAVE_INF_LIB=1],[HAVE_INF_LIB=0])
		  if test ${HAVE_INF_LIB} = 1; then
		     break
		  fi
	      done
   	   else
	      # Try to determine path(s)
	      inc_paths=`cpp -v /dev/null >& cppt`
	      inc_paths=`sed -n '/^#include </,/^End/p' cppt | sed '1d;$d'`
	      rm -f cppt
	      for ibinc in $inc_paths; do
	      	  ac_inc_infiniband=$ibinc/infiniband
		  ACX_IBVERBS_VERSION([$ac_inc_infiniband],[HAVE_INF_HEADER=1],[HAVE_INF_HEADER=0])
	      	  if test ${HAVE_INF_HEADER} = 1; then
		     AC_CHECK_FILE($ac_inc_infiniband/verbs_exp.h,
				   [HAVE_INFINIBAND_EXP=1],[HAVE_INFINIBAND_EXP=0])
	      	     break
	      	  fi
	      done
      	      AC_CHECK_LIB([ibverbs],[ibv_open_device],[HAVE_INF_LIB=1],[HAVE_INF_LIB=0])
   	   fi
	fi
	if test ${HAVE_INF_HEADER} = 1 -a ${HAVE_INF_LIB} = 1; then
	   HAVE_INFINIBAND=1
	   AC_SUBST(ac_inc_infiniband,[-I$ac_inc_infiniband])
	   if test ! -z $ac_lib_infiniband; then
	      AC_SUBST(ac_lib_infiniband,[-L$ac_lib_infiniband])
	   fi
	else
	   HAVE_INFINIBAND=0
	fi
	])

AC_DEFUN([ACX_IBVERBS_VERSION],[
	AC_MSG_CHECKING([whether $1/verbs.h contains IBV_LINK_LAYER_ETHERNET])

	AC_LANG_PUSH(C)
	  OLD_CFLAGS=$CFLAGS
	  CFLAGS="$AM_CFLAGS $CFLAGS -I$1"
	  AC_LINK_IFELSE(
		       [AC_LANG_PROGRAM([[#include <verbs.h>]],
		       [[int a = IBV_LINK_LAYER_ETHERNET;]])],
		       [AC_MSG_RESULT([yes]);
		        $2],
		       [AC_MSG_RESULT([no]);
		        $3])
	  CFLAGS=$OLD_CFLAGS;
	AC_LANG_POP([C])
	])

################################################
# Check and set ETHERNET path
# ----------------------------------
AC_DEFUN([ACX_ETHERNET],[
	AC_CHECK_HEADER(netinet/tcp.h,[HAVE_TCP=1],[HAVE_TCP=0])
	])
