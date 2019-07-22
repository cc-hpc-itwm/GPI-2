################################################
# Check and set INFINIBAND path
# ----------------------------------
AC_DEFUN([ACX_INFINIBAND],[
	if test "x$with_infiniband" != xno; then
   	   if test "x$with_infiniband" != xyes; then
	      # User specifies path(s)
      	      ac_inc_infiniband=$with_infiniband/include/infiniband
	      ACX_IBVERBS_VERSION([$ac_inc_infiniband],[HAVE_INFINIBAND=1],[HAVE_INFINIBAND=0])
	      if test ${HAVE_INFINIBAND} = 1; then
	      	 AC_SUBST(ac_inc_infiniband,[-I$ac_inc_infiniband])
	      fi
	      for iblib in libibverbs.so libibverbs.a; do
      	      	  ac_lib_infiniband=$with_infiniband/lib64
		  AC_CHECK_FILE($ac_lib_infiniband/$iblib,
				[HAVE_INFINIBAND=1;AC_SUBST(ac_lib_infiniband,[-L$ac_lib_infiniband])],
				[HAVE_INFINIBAND=0])
		  if test ${HAVE_INFINIBAND} = 1; then
		     break
		  fi
	      done
   	   else
	      # Try to determine path(s)
      	      AC_CHECK_LIB([ibverbs],[ibv_open_device],[HAVE_INFINIBAND=1],[HAVE_INFINIBAND=0])
	      inc_paths=`cpp -v /dev/null >& cppt`
	      inc_paths=`sed -n '/^#include </,/^End/p' cppt | sed '1d;$d'`
	      rm -f cppt
	      for ibinc in $inc_paths; do
	      	  ac_inc_infiniband=$ibinc/infiniband
		  ACX_IBVERBS_VERSION([$ac_inc_infiniband],[HAVE_INFINIBAND=1],[HAVE_INFINIBAND=0])
	      	  if test ${HAVE_INFINIBAND} = 1; then
		     AC_SUBST(ac_inc_infiniband,[-I$ac_inc_infiniband])
	      	     break
	      	  fi
	      done
   	   fi
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
	HAVE_TCP=0
	if test x${with_ethernet} != xno; then
	   if test x${HAVE_INFINIBAND} = x1; then
	      AC_MSG_WARN([Infiniband was already found])
	      HAVE_INFINIBAND=0
	   fi
	   AC_CHECK_HEADER(netinet/tcp.h,[HAVE_TCP=1],[AC_MSG_ERROR([There is no TCP connection neither])])
	fi
])
