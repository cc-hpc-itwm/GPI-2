################################################
# Check and set INFINIBAND path
# ----------------------------------
AC_DEFUN([ACX_INFINIBAND],[
	if test "x$with_infiniband" != xno; then
   	   if test "x$with_infiniband" != xyes; then
      	      ac_inc_infiniband=$with_infiniband/include/infiniband
      	      AC_CHECK_FILE([$ac_inc_infiniband/verbs.h],
	        [HAVE_INFINIBAND=1;ACX_IBVERBS_VERSION([ac_inc_infiniband])],
		[HAVE_INFINIBAND=0])
      	      ac_lib_infiniband=$with_infiniband/lib64
      	      AC_CHECK_FILE($ac_lib_infiniband/libibverbs.so,
	        [HAVE_INFINIBAND=1;AC_SUBST(ac_lib_infiniband)],
		[HAVE_INFINIBAND=0])
   	   else
	      ## TODO HOW TO EXTRACT INCLUDE AND LIB PATH
      	      AC_CHECK_HEADER([infiniband/verbs.h],[],[HAVE_INFINIBAND=0])
      	      # echo "HERRERE $ac_cv_header_infiniband_verbs_h"
      	      AC_CHECK_LIB([ibverbs],[ibv_open_device],[],[HAVE_INFINIBAND=0])
	      # echo "HERRERE $ac_cv_lib_ibv_open_device"
   	   fi
	fi
	])

AC_DEFUN([ACX_IBVERBS_VERSION],[
	AC_MSG_CHECKING([whether verbs.h contains IBV_LINK_LAYER_ETHERNET])

	OLD_CFLAGS=$CFLAGS
	CFLAGS="$AM_CFLAGS $CFLAGS -I$1"
	AC_LINK_IFELSE(
		       [AC_LANG_PROGRAM([[#include <verbs.h>]],
		       [[int a = IBV_LINK_LAYER_ETHERNET;]])],
		       [AC_MSG_RESULT([yes]);
		        AC_SUBST($1)],
		       [AC_MSG_RESULT([no]);
		        HAVE_INFINIBAND=0])
	CFLAGS=$OLD_CFLAGS;
	])
