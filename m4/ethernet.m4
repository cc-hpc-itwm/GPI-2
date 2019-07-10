################################################
# Check and set INFINIBAND path
# ----------------------------------
AC_DEFUN([ACX_ETHERNET],[
	HAVE_TCP=1
	if test x${with_ethernet} != xno; then
	   if test x${HAVE_INFINIBAND} = x1; then
	      AC_MSG_ERROR([Infiniband was already found])
	   else
	      AC_CHECK_HEADER(netinet/tcp.h,[HAVE_TCP=1],[AC_MSG_ERROR([There is no TCP connection neither])])
	   fi
	fi
	])
