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
