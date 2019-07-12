# INITS
AC_PREREQ([2.69])
AC_INIT([GPI], [1.3.0], [http://gpi-site.com.www488.your-server.de/contact/])
AC_CONFIG_SRCDIR([src/GPI2.c])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

# HACK LIBTOOL FLAGS
if test -z "${CFLAGS}"; then
CFLAGS=
fi
if test -z "${FCFLAGS}"; then
FCFLAGS=
fi
if test -z "${CXXFLAGS}"; then
CXXFLAGS=
fi

# AUTOMAKE
AC_USE_SYSTEM_EXTENSIONS
AC_CONFIG_FILES([Makefile
	src/Makefile
	tests/Makefile
	tests/tests/Makefile
	tests/microbenchmarks/Makefile
	tutorial/code/Makefile
	])
AM_INIT_AUTOMAKE([subdir-objects])
#AM_MAINTAINER_MODE([enable])

# LIBTOOL
AC_DISABLE_SHARED
LT_PREREQ([2.2])
LT_INIT

# CHECKS FOR PROGRAMS.
AC_PROG_CC
ACX_CFLAGS dnl CHOOSE C STD
AC_PROG_FC
#AC_PROG_CXX
AC_PROG_AWK
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# CHECKS FOR LIBRARIES.
AC_CHECK_LIB([m], [sqrt])

# CHECKS FOR HEADER FILES.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h float.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h sys/timeb.h unistd.h sys/signal.h])

# CHECKS FOR TYPEDEFS, STRUCTURES, AND COMPILER CHARACTERISTICS.
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# CHECKS FOR LIBRARY FUNCTIONS.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([ftime ftruncate gethostbyname gethostname gettimeofday inet_ntoa memmove memset pow select socket sqrt strerror strtol uname mktemp])

# MPI
AC_ARG_WITH(mpi,[AS_HELP_STRING([--with-mpi],[compile with MPI support])],
		[with_mpi=$withval],
		[with_mpi=no])
ACX_MPI
AM_CONDITIONAL([WITH_MPI],[test x${HAVE_MPI} = x1])
AM_CONDITIONAL([WITH_MPI_USER],[test x${HAVE_MPI_USER} != x0])

# FORTRAN
AC_ARG_WITH([fortran],[AS_HELP_STRING([--with-fortran], [compile Fortran modules (default)])],
	 	      [with_fortran=$withval],
		      [with_fortran=yes])
ACX_FORTRAN
AM_CONDITIONAL([WITH_FORTRAN],[test x${HAVE_FORTRAN} = x1])

# CHECKS INFINIBAND
AC_ARG_WITH([infiniband],[AS_HELP_STRING([--with-infiniband], [compile for infiniband (default)])],
			 [with_infiniband=$withval],
			 [with_infiniband=yes])
ACX_INFINIBAND
AM_CONDITIONAL([WITH_INFINIBAND],[test x${HAVE_INFINIBAND} = x1])
if [test x${HAVE_INFINIBAND} = x1]; then
   cp tests/defs/default_ib.def tests/defs/default.def
fi

# CHECKS ETHERNET
AC_ARG_WITH([ethernet],[AS_HELP_STRING([--with-ethernet], [compile for ethernet])],
		       [with_ethernet=$withval],
		       [with_ethernet=no])
ACX_ETHERNET
AM_CONDITIONAL([WITH_ETHERNET], test x${HAVE_TCP} = x1)
if [test x${HAVE_TCP} = x1]; then
   cp tests/defs/default_tcp.def tests/defs/default.def
fi

# BUILD DEBUG LIBRARY
AC_ARG_ENABLE([debug],AS_HELP_STRING([--disable-debug], [do not build/install debug library (it is used by tests!)]),
	      [build_debug=$enableval],
	      [build_debug=yes])
AM_CONDITIONAL([ENABLE_DEBUG],[test $build_debug = yes])

# BUILD STAT LIBRARY
AC_ARG_ENABLE([stat],[AS_HELP_STRING([--disable-stat], [do not build/install statistic library])],
		     [build_stat=$enableval],
		     [build_stat=yes])
AM_CONDITIONAL([ENABLE_STAT],[test $build_stat = yes])

# LOADLEVER
AC_ARG_WITH(loadlever,[AS_HELP_STRING([--with-loadlever],[compile with loadlevel support.])],
		      [with_loadlever=$withval],
		      [with_loadlever=no])
AM_CONDITIONAL([WITH_LOADLEVER],[test $with_loadlever = yes])

# DOXYGEN
AC_CHECK_PROGS([DOXYGEN],[doxygen])
if test -z "$DOXYGEN"; then
   AC_MSG_WARN([Doxygen not found])
fi
AC_CHECK_PROGS([DOT],[dot])
if test -z "$DOT"; then
   AC_MSG_WARN([Dot not found])
fi
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN" -a -n "$DOT"])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Makefile])])

AC_OUTPUT
