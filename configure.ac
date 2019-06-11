#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([GPI], [1.3.0], [http://gpi-site.com.www488.your-server.de/contact/])
AC_CONFIG_SRCDIR([src/GPI2.c])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])

#AC_REQUIRE_AUX_FILE([runtests.sh])
AC_USE_SYSTEM_EXTENSIONS
AM_INIT_AUTOMAKE([subdir-objects])
AC_CONFIG_FILES([Makefile
		 ])
		 dnl src/Makefile
                 dnl tests/Makefile
                 dnl tests/tests/Makefile
                 dnl tests/tests/init/Makefile

AM_MAINTAINER_MODE([enable])

# Static lib
AC_DISABLE_SHARED
LT_PREREQ([2.2])
LT_INIT

# Checks for programs.
AC_PROG_CC
# AC_PROG_CC_STDC
AC_PROG_CC_C99
# AM_PROG_CC_C_O
AC_PROG_FC
AC_PROG_AWK
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_MAKE_SET

dnl # Checks for libraries.
dnl # FIXME: Replace `main' with a function in `-lm':
dnl AC_CHECK_LIB([m], [main])
dnl # FIXME: Replace `main' with a function in `-lrt':
dnl AC_CHECK_LIB([rt], [main])

dnl # Checks for header files.
dnl AC_CHECK_HEADERS([arpa/inet.h fcntl.h float.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h sys/timeb.h unistd.h])

dnl # Checks for typedefs, structures, and compiler characteristics.
dnl AC_C_INLINE
dnl AC_TYPE_OFF_T
dnl AC_TYPE_PID_T
dnl AC_TYPE_SIZE_T
dnl AC_TYPE_SSIZE_T
dnl AC_TYPE_UINT16_T
dnl AC_TYPE_UINT32_T
dnl AC_TYPE_UINT64_T
dnl AC_TYPE_UINT8_T

dnl # Checks for library functions.
dnl AC_FUNC_ERROR_AT_LINE
dnl AC_FUNC_MALLOC
dnl AC_FUNC_MMAP
dnl AC_CHECK_FUNCS([ftime ftruncate gethostbyname gethostname gettimeofday inet_ntoa memmove memset pow select socket sqrt strerror strtol uname mktemp])


# CHECKS INFINIBAND
AC_ARG_WITH([infiniband],AS_HELP_STRING([--with-infiniband], [compile for infiniband (default)]),
[with_infiniband=$withval],
[with_infiniband=yes])
ACX_INFINIBAND
AM_CONDITIONAL([WITH_INFINIBAND], test x${HAVE_INFINIBAND} = x1)

# CHECKS ETHERNET
AC_ARG_WITH([ethernet],AS_HELP_STRING([--with-ethernet], [compile for ethernet]),
[with_ethernet=$withval],
[with_ethernet=no])
#  FALLBACK OR EXPLICITLY CHOSEN TCP
if test x${HAVE_INFINIBAND} = x0; then
   AC_CHECK_HEADER(netinet/tcp.h,[HAVE_TCP=1],[AC_MSG_ERROR([There is no TCP connection neither])])
fi
AM_CONDITIONAL([WITH_ETHERNET], test x${HAVE_TCP} = x1)

# BUILD DEBUG AND STAT LIBRARIES
AC_ARG_ENABLE([debug],
	AS_HELP_STRING([--disable-debug], [don't build/install debug libraries]),
	[ac_cv_build_debug=$enableval],
	[ac_cv_build_debug=yes])
AM_CONDITIONAL([ENABLE_DEBUG], [test $ac_cv_build_debug = yes])

AC_ARG_ENABLE([stat],
	AS_HELP_STRING([--disable-stat], [don't build/install stat libraries]),
	[ac_cv_build_stat=$enableval],
	[ac_cv_build_stat=yes])
AM_CONDITIONAL([ENABLE_STAT], [test $ac_cv_build_stat = yes])

# MPI
AC_ARG_WITH(mpi, [AS_HELP_STRING([--with-mpi],[compile with MPI (parallelization) support.])],
[with_mpi=$withval],
[with_mpi=no])
## TODO: SET AUTOMATICALLY CC to mpicc
dnl if test x"$with_mpi" == xyes; then
dnl AC_CHECK_PROGS([CCio], [mpicc mpixlc_r mpixlc hcc mpxlc_r mpxlc sxmpicc mpifcc mpgcc mpcc cmpicc cc gcc])
dnl AC_SUBST([CC],[$CCio])
dnl echo $CC
dnl fi
ACX_MPI
AM_CONDITIONAL([WITH_MPI], test x${HAVE_MPI} = x1)

# LOADLEVER
AC_ARG_WITH(loadlever, [AS_HELP_STRING([--with-loadlever],[compile with loadlevel support.])],
[with_loadlever=$withval],
[with_loadlever=no])
AM_CONDITIONAL([WITH_LOADLEVER], test x${HAVE_LOADLEVER} = x1)

# FORTRAN
AC_ARG_WITH([fortran],AS_HELP_STRING([--with-fortran], [compile Fortran modules]),
[with_fortran=$withval],
[with_fortran=yes])
ACX_FORTRAN
AM_CONDITIONAL([WITH_FORTRAN], test x${HAVE_FORTRAN} = x1)


dnl AC_CONFIG_FILES([Makefile])
                 dnl src/Makefile
                 dnl docs/latex/Makefile
                 dnl tests/microbenchmarks/Makefile
                 dnl tests/tests/atomic/Makefile
                 dnl tests/tests/cm/Makefile
                 dnl tests/tests/collectives/Makefile
                 dnl tests/tests/comm/Makefile
                 dnl tests/tests/config/Makefile
                 dnl tests/tests/error_vec/Makefile
                 dnl tests/tests/extensions/Makefile
                 dnl tests/tests/fortran/Makefile
                 dnl tests/tests/groups/Makefile
                 dnl tests/tests/lists/Makefile
                 dnl tests/tests/mpi/Makefile
                 dnl tests/tests/notifications/Makefile
                 dnl tests/tests/passive/Makefile
                 dnl tests/tests/segments/Makefile
                 dnl tests/tests/threads/Makefile
                 dnl tests/tests/utils/Makefile
                 dnl tutorial/code/advanced/Makefile
                 dnl tutorial/code/basic/Makefile
                 dnl tutorial/code/ring-1d/gaspi/Makefile
                 dnl tutorial/code/ring-1d/mpi/Makefile

AC_OUTPUT
