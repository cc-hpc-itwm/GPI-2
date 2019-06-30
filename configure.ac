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
# DYNAMICAL COMPILATION RULES FOR TESTS
AC_CONFIG_COMMANDS_PRE([echo "# RULES " > tests/tests/compile_rules.in])
AC_CONFIG_COMMANDS_PRE([rm -fv tests/tests/Makefile.am])
AC_CONFIG_COMMANDS_PRE([echo "include compile_rules.in" > tests/tests/Makefile.am])
AC_CONFIG_COMMANDS_PRE([cat tests/tests/Makefile.am.inc >> tests/tests/Makefile.am])
AC_CONFIG_COMMANDS_PRE([echo "# RULES " > tests/microbenchmarks/compile_rules.in])
AC_CONFIG_COMMANDS_PRE([rm -fv tests/microbenchmarks/Makefile.am])
AC_CONFIG_COMMANDS_PRE([echo "include compile_rules.in" > tests/microbenchmarks/Makefile.am])
AC_CONFIG_COMMANDS_PRE([cat tests/microbenchmarks/Makefile.am.inc >> tests/microbenchmarks/Makefile.am])
# THE MAKEFILE.AM'S
AC_CONFIG_FILES([Makefile
	src/Makefile
	tests/Makefile
	tests/tests/Makefile
	tests/microbenchmarks/Makefile
	docs/Makefile
	])
AM_INIT_AUTOMAKE([subdir-objects])

#AM_MAINTAINER_MODE([enable])

# Static lib
AC_DISABLE_SHARED
LT_PREREQ([2.2])
LT_INIT

# Checks for programs.
AC_PROG_CC
AX_CHECK_STDC
# AC_PROG_CC_C99
# AC_PROG_CC_STDC
# AM_PROG_CC_C_O
AC_PROG_FC
AC_PROG_CXX
AC_PROG_AWK
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_MAKE_SET

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lm':
AC_CHECK_LIB([m], [main])
# FIXME: Replace `main' with a function in `-lrt':
AC_CHECK_LIB([rt], [main])

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h float.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h sys/timeb.h unistd.h sys/signal.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([ftime ftruncate gethostbyname gethostname gettimeofday inet_ntoa memmove memset pow select socket sqrt strerror strtol uname mktemp])

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
# if test x"$with_mpi" == xyes; then
# AC_CHECK_PROGS([CCio], [mpicc mpixlc_r mpixlc hcc mpxlc_r mpxlc sxmpicc mpifcc mpgcc mpcc cmpicc cc gcc])
# AC_SUBST([CC],[$CCio])
# echo $CC
# fi
ACX_MPI
AM_CONDITIONAL([WITH_MPI], test x${HAVE_MPI} = x1)

# LOADLEVER
AC_ARG_WITH(loadlever, [AS_HELP_STRING([--with-loadlever],[compile with loadlevel support.])],
[with_loadlever=$withval],
[with_loadlever=no])
AM_CONDITIONAL([WITH_LOADLEVER], test x${HAVE_LOADLEVER} = x1)

# FORTRAN
AC_ARG_WITH([fortran],AS_HELP_STRING([--with-fortran], [compile Fortran modules (default)]),
[with_fortran=$withval],
[with_fortran=yes])
ACX_FORTRAN
AM_CONDITIONAL([WITH_FORTRAN], test x${HAVE_FORTRAN} = x1)

# DOXYGEN
DX_HTML_FEATURE(ON)
DX_DOT_FEATURE(ON)
DX_MAN_FEATURE(OFF)
DX_PDF_FEATURE(ON)
DX_INIT_DOXYGEN(GPI-2, [docs/Doxyfile], [docs/doxygen-doc])

AC_OUTPUT
