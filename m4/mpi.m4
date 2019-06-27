AC_DEFUN([ACX_MPI],[

HAVE_MPI=0
if test "x$with_mpi" != xno; then
HAVE_MPI=1
AC_DEFINE([HAVE_MPI],[1],[Mpi])
if test "x$with_mpi" != xyes; then
ac_inc_mpi=$with_mpi/include/
AC_CHECK_FILE($ac_inc_mpi/mpi.h,[AC_SUBST(ac_inc_mpi)],[HAVE_INFINIBAND=0])
ac_lib_mpi=$with_mpi/lib64
AC_CHECK_FILE($ac_lib_mpi/libmpi.so,[AC_SUBST(ac_lib_mpi)],[HAVE_INFINIBAND=0])
else
AX_PROG_CC_MPI([test "x$with_mpi" != xno],[],[
use_mpi=no
if test x"$with_mpi" = xyes; then
AC_MSG_FAILURE([MPI compiler requested, but couldn't use MPI.])
# else
# AC_MSG_WARN([No MPI compiler found, won't use MPI.])
fi
])
fi
fi
])
