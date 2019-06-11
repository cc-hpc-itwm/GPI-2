AC_DEFUN([ACX_FORTRAN], [
AC_REQUIRE([AC_PROG_FC])
HAVE_FORTRAN=0
if test "x$with_fortran" == xyes; then
  HAVE_FORTRAN=1
  AC_DEFINE(HAVE_FORTRAN, [1], [GPI compiled with fortran support])

  dnl try to find out what is the default FORTRAN 90 compiler
  acx_save_fcflags="${FCFLAGS}"
  AC_PROG_FC([], [Fortran 90])
  if test x"$FC" = x; then
    AC_MSG_ERROR([could not find Fortran 90 compiler])
  fi

  AC_LANG_PUSH(Fortran)
  AC_FC_SRCEXT(f90)
  FCFLAGS="${acx_save_fcflags}"

  # Check flag for out module dir
  AC_FC_MODULE_OUTPUT_FLAG([AC_SUBST([ac_cv_fc_module_output_flag])],[])
  dnl Fortran default flags
#  ACX_FCFLAGS

  dnl how Fortran mangles function names
  AC_FC_WRAPPERS

  dnl check whether the Fortran compiler supports Fortran 2003 iso_c_binding
  ACX_FC_ISO_C_BINDING([ac_cv_build_fortran03=yes], [ac_cv_build_fortran03=no,
    						    AC_MSG_WARN([Could not find Fortran 2003 iso_c_binding. Fortran 2003 interface will not be compiled.])])

  AC_LANG_POP([Fortran])

fi
])

AC_DEFUN([ACX_FC_ISO_C_BINDING], [

AC_MSG_CHECKING([for Fortran 2003 iso_c_binding])

testprog="AC_LANG_PROGRAM([],[
use iso_c_binding
implicit none
type(c_ptr) :: ptr
ptr = c_null_ptr
if (c_associated(ptr)) stop 3])"

acx_iso_c_binding_ok=no
AC_LINK_IFELSE($testprog, [acx_iso_c_binding_ok=yes], [])

AC_MSG_RESULT([$acx_iso_c_binding_ok])
if test x"$acx_iso_c_binding_ok" = xyes; then
AC_DEFINE(ISO_C_BINDING, 1, [compiler supports Fortran 2003 iso_c_binding])
$1
else
$2
fi
])
