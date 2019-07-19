################################################
# Check and set FORTRAN
# 	- Output module flag
#	- iso C bindings
#	- flags for testing
# ----------------------------------
AC_DEFUN([ACX_FORTRAN],[
	HAVE_FORTRAN=0
	if test "x$with_fortran" == xyes; then
	   if test x"$FC" = x; then
    	      AC_MSG_WARN([could not find Fortran 90 compiler])
	   else 

	   AC_LANG_PUSH(Fortran)
	     ACX_FC_ISO_C([AC_MSG_WARN([no iso C Fortran module])])

	     ACX_FC_MODULE_FLAG([HAVE_MODFCFLAG=1],[HAVE_MODFCFLAG=0])
             if test ${HAVE_MODFCFLAG} = 1; then
	        AC_MSG_NOTICE([found output Fortran module flag])
		AC_SUBST([ac_cv_fc_module_output_flag])
	     else
		AC_MSG_WARN([no output Fortran module flag])
	     fi
	
             if test ${HAVE_MODFCFLAG} = 1 -a test ${HAVE_ISOC_FORTRAN} = 1; then
	      ACX_FCFLAGS
	      HAVE_FORTRAN=1
	     fi
  	   AC_LANG_POP([Fortran])
           fi
        fi
	])

AC_DEFUN([ACX_FC_ISO_C],[

	AC_MSG_CHECKING([for iso_C module])

	testprog="AC_LANG_PROGRAM([],[
			use iso_c_binding
			implicit none
			type(c_ptr) :: ptr
			ptr = c_null_ptr
			if (c_associated(ptr)) stop 3])"

	AC_LINK_IFELSE($testprog, [iso_c=yes], [iso_c=no])

	AC_MSG_RESULT([$iso_c])

	if test x"$iso_c" != xyes; then
	   $1
           HAVE_ISOC_FORTRAN=0
	else
           HAVE_ISOC_FORTRAN=1
	fi
	])

# TAKEN FROM AUTOCONF 2.69
# AC_FC_MODULE_FLAG([ACTION-IF-SUCCESS], [ACTION-IF-FAILURE = FAILURE])
# ---------------------------------------------------------------------
# Find a flag to include Fortran 90 modules from another directory.
# If successful, run ACTION-IF-SUCCESS (defaults to nothing), otherwise
# run ACTION-IF-FAILURE (defaults to failing with an error message).
# The module flag is cached in the ac_cv_fc_module_flag variable.
# It may contain significant trailing whitespace.
#
# Known flags:
# gfortran: -Idir, -I dir (-M dir, -Mdir (deprecated), -Jdir for writing)
# g95: -I dir (-fmod=dir for writing)
# SUN: -Mdir, -M dir (-moddir=dir for writing;
#                     -Idir for includes is also searched)
# HP: -Idir, -I dir (+moddir=dir for writing)
# IBM: -Idir (-qmoddir=dir for writing)
# Intel: -Idir -I dir (-mod dir for writing)
# Absoft: -pdir
# Lahey: -mod dir
# Cray: -module dir, -p dir (-J dir for writing)
#       -e m is needed to enable writing .mod files at all
# Compaq: -Idir
# NAGWare: -I dir
# PathScale: -I dir  (but -module dir is looked at first)
# Portland: -module dir (first -module also names dir for writing)
# Fujitsu: -Am -Idir (-Mdir for writing is searched first, then '.', then -I)
#                    (-Am indicates how module information is saved)
AC_DEFUN([ACX_FC_MODULE_FLAG],[
AC_CACHE_CHECK([Fortran 90 module inclusion flag], [ac_cv_fc_module_flag],
[AC_LANG_PUSH([Fortran])
ac_cv_fc_module_flag=unknown
mkdir conftest.dir
cd conftest.dir
AC_COMPILE_IFELSE([[
      module conftest_module
      contains
      subroutine conftest_routine
      write(*,'(a)') 'gotcha!'
      end subroutine
      end module]],
  [cd ..
   ac_fc_module_flag_FCFLAGS_save=$FCFLAGS
   # Flag ordering is significant for gfortran and Sun.
   for ac_flag in -M -I '-I ' '-M ' -p '-mod ' '-module ' '-Am -I'; do
     # Add the flag twice to prevent matching an output flag.
     FCFLAGS="$ac_fc_module_flag_FCFLAGS_save ${ac_flag}conftest.dir ${ac_flag}conftest.dir"
     AC_COMPILE_IFELSE([[
      program main
      use conftest_module
      call conftest_routine
      end program]],
       [ac_cv_fc_module_flag="$ac_flag"])
     if test "$ac_cv_fc_module_flag" != unknown; then
       break
     fi
   done
   FCFLAGS=$ac_fc_module_flag_FCFLAGS_save
])
rm -rf conftest.dir
AC_LANG_POP([Fortran])
])
if test "$ac_cv_fc_module_flag" != unknown; then
  FC_MODINC=$ac_cv_fc_module_flag
  $1
else
  FC_MODINC=
  m4_default([$2],
    [AC_MSG_ERROR([unable to find compiler flag for module search path])])
fi
AC_SUBST([FC_MODINC])
# Ensure trailing whitespace is preserved in a Makefile.
AC_SUBST([ac_empty], [""])
#AC_CONFIG_COMMANDS_PRE([case $FC_MODINC in #(
#  *\ ) FC_MODINC=$FC_MODINC'${ac_empty}' ;;
#esac])dnl
])
