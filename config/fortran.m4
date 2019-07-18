################################################
# Check and set FORTRAN
# 	- Output module flag
#	- iso C bindings
#	- flags for testing
# ----------------------------------
AC_DEFUN([ACX_FORTRAN],[
	HAVE_FORTRAN=0
	if test "x$with_fortran" == xyes; then
	   HAVE_FORTRAN=1
	   if test x"$FC" = x; then
    	      AC_MSG_ERROR([could not find Fortran 90 compiler])
	   fi

	   AC_LANG_PUSH(Fortran)
	     AC_FC_MODULE_OUTPUT_FLAG([AC_SUBST([ac_cv_fc_module_output_flag])],
			[AC_MSG_ERROR([no output Fortran module flag])])

	     ACX_FC_ISO_C([AC_MSG_ERROR([no iso C Fortran module])])

	     ACX_FCFLAGS
  	   AC_LANG_POP([Fortran])
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
	fi
	])
