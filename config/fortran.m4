################################################
# Check and set FORTRAN
#	- Iso C bindings
# 	- Output module flag
#	- Flags for testing
# ----------------------------------
AC_DEFUN([ACX_FORTRAN],[
	HAVE_FORTRAN=0
	if test x"$FC" = x; then
    	      AC_MSG_NOTICE([Could not find Fortran 90 compiler])
	else
	      AC_LANG_PUSH(Fortran)
	         ACX_FC_ISO_C()
		 if test ${HAVE_ISOC_FORTRAN} = 1; then
	      	    ACX_FCFLAGS
	      	    HAVE_FORTRAN=1
		 fi
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
	   AC_MSG_NOTICE([No iso C Fortran module])
           HAVE_ISOC_FORTRAN=0
	else
           HAVE_ISOC_FORTRAN=1
	fi
	])

################################################
# Set default FCFLAGS for tests
# ----------------------------------
AC_DEFUN([ACX_FCFLAGS],[
	FCLOC=`basename ${FC}`
	case "${FCLOC}" in
	     gfortran|mpif90)
                AC_SUBST([FCFLAGS_TESTS],["-fno-range-check"])
		;;
	     ifort|mpiifort)
                AC_SUBST([FCFLAGS_TESTS],["-mcmodel=medium"])
		;;
	     *)
		AC_SUBST([FCFLAGS_TESTS],[" "])
		;;
	esac
	AC_MSG_NOTICE([Using test FCFLAGS_TESTS="$FCFLAGS_TESTS"])
])
