################################################
# Set default CFLAGS
# ----------------------------------
AC_DEFUN([ACX_CFLAGS],
[
AC_REQUIRE([AC_CANONICAL_HOST])

if test -z "${CFLAGS}"; then
  case "${CC}" in
    gcc|mpicc*)
      CFLAGS+=" "
      ;;
    icc*|mpiicc*)
        CFLAGS+=" -std=gnu11"
      ;;
    *)
        CFLAGS=" -std=c99"
       ;;
  esac
fi
AC_MSG_NOTICE([Using std CFLAGS="$CFLAGS"])
])
