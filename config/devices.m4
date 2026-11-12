################################################
# Check and select device
# ----------------------------------
AC_DEFUN([ACX_USABLE_DEVICE],[
        if test x${with_infiniband} != xno -a x${with_ethernet} != xno; then
           TITLE([Checking for device(s):])
           AC_MSG_ERROR([Concurrently Infiniband and Ethernet is not supported])
        elif test x${with_infiniband} != xno -a x${with_ethernet} = xno; then
           TITLE([Checking for Infiniband])
           ACX_INFINIBAND
           if test x${HAVE_INFINIBAND} = x0; then
              AC_MSG_ERROR([Infiniband requested, but can not use it])
           fi
        elif test x${with_ofi} != xno; then
           TITLE([Checking for OFI])
           ACX_OFI
           if test x${HAVE_OFI} = x0; then
             AC_MSG_ERROR([OFI requested, but can not use it])
           fi
        elif test x${with_infiniband} = xno -a x${with_ethernet} != xno; then
           TITLE([Checking for Ethernet])
           ACX_ETHERNET
           if test x${HAVE_TCP} = x0; then
              AC_MSG_ERROR([Ethernet requested, but can not use it])
           fi
        else
	   TITLE([Infiniband or Ethernet is required, checking for Infiniband...])
           with_infiniband=yes
           ACX_INFINIBAND
           if test x${HAVE_INFINIBAND} = x0; then
	            AC_MSG_NOTICE([Infiniband can not be used])
              TITLE([Checking for Ethernet])
              ACX_ETHERNET
              if test x${HAVE_TCP} = x0; then
              	 AC_MSG_ERROR([Neither Infiniband nor Ethernet are usable])
              fi
           fi
        fi

	# COPY DEFAULT FILES FOR TESTING
        AM_CONDITIONAL([WITH_OFI], [test x${HAVE_OFI} = x1])
        if [test x${HAVE_OFI} = x1]; then
          options="$options OFI"
        fi
        AM_CONDITIONAL([WITH_ETHERNET], test x${HAVE_TCP} = x1)
        if [test x${HAVE_TCP} = x1]; then
           options="$options Ethernet"
        fi
        AM_CONDITIONAL([WITH_INFINIBAND],[test x${HAVE_INFINIBAND} = x1])
        AM_CONDITIONAL([WITH_IB_DC],[test x${HAVE_INFINIBAND} = x1 -a x${HAVE_IB_DC} = x1])

	# Record the resolved device name so the test runner can match
	# per-test XFAIL markers. Derived from the HAVE_* truth (not the
	# --with-* request, which may differ after the IB->Ethernet fallback).
	AS_IF([test x${HAVE_INFINIBAND} = x1], [GPI2_DEVICE_NAME=ib],
	      [test x${HAVE_OFI} = x1],        [GPI2_DEVICE_NAME=ofi],
	      [test x${HAVE_TCP} = x1],        [GPI2_DEVICE_NAME=tcp],
	      [GPI2_DEVICE_NAME=unknown])
	AC_SUBST([GPI2_DEVICE_NAME])
        if [test x${HAVE_INFINIBAND} = x1]; then
           options="$options Infiniband"
	   if [test x${HAVE_IB_DC} = x1]; then
	      options="$options (with DC)"
	   fi
	   if [test x${HAVE_INFINIBAND_DEVICES} = x1]; then
	      options="$options with actual devices"
	   else
	      options="$options without actual devices"
	   fi
        fi
	])

################################################
# Check and set INFINIBAND path
# ----------------------------------
AC_DEFUN([ACX_INFINIBAND],[
	if test "x$with_infiniband" != xno; then
   	   if test "x$with_infiniband" != xyes; then
	      # User specifies path(s)
	      ac_path_infiniband=$with_infiniband
      	      ac_inc_infiniband=$ac_path_infiniband/include/infiniband
	      AC_CHECK_FILE($ac_inc_infiniband/verbs.h,
	      	      [HAVE_INF_HEADER=1],[HAVE_INF_HEADER=0])
	      for iblib in libibverbs.so libibverbs.a; do
	          for iblib_path in lib lib64; do
	      	      ac_lib_infiniband=$ac_path_infiniband/$iblib_path
		            AC_CHECK_FILE($ac_lib_infiniband/$iblib,[HAVE_INF_LIB=1],[HAVE_INF_LIB=0])
		            if test ${HAVE_INF_LIB} = 1; then
	      	          break
		            fi
		        done
	        if test ${HAVE_INF_LIB} = 1; then
	           break
	  	    fi
  	    done
  else
	      # Try to determine include path(s)
	      inc_paths=`cpp -v /dev/null >& cppt`
	      inc_paths=`sed -n '/^#include </,/^End/p' cppt | sed '1d;$d'`
	      rm -f cppt
	      for ibinc in $inc_paths; do
	      	  ac_inc_infiniband=$ibinc/infiniband
		  AC_CHECK_FILE($ac_inc_infiniband/verbs.h,
		  	      	  [HAVE_INF_HEADER=1],[HAVE_INF_HEADER=0])
	      done
	      # Try to determine library path(s)
	      for ibinc in $inc_paths; do
	      	  ac_path_infiniband=${ibinc%/include*}
		  for iblib in libibverbs.so libibverbs.a; do
	              for iblib_path in lib lib64; do
	      	      	  ac_lib_infiniband=$ac_path_infiniband/$iblib_path
		  	  AC_CHECK_FILE($ac_lib_infiniband/$iblib,[HAVE_INF_LIB=1],[HAVE_INF_LIB=0])
		          if test ${HAVE_INF_LIB} = 1; then
	      	      	     break
		          fi
		      done
	              if test ${HAVE_INF_LIB} = 1; then
	              	 break
	  	      fi
  		  done
		  if test ${HAVE_INF_LIB} = 1; then
	             break
	  	  fi
	      done
	      # If the above lib search fails, use autotools
	      if test ${HAVE_INF_LIB} != 1; then
 	         ac_lib_infiniband=
	      	 AC_CHECK_LIB([ibverbs],[ibv_open_device],[HAVE_INF_LIB=1],[HAVE_INF_LIB=0])
  	      fi
   	   fi
	fi
	if test ${HAVE_INF_HEADER} = 1 -a ${HAVE_INF_LIB} = 1; then
	   ACX_IB_DEVI($ac_inc_infiniband,$ac_lib_infiniband,[HAVE_INFINIBAND=1],[HAVE_INFINIBAND=0])
	   ACX_IB_DC($ac_inc_infiniband,$ac_lib_infiniband,[HAVE_IB_DC=1],[HAVE_IB_DC=0])
	   if test x${HAVE_IB_DC} = x1; then
	      ac_lib_ib_dc=" -lmlx5"
	   fi
	   AC_SUBST(ac_inc_infiniband,[-I$ac_inc_infiniband])
	   if test ! -z $ac_lib_infiniband; then
	      AC_SUBST(ac_lib_infiniband,["-L$ac_lib_infiniband -libverbs$ac_lib_ib_dc"])
	   else
	      AC_SUBST(ac_lib_infiniband,["-libverbs$ac_lib_ib_dc"])
	   fi
	else
	   HAVE_INFINIBAND=0
	fi
	])

AC_DEFUN([ACX_IB_DEVI],[
	AC_MSG_CHECKING([whether ibverbs contains IBV_LINK_LAYER_ETHERNET and basic device routines])

	AC_LANG_PUSH(C)

cat >conftest_ib.c <<_ACEOF
#include <verbs.h>
#include <assert.h>
#include <stdio.h>

int main(){
int a = IBV_LINK_LAYER_ETHERNET;
struct ibv_device **device_list;
int num_devices;
device_list = ibv_get_device_list(&num_devices);
if (device_list != NULL){
ibv_free_device_list(device_list);
}
return num_devices;
}
_ACEOF

	OLD_CFLAGS=$CFLAGS
	if test ! -z $2; then
	   CFLAGS="$AM_CFLAGS $CFLAGS -Wno-unused-variable -I$1 -L$2 -libverbs"
	else
	   CFLAGS="$AM_CFLAGS $CFLAGS -Wno-unused-variable -I$1 -libverbs"
        fi
	AS_IF($CC conftest_ib.c $CFLAGS -o conftest_ib.exe,
	  [AC_MSG_RESULT([yes]);
	  $3],
	  [AC_MSG_RESULT([no]);
	  $4]
	  )
	CFLAGS=$OLD_CFLAGS
	AC_LANG_POP([C])

AC_MSG_CHECKING([whether there are actual IB devices])
AS_IF(test `./conftest_ib.exe; echo $?` -gt 0,
	   [AC_MSG_RESULT([yes]);
	   HAVE_INFINIBAND_DEVICES=1],
	   [AC_MSG_RESULT([no]);
	   HAVE_INFINIBAND_DEVICES=0]
	   )
])

# Dynamically Connected transport (GASPI_IB_DCT): mlx5 direct verbs with
# DC queue pairs and the extended post API. The result is cached, so
# "./configure gpi2_cv_ib_dc=no" builds without it.
AC_DEFUN([ACX_IB_DC],[
	AC_CACHE_CHECK([whether mlx5dv supports the Dynamically Connected transport],
	[gpi2_cv_ib_dc],[

	AC_LANG_PUSH(C)

cat >conftest_ib_dc.c <<_ACEOF
#include <infiniband/mlx5dv.h>

int main(int argc, char *argv[]){
struct ibv_qp_init_attr_ex attr = { .qp_type = IBV_QPT_DRIVER };
struct mlx5dv_qp_init_attr dv_attr = {
  .comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC,
  .dc_init_attr = { .dc_type = MLX5DV_DCTYPE_DCI } };
if (argc > 1){
struct ibv_qp *qp = mlx5dv_create_qp(NULL, &attr, &dv_attr);
struct ibv_qp_ex *qpx = ibv_qp_to_qp_ex(qp);
ibv_wr_start(qpx);
mlx5dv_wr_set_dc_addr(mlx5dv_qp_ex_from_ibv_qp_ex(qpx), NULL, 0, 0);
return ibv_wr_complete(qpx) + mlx5dv_is_supported(NULL);
}
return 0;
}
_ACEOF

	OLD_CFLAGS=$CFLAGS
	if test ! -z $2; then
	   CFLAGS="$AM_CFLAGS $CFLAGS -I$1/.. -L$2 -lmlx5 -libverbs"
	else
	   CFLAGS="$AM_CFLAGS $CFLAGS -I$1/.. -lmlx5 -libverbs"
	fi
	AS_IF($CC conftest_ib_dc.c $CFLAGS -o conftest_ib_dc.exe > /dev/null 2>&1,
	  [gpi2_cv_ib_dc=yes],
	  [gpi2_cv_ib_dc=no]
	  )
	CFLAGS=$OLD_CFLAGS
	rm -f conftest_ib_dc.c conftest_ib_dc.exe
	AC_LANG_POP([C])
	])
	AS_IF([test x$gpi2_cv_ib_dc = xyes], [$3], [$4])
])

################################################
# Check and set OFI path
# ----------------------------------
AC_DEFUN([ACX_OFI],[
	HAVE_OFI_HEADER=0
	HAVE_OFI_LIB=0
	ofi_cflags=
	ofi_libs=

	if test "x$with_ofi" != xno; then
	   AC_PATH_PROG([PKG_CONFIG],[pkg-config],[no])
	   ofi_triplet=`$CC -dumpmachine 2>/dev/null`

	   if test "x$with_ofi" != xyes; then
	      # --- User specifies a prefix path ---
	      ac_path_ofi=$with_ofi

	      # 1. Try pkg-config with prefix-local paths
	      if test "$PKG_CONFIG" != no; then
	         ofi_pc_path="$ac_path_ofi/lib/pkgconfig:$ac_path_ofi/lib64/pkgconfig:$ac_path_ofi/share/pkgconfig"
	         if test -n "$ofi_triplet"; then
	            ofi_pc_path="$ofi_pc_path:$ac_path_ofi/lib/$ofi_triplet/pkgconfig"
	         fi
	         if PKG_CONFIG_PATH="$ofi_pc_path" $PKG_CONFIG --exists libfabric 2>/dev/null; then
	            ofi_cflags=`PKG_CONFIG_PATH="$ofi_pc_path" $PKG_CONFIG --cflags libfabric`
	            ofi_libs=`PKG_CONFIG_PATH="$ofi_pc_path" $PKG_CONFIG --libs libfabric`
	            HAVE_OFI_HEADER=1
	            HAVE_OFI_LIB=1
	            AC_MSG_NOTICE([OFI: found via pkg-config under $ac_path_ofi])
	         fi
	      fi

	      # 2. Manual probing fallback
	      if test $HAVE_OFI_HEADER = 0; then
	         AC_CHECK_FILE($ac_path_ofi/include/rdma/fi_endpoint.h,
	         	[HAVE_OFI_HEADER=1; ofi_cflags="-I$ac_path_ofi/include"],
	         	[HAVE_OFI_HEADER=0])
	      fi
	      if test $HAVE_OFI_LIB = 0; then
	         for ofilib in libfabric.so libfabric.a; do
	             for ofilib_path in lib lib64 lib/$ofi_triplet; do
	                 AC_CHECK_FILE($ac_path_ofi/$ofilib_path/$ofilib,
	                 	[HAVE_OFI_LIB=1; ofi_libs="-L$ac_path_ofi/$ofilib_path -lfabric"],
	                 	[HAVE_OFI_LIB=0])
	                 if test $HAVE_OFI_LIB = 1; then
	                    break 2
	                 fi
	             done
	         done
	      fi

	   else
	      # --- Auto-detect ---

	      # 1. Try system pkg-config
	      if test "$PKG_CONFIG" != no && $PKG_CONFIG --exists libfabric 2>/dev/null; then
	         ofi_cflags=`$PKG_CONFIG --cflags libfabric`
	         ofi_libs=`$PKG_CONFIG --libs libfabric`
	         HAVE_OFI_HEADER=1
	         HAVE_OFI_LIB=1
	         AC_MSG_NOTICE([OFI: found via pkg-config])
	      else
	         # 2. Scan compiler include paths for rdma/fi_endpoint.h
	         inc_paths=`cpp -v /dev/null >& cppt`
	         inc_paths=`sed -n '/^#include </,/^End/p' cppt | sed '1d;$d'`
	         rm -f cppt
	         for ofiinc in $inc_paths; do
	             AC_CHECK_FILE($ofiinc/rdma/fi_endpoint.h,
	             	[HAVE_OFI_HEADER=1; ofi_cflags="-I$ofiinc"],
	             	[HAVE_OFI_HEADER=0])
	             if test $HAVE_OFI_HEADER = 1; then
	                break
	             fi
	         done
	         # 3. Scan lib paths (lib, lib64, multiarch)
	         for ofiinc in $inc_paths; do
	             ac_path_ofi=${ofiinc%/include*}
	             for ofilib in libfabric.so libfabric.a; do
	                 for ofilib_path in lib lib64 lib/$ofi_triplet; do
	                     AC_CHECK_FILE($ac_path_ofi/$ofilib_path/$ofilib,
	                     	[HAVE_OFI_LIB=1; ofi_libs="-L$ac_path_ofi/$ofilib_path -lfabric"],
	                     	[HAVE_OFI_LIB=0])
	                     if test $HAVE_OFI_LIB = 1; then
	                        break 3
	                     fi
	                 done
	             done
	         done
	         # 4. Last resort: let the linker find it
	         if test $HAVE_OFI_LIB != 1; then
	            AC_CHECK_LIB([fabric],[fi_endpoint],
	            	[HAVE_OFI_LIB=1; ofi_libs="-lfabric"],
	            	[HAVE_OFI_LIB=0])
	         fi
	      fi
	   fi
	fi

	if test $HAVE_OFI_HEADER = 1 -a $HAVE_OFI_LIB = 1; then
	   ACX_OFI_SMOKE([$ofi_cflags],[$ofi_libs],[HAVE_OFI=1],[HAVE_OFI=0])
	else
	   HAVE_OFI=0
	fi

	if test $HAVE_OFI = 1; then
	   # A run-time path to a libfabric outside the linker's default
	   # directories: ranks need not inherit LD_LIBRARY_PATH (ssh).
	   for ofi_flag in $ofi_libs; do
	      case $ofi_flag in
	         -L*)
	            ofi_dir=`echo "$ofi_flag" | sed 's/^-L//'`
	            case $ofi_dir in
	               /lib|/lib64|/usr/lib|/usr/lib64|/lib/$ofi_triplet|/usr/lib/$ofi_triplet) ;;
	               *) ofi_libs="$ofi_libs -Wl,-rpath,$ofi_dir"
	                  AC_MSG_NOTICE([OFI: run-time path $ofi_dir]) ;;
	            esac ;;
	      esac
	   done
	   AC_SUBST(ac_inc_ofi,[$ofi_cflags])
	   AC_SUBST(ac_lib_ofi,[$ofi_libs])
	else
	   if test $HAVE_OFI_HEADER = 0 -a $HAVE_OFI_LIB = 0; then
	      AC_MSG_NOTICE([OFI: could not find libfabric headers or library.])
	      AC_MSG_NOTICE([  Debian/Ubuntu: apt install libfabric-dev])
	      AC_MSG_NOTICE([  RHEL/Fedora:   dnf install libfabric-devel])
	      if test "x$with_ofi" != xyes -a "$PKG_CONFIG" = no; then
	         AC_MSG_NOTICE([  Tip: installing pkg-config improves library detection.])
	      fi
	   fi
	fi
	])

################################################
# OFI compile and link smoke test
# ----------------------------------
AC_DEFUN([ACX_OFI_SMOKE],[
	AC_MSG_CHECKING([whether OFI headers and libraries are usable])
	AC_LANG_PUSH(C)

cat >conftest_ofi.c <<_ACEOF
#include <rdma/fabric.h>
int main(){
  struct fi_info *hints = fi_allocinfo();
  fi_freeinfo(hints);
  return 0;
}
_ACEOF

	OLD_CFLAGS=$CFLAGS
	CFLAGS="$AM_CFLAGS $CFLAGS $1 $2"
	AS_IF($CC conftest_ofi.c $CFLAGS -o conftest_ofi.exe,
	  [AC_MSG_RESULT([yes]); $3],
	  [AC_MSG_RESULT([no]); $4]
	)
	CFLAGS=$OLD_CFLAGS
	rm -f conftest_ofi.c conftest_ofi.exe
	AC_LANG_POP([C])
	])

################################################
# Check and set ETHERNET path
# ----------------------------------
AC_DEFUN([ACX_ETHERNET],[
	AC_CHECK_HEADER(netinet/tcp.h,[HAVE_TCP=1],[HAVE_TCP=0])
	])
