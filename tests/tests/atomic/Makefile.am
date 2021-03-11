include ../Make.inc

## ATOMIC
atomic_tests = ../../bin/atomic_fail_no_seg.bin \
	     ../../bin/compare_swap.bin \
	     ../../bin/compare_swap_nobuild.bin \
	     ../../bin/compare_swap_test.bin \
	     ../../bin/fetch_add.bin \
	     ../../bin/fetch_add_loop.bin \
	     ../../bin/fetch_add_nobuild.bin \
	     ../../bin/fetch_add_test.bin \
	     ../../bin/fetch_add_unalignd.bin

______bin_atomic_fail_no_seg_bin_SOURCES = atomic_fail_no_seg.c $(auxSRCS)
______bin_compare_swap_bin_SOURCES = compare_swap.c $(auxSRCS)
______bin_compare_swap_nobuild_bin_SOURCES = compare_swap_nobuild.c $(auxSRCS)
______bin_compare_swap_test_bin_SOURCES = compare_swap_test.c $(auxSRCS)
______bin_fetch_add_bin_SOURCES = fetch_add.c $(auxSRCS)
______bin_fetch_add_loop_bin_SOURCES = fetch_add_loop.c $(auxSRCS)
______bin_fetch_add_nobuild_bin_SOURCES = fetch_add_nobuild.c $(auxSRCS)
______bin_fetch_add_test_bin_SOURCES = fetch_add_test.c $(auxSRCS)
______bin_fetch_add_unalignd_bin_SOURCES = fetch_add_unalignd.c $(auxSRCS)

test_pkglibexec_PROGRAMS = $(atomic_tests)
