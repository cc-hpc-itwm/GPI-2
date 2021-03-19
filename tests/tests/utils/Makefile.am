include ../Make.inc

## UTILS
utils_tests = ../../bin/error_str.bin \
	    ../../bin/print_error.bin \
	    ../../bin/print_error2.bin \
	    ../../bin/time_get.bin

______bin_error_str_bin_SOURCES = error_str.c $(auxSRCS)
______bin_print_error_bin_SOURCES = print_error.c $(auxSRCS)
______bin_print_error2_bin_SOURCES = print_error2.c $(auxSRCS)
______bin_time_get_bin_SOURCES =  time_get.c $(auxSRCS)

test_pkglibexec_PROGRAMS = $(utils_tests)
