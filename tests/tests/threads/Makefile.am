include ../Make.inc

## THREADS
threads_tests = ../../bin/threads_init.bin \
		../../bin/threads_sync_stress.bin

______bin_threads_init_bin_SOURCES = threads_init.c $(auxSRCS)
______bin_threads_sync_stress_bin_SOURCES = threads_sync_stress.c $(auxSRCS)

noinst_PROGRAMS = $(threads_tests)
