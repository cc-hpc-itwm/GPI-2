include ../Make.inc

## EXTENSIONS
extensions_tests = ../../bin/ping_procs.bin \
		   ../../bin/q_create.bin \
		   ../../bin/seg_avail_local.bin

______bin_ping_procs_bin_SOURCES = ping_procs.c $(auxSRCS)
______bin_q_create_bin_SOURCES = q_create.c $(auxSRCS)
______bin_seg_avail_local_bin_SOURCES = seg_avail_local.c $(auxSRCS)

noinst_PROGRAMS = $(extensions_tests)
