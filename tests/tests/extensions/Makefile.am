include ../Make.inc

## EXTENSIONS
extensions_tests = ../../bin/ping_procs.bin \
		   ../../bin/q_create_use_queue.bin \
		   ../../bin/q_create_delete_wait.bin \
		   ../../bin/seg_avail_local.bin \
		   ../../bin/seg_avail_local_usermsegs.bin

______bin_ping_procs_bin_SOURCES = ping_procs.c $(auxSRCS)
______bin_q_create_use_queue_bin_SOURCES = q_create_use_queue.c $(auxSRCS)
______bin_q_create_delete_wait_bin_SOURCES = q_create_delete_wait.c $(auxSRCS)
______bin_seg_avail_local_bin_SOURCES = seg_avail_local.c $(auxSRCS)
______bin_seg_avail_local_usermsegs_bin_SOURCES = seg_avail_local_usermsegs.c $(auxSRCS)

test_pkglibexec_PROGRAMS = $(extensions_tests)
