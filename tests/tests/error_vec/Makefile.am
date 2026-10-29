include ../Make.inc

 ## ERROR_VEC
error_vec_tests = \
	../../bin/barrier_recover.bin \
	../../bin/barrier_recover_ping.bin \
	../../bin/get_error_vec.bin \
	../../bin/one_dies.bin \
	../../bin/one_dies_passive_atomic.bin \
	../../bin/passive_send_dead.bin

______bin_barrier_recover_bin_SOURCES = barrier_recover.c $(auxSRCS)
______bin_barrier_recover_ping_bin_SOURCES = barrier_recover_ping.c $(auxSRCS)
______bin_get_error_vec_bin_SOURCES = get_error_vec.c $(auxSRCS)
______bin_one_dies_bin_SOURCES = one_dies.c $(auxSRCS)
______bin_one_dies_passive_atomic_bin_SOURCES = one_dies_passive_atomic.c $(auxSRCS)
______bin_passive_send_dead_bin_SOURCES = passive_send_dead.c $(auxSRCS)


test_pkglibexec_PROGRAMS = $(error_vec_tests)
