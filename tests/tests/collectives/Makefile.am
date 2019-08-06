include ../Make.inc

## COLLECTIVES
collectives_tests = ../../bin/allreduce.bin \
		  ../../bin/allreduce_user_fun.bin \
		  ../../bin/allreduce_user_type.bin \
		  ../../bin/allreduce_utils.bin \
		  ../../bin/allreduce_vlarge.bin \
		  ../../bin/barrier.bin \
		  ../../bin/barrier_timeout.bin \
		  ../../bin/loop_barrier_group_timeout.bin

______bin_allreduce_bin_SOURCES = allreduce.c $(auxSRCS)
______bin_allreduce_user_fun_bin_SOURCES = allreduce_user_fun.c $(auxSRCS)
______bin_allreduce_user_type_bin_SOURCES = allreduce_user_type.c $(auxSRCS)
______bin_allreduce_utils_bin_SOURCES = allreduce_utils.c $(auxSRCS)
______bin_allreduce_vlarge_bin_SOURCES = allreduce_vlarge.c $(auxSRCS)
______bin_barrier_bin_SOURCES = barrier.c $(auxSRCS)
______bin_barrier_timeout_bin_SOURCES = barrier_timeout.c $(auxSRCS)
______bin_loop_barrier_group_timeout_bin_SOURCES = loop_barrier_group_timeout.c $(auxSRCS)

noinst_PROGRAMS = $(collectives_tests)
