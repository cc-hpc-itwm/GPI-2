include ../Make.inc

## COLLECTIVES
collectives_tests = ../../bin/allreduce.bin \
		  ../../bin/allreduce_user_grp.bin \
		  ../../bin/allreduce_user_fun.bin \
		  ../../bin/allreduce_user_type.bin \
                  ../../bin/allreduce_user_grp_fun.bin \
		  ../../bin/allreduce_utils.bin \
		  ../../bin/allreduce_vlarge.bin \
		  ../../bin/barrier.bin \
		  ../../bin/barrier_grp.bin \
		  ../../bin/barrier_timeout.bin \
		  ../../bin/loop_barrier_group_timeout.bin \
		  ../../bin/mtt_barrier_different_grps.bin \
		  ../../bin/mtt_barrier_allreduce_different_grps.bin

______bin_allreduce_bin_SOURCES = allreduce.c $(auxSRCS)
______bin_allreduce_user_grp_bin_SOURCES = allreduce_user_grp.c $(auxSRCS)
______bin_allreduce_user_fun_bin_SOURCES = allreduce_user_fun.c $(auxSRCS)
______bin_allreduce_user_type_bin_SOURCES = allreduce_user_type.c $(auxSRCS)
______bin_allreduce_user_grp_fun_bin_SOURCES = allreduce_user_grp_fun.c $(auxSRCS)
______bin_allreduce_utils_bin_SOURCES = allreduce_utils.c $(auxSRCS)
______bin_allreduce_vlarge_bin_SOURCES = allreduce_vlarge.c $(auxSRCS)
______bin_barrier_bin_SOURCES = barrier.c $(auxSRCS)
______bin_barrier_grp_bin_SOURCES = barrier_grp.c $(auxSRCS)
______bin_barrier_timeout_bin_SOURCES = barrier_timeout.c $(auxSRCS)
______bin_loop_barrier_group_timeout_bin_SOURCES = loop_barrier_group_timeout.c $(auxSRCS)
______bin_mtt_barrier_different_grps_bin_SOURCES = mtt_barrier_different_grps.c $(auxSRCS)
______bin_mtt_barrier_allreduce_different_grps_bin_SOURCES = mtt_barrier_allreduce_different_grps.c $(auxSRCS)

test_pkglibexec_PROGRAMS = $(collectives_tests)
