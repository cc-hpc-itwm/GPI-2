AM_CFLAGS = -g
AM_CPPFLAGS = -I$(top_builddir)/src/include

LDADD = -L$(top_builddir)/src/.libs -lGPI2-dbg -lpthread
if WITH_INFINIBAND
LDADD += -L@ac_lib_infiniband@ -libverbs
endif

if WITH_FORTRAN
AM_FCFLAGS = -I$(top_builddir)/src/include @FCFLAGS_TESTS@
endif

if WITH_MPI
AM_CFLAGS += -DGPI2_WITH_MPI
if WITH_MPI_USER
AM_CFLAGS += -I@ac_inc_mpi@
LDADD += -L@ac_lib_mpi@ -lmpi
endif
if WITH_FORTRAN
AM_FCFLAGS += -DGPI2_WITH_MPI
endif
endif

AM_CXXFLAGS = -std=c++11 $(AM_CFLAGS) -fpermissive

# TESTS
auxSRCS = test_utils.c test_utils.h

## ATOMIC
atomic_tests = ../bin/atomic_fail_no_seg.bin \
	     ../bin/compare_swap.bin \
	     ../bin/compare_swap_nobuild.bin \
	     ../bin/compare_swap_test.bin \
	     ../bin/fetch_add.bin \
	     ../bin/fetch_add_loop.bin \
	     ../bin/fetch_add_nobuild.bin \
	     ../bin/fetch_add_test.bin \
	     ../bin/fetch_add_unalignd.bin

___bin_atomic_fail_no_seg_bin_SOURCES = atomic/atomic_fail_no_seg.c $(auxSRCS)
___bin_compare_swap_bin_SOURCES = atomic/compare_swap.c $(auxSRCS)
___bin_compare_swap_nobuild_bin_SOURCES = atomic/compare_swap_nobuild.c $(auxSRCS)
___bin_compare_swap_test_bin_SOURCES = atomic/compare_swap_test.c $(auxSRCS)
___bin_fetch_add_bin_SOURCES = atomic/fetch_add.c $(auxSRCS)
___bin_fetch_add_loop_bin_SOURCES = atomic/fetch_add_loop.c $(auxSRCS)
___bin_fetch_add_nobuild_bin_SOURCES = atomic/fetch_add_nobuild.c $(auxSRCS)
___bin_fetch_add_test_bin_SOURCES = atomic/fetch_add_test.c $(auxSRCS)
___bin_fetch_add_unalignd_bin_SOURCES = atomic/fetch_add_unalignd.c $(auxSRCS)

## CM
cm_tests = ../bin/con_disconnect.bin

___bin_con_disconnect_bin_SOURCES = cm/con_disconnect.c $(auxSRCS)

## COLLECTIVES
collectives_tests = ../bin/allreduce.bin \
		  ../bin/allreduce_user_fun.bin \
		  ../bin/allreduce_user_type.bin \
		  ../bin/allreduce_utils.bin \
		  ../bin/allreduce_vlarge.bin \
		  ../bin/barrier.bin \
		  ../bin/barrier_timeout.bin \
		  ../bin/loop_barrier_group_timeout.bin

___bin_allreduce_bin_SOURCES = collectives/allreduce.c $(auxSRCS)
___bin_allreduce_user_fun_bin_SOURCES = collectives/allreduce_user_fun.c $(auxSRCS)
___bin_allreduce_user_type_bin_SOURCES = collectives/allreduce_user_type.c $(auxSRCS)
___bin_allreduce_utils_bin_SOURCES = collectives/allreduce_utils.c $(auxSRCS)
___bin_allreduce_vlarge_bin_SOURCES = collectives/allreduce_vlarge.c $(auxSRCS)
___bin_barrier_bin_SOURCES = collectives/barrier.c $(auxSRCS)
___bin_barrier_timeout_bin_SOURCES = collectives/barrier_timeout.c $(auxSRCS)
___bin_loop_barrier_group_timeout_bin_SOURCES = collectives/loop_barrier_group_timeout.c $(auxSRCS)

## COMM
comm_tests = \
	../bin/all-to-all.bin \
	../bin/all-to-rank0.bin \
	../bin/comm_limits.bin \
	../bin/queue_full.bin \
	../bin/read_nsizes.bin \
	../bin/read_smalls.bin \
	../bin/read_write.bin \
	../bin/small_writes_notify.bin \
	../bin/strings.bin \
	../bin/write.bin \
	../bin/write_all.bin \
	../bin/write_all_mtt.bin \
	../bin/write_all_nsizes.bin \
	../bin/write_all_nsizes_mtt.bin \
	../bin/write_all_nsizes_nobuild.bin \
	../bin/comm_write_m_to_1.bin \
	../bin/write_right_left.bin \
	../bin/write_simple.bin \
	../bin/write_timeout.bin \
	../bin/z4k_pressure.bin \
	../bin/z4k_pressure_mtt.bin

___bin_all_to_all_bin_SOURCES = comm/all-to-all.c $(auxSRCS)
___bin_all_to_rank0_bin_SOURCES = comm/all-to-rank0.c $(auxSRCS)
___bin_comm_limits_bin_SOURCES = comm/comm_limits.c $(auxSRCS)
___bin_queue_full_bin_SOURCES = comm/queue_full.c $(auxSRCS)
___bin_read_nsizes_bin_SOURCES = comm/read_nsizes.c $(auxSRCS)
___bin_read_smalls_bin_SOURCES = comm/read_smalls.c $(auxSRCS)
___bin_read_write_bin_SOURCES = comm/read_write.c $(auxSRCS)
___bin_small_writes_notify_bin_SOURCES = comm/small_writes_notify.c $(auxSRCS)
___bin_strings_bin_SOURCES = comm/strings.c $(auxSRCS)
___bin_write_bin_SOURCES = comm/write.c $(auxSRCS)
___bin_write_all_bin_SOURCES = comm/write_all.c $(auxSRCS)
___bin_write_all_mtt_bin_SOURCES = comm/write_all_mtt.c $(auxSRCS)
___bin_write_all_nsizes_bin_SOURCES = comm/write_all_nsizes.c $(auxSRCS)
___bin_write_all_nsizes_mtt_bin_SOURCES = comm/write_all_nsizes_mtt.c $(auxSRCS)
___bin_write_all_nsizes_nobuild_bin_SOURCES = comm/write_all_nsizes_nobuild.c $(auxSRCS)
___bin_comm_write_m_to_1_bin_SOURCES = comm/comm_write_m_to_1.c $(auxSRCS)
___bin_write_right_left_bin_SOURCES = comm/write_right_left.c $(auxSRCS)
___bin_write_simple_bin_SOURCES = comm/write_simple.c $(auxSRCS)
___bin_write_timeout_bin_SOURCES = comm/write_timeout.c $(auxSRCS)
___bin_z4k_pressure_bin_SOURCES = comm/z4k_pressure.c $(auxSRCS)
___bin_z4k_pressure_mtt_bin_SOURCES = comm/z4k_pressure_mtt.c $(auxSRCS)

## CONFIG
config_tests = ../bin/set_config.bin \
	../bin/config_netinfo.bin \
	../bin/print_config.bin ##Not include before?

___bin_set_config_bin_SOURCES = config/set_config.c $(auxSRCS)
___bin_config_netinfo_bin_SOURCES = config/config_netinfo.c $(auxSRCS)
___bin_print_config_bin_SOURCES = config/print_config.c $(auxSRCS)

## ERROR_VEC
error_vec_tests = ../bin/barrier_recover.bin \
	../bin/barrier_recover_ping.bin \
	../bin/get_error_vec.bin \
	../bin/one_dies.bin

___bin_barrier_recover_bin_SOURCES = error_vec/barrier_recover.c $(auxSRCS)
___bin_barrier_recover_ping_bin_SOURCES = error_vec/barrier_recover_ping.c $(auxSRCS)
___bin_get_error_vec_bin_SOURCES = error_vec/get_error_vec.c $(auxSRCS)
___bin_one_dies_bin_SOURCES = error_vec/one_dies.c $(auxSRCS)

## EXTENSIONS
extensions_tests = ../bin/ping_procs.bin \
		   ../bin/q_create.bin \
		   ../bin/seg_avail_local.bin

___bin_ping_procs_bin_SOURCES = extensions/ping_procs.c $(auxSRCS)
___bin_q_create_bin_SOURCES = extensions/q_create.c $(auxSRCS)
___bin_seg_avail_local_bin_SOURCES = extensions/seg_avail_local.c $(auxSRCS)

## FORTRAN
if WITH_FORTRAN
fortran_tests = ../bin/fortran_allreduces.bin \
      ../bin/fortran_print.bin \
      ../bin/fortran_seg_ptr.bin \
      ../bin/fortran_seg_use.bin \
      ../bin/fortran_use.bin

___bin_fortran_allreduces_bin_SOURCES = fortran/fortran_allreduces.f90
___bin_fortran_print_bin_SOURCES = fortran/fortran_print.f90
___bin_fortran_seg_ptr_bin_SOURCES = fortran/fortran_seg_ptr.f90
___bin_fortran_seg_use_bin_SOURCES = fortran/fortran_seg_use.f90
___bin_fortran_use_bin_SOURCES = fortran/fortran_use.f90
endif

## GROUPS
groups_tests = ../bin/g_before_start.bin \
      ../bin/g_coll_del_coll.bin \
      ../bin/g_commit_g_all.bin \
      ../bin/g_commit_try_block.bin \
      ../bin/g_elems.bin \
      ../bin/g_force_timeout.bin \
      ../bin/g_max_groups.bin \
      ../bin/g_no_own.bin \
      ../bin/g_num.bin \
      ../bin/g_size.bin \
      ../bin/g_some_from_all.bin

___bin_g_before_start_bin_SOURCES = groups/g_before_start.c $(auxSRCS)
___bin_g_coll_del_coll_bin_SOURCES = groups/g_coll_del_coll.c $(auxSRCS)
___bin_g_commit_g_all_bin_SOURCES = groups/g_commit_g_all.c $(auxSRCS)
___bin_g_commit_try_block_bin_SOURCES = groups/g_commit_try_block.c $(auxSRCS)
___bin_g_elems_bin_SOURCES = groups/g_elems.c $(auxSRCS)
___bin_g_force_timeout_bin_SOURCES = groups/g_force_timeout.c $(auxSRCS)
___bin_g_max_groups_bin_SOURCES = groups/g_max_groups.c $(auxSRCS)
___bin_g_no_own_bin_SOURCES = groups/g_no_own.c $(auxSRCS)
___bin_g_num_bin_SOURCES = groups/g_num.c $(auxSRCS)
___bin_g_size_bin_SOURCES = groups/g_size.c $(auxSRCS)
___bin_g_some_from_all_bin_SOURCES = groups/g_some_from_all.c $(auxSRCS)

## INIT
init_tests = ../bin/proc_init.bin \
	../bin/proc_init_timeout.bin \
	../bin/cmd_line_args.bin \
	../bin/kill_procs.bin \
	../bin/print_to.bin \
	../bin/null_ptrs.bin \
	../bin/numa_check.bin \
	../bin/strong_sym.bin \
	../bin/local_rank.bin \
	../bin/initialized.bin

___bin_proc_init_bin_SOURCES = init/proc_init.c $(auxSRCS)
___bin_proc_init_timeout_bin_SOURCES = init/proc_init_timeout.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_cmd_line_args_bin_SOURCES = init/cmd_line_args.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_kill_procs_bin_SOURCES = init/kill_procs.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_print_to_bin_SOURCES = init/print_to.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_null_ptrs_bin_SOURCES = init/null_ptrs.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_numa_check_bin_SOURCES = init/numa_check.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_strong_sym_bin_SOURCES = init/strong_sym.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_local_rank_bin_SOURCES = init/local_rank.c $(auxSRCS) $(lib_LTLIBRARIES)
___bin_initialized_bin_SOURCES = init/initialized.c $(auxSRCS) $(lib_LTLIBRARIES)

## LISTS
lists_tests = ../bin/list_limit.bin \
      ../bin/list_read.bin \
      ../bin/list_read_notify.bin \
      ../bin/list_write.bin \
      ../bin/list_write_notify.bin

___bin_list_limit_bin_SOURCES = lists/list_limit.c $(auxSRCS)
___bin_list_read_bin_SOURCES = lists/list_read.c $(auxSRCS)
___bin_list_read_notify_bin_SOURCES = lists/list_read_notify.c $(auxSRCS)
___bin_list_write_bin_SOURCES = lists/list_write.c $(auxSRCS)
___bin_list_write_notify_bin_SOURCES = lists/list_write_notify.c $(auxSRCS)

## MPI
if WITH_MPI

mpi_tests = ../bin/mpi_allreduce.bin \
      ../bin/mpi_barrier.bin \
      ../bin/with_mpi.bin

___bin_mpi_allreduce_bin_SOURCES = mpi/mpi_allreduce.c   $(auxSRCS)
___bin_mpi_barrier_bin_SOURCES = mpi/mpi_barrier.c $(auxSRCS)
___bin_with_mpi_bin_SOURCES = mpi/with_mpi.c  $(auxSRCS)

endif

## NOTIFICATIONS
notifications_tests = ../bin/not_zero_wait.bin \
      ../bin/notify_after_delete.bin \
      ../bin/notify_all.bin \
      ../bin/notify_last.bin \
      ../bin/notify_null.bin \
      ../bin/notify_seg_alloc.bin \
      ../bin/read_notify.bin \
      ../bin/write_m_to_1.bin \
      ../bin/write_notify.bin \
      ../bin/write_notify_outstanding.bin
#      ../bin/notification_value.bin

___bin_not_zero_wait_bin_SOURCES = notifications/not_zero_wait.c $(auxSRCS)
#___bin_notification_value_bin_SOURCES = notifications/notification_value.cpp $(auxSRCS)
___bin_notify_after_delete_bin_SOURCES = notifications/notify_after_delete.c $(auxSRCS)
___bin_notify_all_bin_SOURCES = notifications/notify_all.c $(auxSRCS)
___bin_notify_last_bin_SOURCES = notifications/notify_last.c $(auxSRCS)
___bin_notify_null_bin_SOURCES = notifications/notify_null.c $(auxSRCS)
___bin_notify_seg_alloc_bin_SOURCES = notifications/notify_seg_alloc.c $(auxSRCS)
___bin_read_notify_bin_SOURCES = notifications/read_notify.c $(auxSRCS)
___bin_write_m_to_1_bin_SOURCES = notifications/write_m_to_1.c $(auxSRCS)
___bin_write_notify_bin_SOURCES = notifications/write_notify.c $(auxSRCS)
___bin_write_notify_outstanding_bin_SOURCES = notifications/write_notify_outstanding.c $(auxSRCS)

## PASSIVE
passive_tests = ../bin/passive.bin \
      ../bin/passive_limits.bin \
      ../bin/passive_n.bin \
      ../bin/passive_nbytes.bin \
      ../bin/passive_own.bin \
      ../bin/passive_own_block.bin \
      ../bin/passive_recvtimeout.bin \
      ../bin/passive_tout_test.bin

___bin_passive_bin_SOURCES = passive/passive.c $(auxSRCS)
___bin_passive_limits_bin_SOURCES = passive/passive_limits.c $(auxSRCS)
___bin_passive_n_bin_SOURCES = passive/passive_n.c $(auxSRCS)
___bin_passive_nbytes_bin_SOURCES = passive/passive_nbytes.c $(auxSRCS)
___bin_passive_own_bin_SOURCES = passive/passive_own.c $(auxSRCS)
___bin_passive_own_block_bin_SOURCES = passive/passive_own_block.c $(auxSRCS)
___bin_passive_recvtimeout_bin_SOURCES = passive/passive_recvtimeout.c $(auxSRCS)
___bin_passive_tout_test_bin_SOURCES = passive/passive_tout_test.c $(auxSRCS)

## SEGMENTS
segments_tests = ../bin/max_mem.bin \
      ../bin/seg_alloc_all.bin \
      ../bin/seg_alloc_diff.bin \
      ../bin/seg_avail_locally.bin \
      ../bin/seg_create_all.bin \
      ../bin/seg_create_same.bin \
      ../bin/seg_id_list.bin \
      ../bin/seg_reuse.bin \
      ../bin/seg_reuse_buffer.bin \
      ../bin/seg_subsegments.bin \
      ../bin/seg_use_buffer.bin
#      ../bin/seg_shmem.bin

___bin_max_mem_bin_SOURCES = segments/max_mem.c $(auxSRCS)
___bin_seg_alloc_all_bin_SOURCES = segments/seg_alloc_all.c $(auxSRCS)
___bin_seg_alloc_diff_bin_SOURCES = segments/seg_alloc_diff.c $(auxSRCS)
___bin_seg_avail_locally_bin_SOURCES = segments/seg_avail_locally.c $(auxSRCS)
___bin_seg_create_all_bin_SOURCES = segments/seg_create_all.c $(auxSRCS)
___bin_seg_create_same_bin_SOURCES = segments/seg_create_same.c $(auxSRCS)
___bin_seg_id_list_bin_SOURCES = segments/seg_id_list.c $(auxSRCS)
___bin_seg_reuse_bin_SOURCES = segments/seg_reuse.c $(auxSRCS)
___bin_seg_reuse_buffer_bin_SOURCES = segments/seg_reuse_buffer.c $(auxSRCS)
#___bin_seg_shmem_bin_SOURCES = segments/seg_shmem.cpp $(auxSRCS)
___bin_seg_subsegments_bin_SOURCES = segments/seg_subsegments.c $(auxSRCS)
___bin_seg_use_buffer_bin_SOURCES = segments/seg_use_buffer.c $(auxSRCS)

## THREADS
threads_tests = ../bin/threads_init.bin \
		../bin/threads_sync_stress.bin

___bin_threads_init_bin_SOURCES = threads/threads_init.c $(auxSRCS)
___bin_threads_sync_stress_bin_SOURCES = threads/threads_sync_stress.c $(auxSRCS)

## UTILS
utils_tests = ../bin/error_str.bin \
	    ../bin/print_error.bin \
	    ../bin/print_error2.bin \
	    ../bin/time_get.bin

___bin_error_str_bin_SOURCES = utils/error_str.c $(auxSRCS)
___bin_print_error_bin_SOURCES = utils/print_error.c $(auxSRCS)
___bin_print_error2_bin_SOURCES = utils/print_error2.c $(auxSRCS)
___bin_time_get_bin_SOURCES =  utils/time_get.c $(auxSRCS)

noinst_PROGRAMS = $(atomic_tests) \
		$(cm_tests) \
		$(collectives_tests) \
		$(comm_tests) \
		$(config_tests) \
		$(error_vec_tests) \
		$(extensions_tests) \
		$(groups_tests) \
		$(init_tests) \
		$(lists_tests) \
		$(notifications_tests) \
		$(passive_tests) \
		$(segments_tests) \
		$(threads_tests) \
		$(utils_tests)

if WITH_FORTRAN
noinst_PROGRAMS += $(fortran_tests)
endif

if WITH_MPI
noinst_PROGRAMS += $(mpi_tests)
endif

CLEANFILES = *~ *.bak *.bin fortran/*.mod
