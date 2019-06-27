LDADD = -lGPI2dbg -lpthread
if WITH_INFINIBAND
LDADD += -L@ac_lib_infiniband@ -libverbs
endif
AM_CFLAGS = $(libGPI2dbg_la_CFLAGS)
AM_CPPFLAGS = -Itests/tests $(libGPI2dbg_la_CPPFLAGS)
AM_CXXFLAGS = -std=c++11 $(AM_CFLAGS) $(AM_CPPFLAGS)

# TESTS
auxSRCS = $(LTLIBRARIES) tests/tests/test_utils.c

## ATOMIC
atomic_tests = tests/bin/atomic_fail_no_seg.bin \
	     tests/bin/compare_swap.bin \
	     tests/bin/compare_swap_nobuild.bin \
	     tests/bin/compare_swap_test.bin \
	     tests/bin/fetch_add.bin \
	     tests/bin/fetch_add_loop.bin \
	     tests/bin/fetch_add_nobuild.bin \
	     tests/bin/fetch_add_test.bin \
	     tests/bin/fetch_add_unalignd.bin

tests_bin_atomic_fail_no_seg_bin_SOURCES = tests/tests/atomic/atomic_fail_no_seg.c $(auxSRCS)
tests_bin_compare_swap_bin_SOURCES = tests/tests/atomic/compare_swap.c $(auxSRCS)
tests_bin_compare_swap_nobuild_bin_SOURCES = tests/tests/atomic/compare_swap_nobuild.c $(auxSRCS)
tests_bin_compare_swap_test_bin_SOURCES = tests/tests/atomic/compare_swap_test.c $(auxSRCS)
tests_bin_fetch_add_bin_SOURCES = tests/tests/atomic/fetch_add.c $(auxSRCS)
tests_bin_fetch_add_loop_bin_SOURCES = tests/tests/atomic/fetch_add_loop.c $(auxSRCS)
tests_bin_fetch_add_nobuild_bin_SOURCES = tests/tests/atomic/fetch_add_nobuild.c $(auxSRCS)
tests_bin_fetch_add_test_bin_SOURCES = tests/tests/atomic/fetch_add_test.c $(auxSRCS)
tests_bin_fetch_add_unalignd_bin_SOURCES = tests/tests/atomic/fetch_add_unalignd.c $(auxSRCS)

## CM
cm_tests = tests/bin/con_disconnect.bin

tests_bin_con_disconnect_bin_SOURCES = tests/tests/cm/con_disconnect.c $(auxSRCS)

## COLLECTIVES
collectives_tests = tests/bin/allreduce.bin \
		  tests/bin/allreduce_user_fun.bin \
		  tests/bin/allreduce_user_type.bin \
		  tests/bin/allreduce_utils.bin \
		  tests/bin/allreduce_vlarge.bin \
		  tests/bin/barrier.bin \
		  tests/bin/barrier_timeout.bin \
		  tests/bin/loop_barrier_group_timeout.bin

tests_bin_allreduce_bin_SOURCES = tests/tests/collectives/allreduce.c $(auxSRCS)
tests_bin_allreduce_user_fun_bin_SOURCES =		tests/tests/collectives/allreduce_user_fun.c $(auxSRCS)
tests_bin_allreduce_user_type_bin_SOURCES =		tests/tests/collectives/allreduce_user_type.c $(auxSRCS)
tests_bin_allreduce_utils_bin_SOURCES =			tests/tests/collectives/allreduce_utils.c $(auxSRCS)
tests_bin_allreduce_vlarge_bin_SOURCES =			tests/tests/collectives/allreduce_vlarge.c $(auxSRCS)
tests_bin_barrier_bin_SOURCES =				tests/tests/collectives/barrier.c $(auxSRCS)
tests_bin_barrier_timeout_bin_SOURCES =			tests/tests/collectives/barrier_timeout.c $(auxSRCS)
tests_bin_loop_barrier_group_timeout_bin_SOURCES =        tests/tests/collectives/loop_barrier_group_timeout.c $(auxSRCS)

## COMM
comm_tests = \
	tests/bin/all-to-all.bin \
	tests/bin/all-to-rank0.bin \
	tests/bin/comm_limits.bin \
	tests/bin/queue_full.bin \
	tests/bin/read_nsizes.bin \
	tests/bin/read_smalls.bin \
	tests/bin/read_write.bin \
	tests/bin/small_writes_notify.bin \
	tests/bin/strings.bin \
	tests/bin/write.bin \
	tests/bin/write_all.bin \
	tests/bin/write_all_mtt.bin \
	tests/bin/write_all_nsizes.bin \
	tests/bin/write_all_nsizes_mtt.bin \
	tests/bin/write_all_nsizes_nobuild.bin \
	tests/bin/comm_write_m_to_1.bin \
	tests/bin/write_right_left.bin \
	tests/bin/write_simple.bin \
	tests/bin/write_timeout.bin \
	tests/bin/z4k_pressure.bin \
	tests/bin/z4k_pressure_mtt.bin

tests_bin_all_to_all_bin_SOURCES =                   tests/tests/comm/all-to-all.c $(auxSRCS)
tests_bin_all_to_rank0_bin_SOURCES =		     tests/tests/comm/all-to-rank0.c $(auxSRCS)
tests_bin_comm_limits_bin_SOURCES =		     tests/tests/comm/comm_limits.c $(auxSRCS)
tests_bin_queue_full_bin_SOURCES =		     tests/tests/comm/queue_full.c $(auxSRCS)
tests_bin_read_nsizes_bin_SOURCES =		     tests/tests/comm/read_nsizes.c $(auxSRCS)
tests_bin_read_smalls_bin_SOURCES =		     tests/tests/comm/read_smalls.c $(auxSRCS)
tests_bin_read_write_bin_SOURCES =		     tests/tests/comm/read_write.c $(auxSRCS)
tests_bin_small_writes_notify_bin_SOURCES =	     tests/tests/comm/small_writes_notify.c $(auxSRCS)
tests_bin_strings_bin_SOURCES =			     tests/tests/comm/strings.c $(auxSRCS)
tests_bin_write_bin_SOURCES =			     tests/tests/comm/write.c $(auxSRCS)
tests_bin_write_all_bin_SOURCES =		     tests/tests/comm/write_all.c $(auxSRCS)
tests_bin_write_all_mtt_bin_SOURCES =		     tests/tests/comm/write_all_mtt.c $(auxSRCS)
tests_bin_write_all_nsizes_bin_SOURCES =	     tests/tests/comm/write_all_nsizes.c $(auxSRCS)
tests_bin_write_all_nsizes_mtt_bin_SOURCES =	     tests/tests/comm/write_all_nsizes_mtt.c $(auxSRCS)
tests_bin_write_all_nsizes_nobuild_bin_SOURCES =     tests/tests/comm/write_all_nsizes_nobuild.c $(auxSRCS)
tests_bin_comm_write_m_to_1_bin_SOURCES =            tests/tests/comm/comm_write_m_to_1.c $(auxSRCS)
tests_bin_write_right_left_bin_SOURCES =	     tests/tests/comm/write_right_left.c $(auxSRCS)
tests_bin_write_simple_bin_SOURCES =		     tests/tests/comm/write_simple.c $(auxSRCS)
tests_bin_write_timeout_bin_SOURCES =		     tests/tests/comm/write_timeout.c $(auxSRCS)
tests_bin_z4k_pressure_bin_SOURCES =		     tests/tests/comm/z4k_pressure.c $(auxSRCS)
tests_bin_z4k_pressure_mtt_bin_SOURCES =	     tests/tests/comm/z4k_pressure_mtt.c $(auxSRCS)

## CONFIG
config_tests = tests/bin/set_config.bin \
	tests/bin/config_netinfo.bin \
	tests/bin/print_config.bin ##Not include before?

tests_bin_set_config_bin_SOURCES = tests/tests/config/set_config.c $(auxSRCS)
tests_bin_config_netinfo_bin_SOURCES = tests/tests/config/config_netinfo.c $(auxSRCS)
tests_bin_print_config_bin_SOURCES = tests/tests/config/print_config.c $(auxSRCS)

## ERROR_VEC
error_vec_tests = tests/bin/barrier_recover.bin \
	tests/bin/barrier_recover_ping.bin \
	tests/bin/get_error_vec.bin \
	tests/bin/one_dies.bin

tests_bin_barrier_recover_bin_SOURCES =               tests/tests/error_vec/barrier_recover.c $(auxSRCS)
tests_bin_barrier_recover_ping_bin_SOURCES =	      tests/tests/error_vec/barrier_recover_ping.c $(auxSRCS)
tests_bin_get_error_vec_bin_SOURCES =		      tests/tests/error_vec/get_error_vec.c $(auxSRCS)
tests_bin_one_dies_bin_SOURCES =		      tests/tests/error_vec/one_dies.c $(auxSRCS)

## EXTENSIONS
extensions_tests = tests/bin/ping_procs.bin \
     		   tests/bin/q_create.bin \
     		   tests/bin/seg_avail_local.bin

tests_bin_ping_procs_bin_SOURCES =            tests/tests/extensions/ping_procs.c $(auxSRCS)
tests_bin_q_create_bin_SOURCES =       	      tests/tests/extensions/q_create.c $(auxSRCS)
tests_bin_seg_avail_local_bin_SOURCES =	      tests/tests/extensions/seg_avail_local.c $(auxSRCS)

## FORTRAN
if WITH_FORTRAN
auxFCSRCS = src/GASPI_types.f90 src/GASPI.f90 src/GASPI_Ext.f90
fortran_tests = tests/bin/fortran_allreduces.bin \
      tests/bin/fortran_print.bin \
      tests/bin/fortran_seg_ptr.bin \
      tests/bin/fortran_seg_use.bin \
      tests/bin/fortran_use.bin

tests_bin_fortran_allreduces_bin_SOURCES =        tests/tests/fortran/fortran_allreduces.f90 $(auxFCSRCS)
tests_bin_fortran_print_bin_SOURCES =             tests/tests/fortran/fortran_print.f90 $(auxFCSRCS)
tests_bin_fortran_seg_ptr_bin_SOURCES =           tests/tests/fortran/fortran_seg_ptr.f90 $(auxFCSRCS)
tests_bin_fortran_seg_use_bin_SOURCES =           tests/tests/fortran/fortran_seg_use.f90 $(auxFCSRCS)
tests_bin_fortran_use_bin_SOURCES =               tests/tests/fortran/fortran_use.f90 $(auxFCSRCS)
endif

## GROUPS
groups_tests = tests/bin/g_before_start.bin \
      tests/bin/g_coll_del_coll.bin \
      tests/bin/g_commit_g_all.bin \
      tests/bin/g_commit_try_block.bin \
      tests/bin/g_elems.bin \
      tests/bin/g_force_timeout.bin \
      tests/bin/g_max_groups.bin \
      tests/bin/g_no_own.bin \
      tests/bin/g_num.bin \
      tests/bin/g_size.bin \
      tests/bin/g_some_from_all.bin

tests_bin_g_before_start_bin_SOURCES =                       tests/tests/groups/g_before_start.c $(auxSRCS)
tests_bin_g_coll_del_coll_bin_SOURCES =   	             tests/tests/groups/g_coll_del_coll.c $(auxSRCS)
tests_bin_g_commit_g_all_bin_SOURCES =    	             tests/tests/groups/g_commit_g_all.c $(auxSRCS)
tests_bin_g_commit_try_block_bin_SOURCES =	             tests/tests/groups/g_commit_try_block.c $(auxSRCS)
tests_bin_g_elems_bin_SOURCES =           	             tests/tests/groups/g_elems.c $(auxSRCS)
tests_bin_g_force_timeout_bin_SOURCES =   	             tests/tests/groups/g_force_timeout.c $(auxSRCS)
tests_bin_g_max_groups_bin_SOURCES =      	             tests/tests/groups/g_max_groups.c $(auxSRCS)
tests_bin_g_no_own_bin_SOURCES =          	             tests/tests/groups/g_no_own.c $(auxSRCS)
tests_bin_g_num_bin_SOURCES =             	             tests/tests/groups/g_num.c $(auxSRCS)
tests_bin_g_size_bin_SOURCES =            	             tests/tests/groups/g_size.c $(auxSRCS)
tests_bin_g_some_from_all_bin_SOURCES =   	             tests/tests/groups/g_some_from_all.c $(auxSRCS)

## INIT
init_tests = tests/bin/proc_init.bin \
	tests/bin/proc_init_timeout.bin \
	tests/bin/cmd_line_args.bin \
	tests/bin/kill_procs.bin \
	tests/bin/print_to.bin \
	tests/bin/null_ptrs.bin \
	tests/bin/numa_check.bin \
	tests/bin/strong_sym.bin \
	tests/bin/local_rank.bin \
	tests/bin/initialized.bin

tests_bin_proc_init_bin_SOURCES = tests/tests/init/proc_init.c $(auxSRCS)
tests_bin_proc_init_timeout_bin_SOURCES = tests/tests/init/proc_init_timeout.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_cmd_line_args_bin_SOURCES = tests/tests/init/cmd_line_args.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_kill_procs_bin_SOURCES = tests/tests/init/kill_procs.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_print_to_bin_SOURCES = tests/tests/init/print_to.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_null_ptrs_bin_SOURCES = tests/tests/init/null_ptrs.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_numa_check_bin_SOURCES = tests/tests/init/numa_check.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_strong_sym_bin_SOURCES = tests/tests/init/strong_sym.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_local_rank_bin_SOURCES = tests/tests/init/local_rank.c $(auxSRCS) $(lib_LTLIBRARIES)
tests_bin_initialized_bin_SOURCES = tests/tests/init/initialized.c $(auxSRCS) $(lib_LTLIBRARIES)

## LISTS
lists_tests = tests/bin/list_limit.bin \
      tests/bin/list_read.bin \
      tests/bin/list_read_notify.bin \
      tests/bin/list_write.bin \
      tests/bin/list_write_notify.bin

tests_bin_list_limit_bin_SOURCES =                     tests/tests/lists/list_limit.c $(auxSRCS)
tests_bin_list_read_bin_SOURCES =        	       tests/tests/lists/list_read.c $(auxSRCS)
tests_bin_list_read_notify_bin_SOURCES = 	       tests/tests/lists/list_read_notify.c $(auxSRCS)
tests_bin_list_write_bin_SOURCES =       	       tests/tests/lists/list_write.c $(auxSRCS)
tests_bin_list_write_notify_bin_SOURCES =	       tests/tests/lists/list_write_notify.c $(auxSRCS)

## MPI
if WITH_MPI

mpi_tests = tests/bin/mpi_allreduce.bin \
      tests/bin/mpi_barrier.bin \
      tests/bin/with_mpi.bin

tests_bin_mpi_allreduce_bin_SOURCES =        tests/tests/mpi/mpi_allreduce.c $(auxSRCS)
tests_bin_mpi_barrier_bin_SOURCES =          tests/tests/mpi/mpi_barrier.c $(auxSRCS)
tests_bin_with_mpi_bin_SOURCES =             tests/tests/mpi/with_mpi.c $(auxSRCS)

endif

## NOTIFICATIONS
notifications_tests = tests/bin/not_zero_wait.bin \
      tests/bin/notify_after_delete.bin \
      tests/bin/notify_all.bin \
      tests/bin/notify_last.bin \
      tests/bin/notify_null.bin \
      tests/bin/notify_seg_alloc.bin \
      tests/bin/read_notify.bin \
      tests/bin/write_m_to_1.bin \
      tests/bin/write_notify.bin \
      tests/bin/write_notify_outstanding.bin \
      tests/bin/notification_value.bin

tests_bin_not_zero_wait_bin_SOURCES =                   tests/tests/notifications/not_zero_wait.c $(auxSRCS)
tests_bin_notification_value_bin_SOURCES =    	        tests/tests/notifications/notification_value.cpp $(auxSRCS)
tests_bin_notify_after_delete_bin_SOURCES =             tests/tests/notifications/notify_after_delete.c $(auxSRCS)
tests_bin_notify_all_bin_SOURCES =                      tests/tests/notifications/notify_all.c $(auxSRCS)
tests_bin_notify_last_bin_SOURCES =                     tests/tests/notifications/notify_last.c $(auxSRCS)
tests_bin_notify_null_bin_SOURCES =                     tests/tests/notifications/notify_null.c $(auxSRCS)
tests_bin_notify_seg_alloc_bin_SOURCES =                tests/tests/notifications/notify_seg_alloc.c $(auxSRCS)
tests_bin_read_notify_bin_SOURCES =                     tests/tests/notifications/read_notify.c $(auxSRCS)
tests_bin_write_m_to_1_bin_SOURCES =                    tests/tests/notifications/write_m_to_1.c $(auxSRCS)
tests_bin_write_notify_bin_SOURCES =                    tests/tests/notifications/write_notify.c $(auxSRCS)
tests_bin_write_notify_outstanding_bin_SOURCES =        tests/tests/notifications/write_notify_outstanding.c $(auxSRCS)

## PASSIVE
passive_tests = tests/bin/passive.bin \
      tests/bin/passive_limits.bin \
      tests/bin/passive_n.bin \
      tests/bin/passive_nbytes.bin \
      tests/bin/passive_own.bin \
      tests/bin/passive_own_block.bin \
      tests/bin/passive_recvtimeout.bin \
      tests/bin/passive_tout_test.bin

tests_bin_passive_bin_SOURCES =                     tests/tests/passive/passive.c $(auxSRCS)
tests_bin_passive_limits_bin_SOURCES =              tests/tests/passive/passive_limits.c $(auxSRCS)
tests_bin_passive_n_bin_SOURCES =                   tests/tests/passive/passive_n.c $(auxSRCS)
tests_bin_passive_nbytes_bin_SOURCES =              tests/tests/passive/passive_nbytes.c $(auxSRCS)
tests_bin_passive_own_bin_SOURCES =                 tests/tests/passive/passive_own.c $(auxSRCS)
tests_bin_passive_own_block_bin_SOURCES =           tests/tests/passive/passive_own_block.c $(auxSRCS)
tests_bin_passive_recvtimeout_bin_SOURCES =         tests/tests/passive/passive_recvtimeout.c $(auxSRCS)
tests_bin_passive_tout_test_bin_SOURCES =           tests/tests/passive/passive_tout_test.c $(auxSRCS)

## SEGMENTS
segments_tests = tests/bin/max_mem.bin \
      tests/bin/seg_alloc_all.bin \
      tests/bin/seg_alloc_diff.bin \
      tests/bin/seg_avail_locally.bin \
      tests/bin/seg_create_all.bin \
      tests/bin/seg_create_same.bin \
      tests/bin/seg_id_list.bin \
      tests/bin/seg_reuse.bin \
      tests/bin/seg_reuse_buffer.bin \
      tests/bin/seg_subsegments.bin \
      tests/bin/seg_use_buffer.bin

#      tests/bin/seg_shmem.bin

tests_bin_max_mem_bin_SOURCES =                   tests/tests/segments/max_mem.c $(auxSRCS)
tests_bin_seg_alloc_all_bin_SOURCES =             tests/tests/segments/seg_alloc_all.c $(auxSRCS)
tests_bin_seg_alloc_diff_bin_SOURCES =            tests/tests/segments/seg_alloc_diff.c $(auxSRCS)
tests_bin_seg_avail_locally_bin_SOURCES =         tests/tests/segments/seg_avail_locally.c $(auxSRCS)
tests_bin_seg_create_all_bin_SOURCES =            tests/tests/segments/seg_create_all.c $(auxSRCS)
tests_bin_seg_create_same_bin_SOURCES =           tests/tests/segments/seg_create_same.c $(auxSRCS)
tests_bin_seg_id_list_bin_SOURCES =               tests/tests/segments/seg_id_list.c $(auxSRCS)
tests_bin_seg_reuse_bin_SOURCES =                 tests/tests/segments/seg_reuse.c $(auxSRCS)
tests_bin_seg_reuse_buffer_bin_SOURCES =          tests/tests/segments/seg_reuse_buffer.c $(auxSRCS)
#tests_bin_seg_shmem_bin_SOURCES =                tests/tests/segments/seg_shmem.cpp $(auxSRCS)
tests_bin_seg_subsegments_bin_SOURCES =           tests/tests/segments/seg_subsegments.c $(auxSRCS)
tests_bin_seg_use_buffer_bin_SOURCES =            tests/tests/segments/seg_use_buffer.c $(auxSRCS)

## THREADS
threads_tests = tests/bin/threads_init.bin \
		tests/bin/threads_sync_stress.bin

tests_bin_threads_init_bin_SOURCES =                tests/tests/threads/threads_init.c $(auxSRCS)
tests_bin_threads_sync_stress_bin_SOURCES =         tests/tests/threads/threads_sync_stress.c $(auxSRCS)

## UTILS
utils_tests = tests/bin/error_str.bin \
	    tests/bin/print_error.bin \
	    tests/bin/print_error2.bin \
	    tests/bin/time_get.bin

tests_bin_error_str_bin_SOURCES =            tests/tests/utils/error_str.c $(auxSRCS)
tests_bin_print_error_bin_SOURCES =          tests/tests/utils/print_error.c $(auxSRCS)
tests_bin_print_error2_bin_SOURCES =         tests/tests/utils/print_error2.c $(auxSRCS)
tests_bin_time_get_bin_SOURCES =             tests/tests/utils/time_get.c $(auxSRCS)

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
AM_FCFLAGS = -Isrc/gaspi/include
noinst_PROGRAMS += $(fortran_tests)
endif

if WITH_MPI
noinst_PROGRAMS += $(mpi_tests)
endif

# MICROBENCHMARKS
auxMICSRCS = tests/microbenchmarks/common.c
# LDADD = -lGPI2 -lpthread
# if WITH_INFINIBAND
# LDADD += -L@ac_lib_infiniband@ -libverbs
# endif
AM_CFLAGS = $(libGPI2_la_CFLAGS)
AM_CPPFLAGS = -Itests/microbenchmarks $(libGPI2_la_CPPFLAGS)

micro_tests = tests/microbenchmarks/allreduce.bin \
    tests/microbenchmarks/barrier.bin \
    tests/microbenchmarks/init_time.bin \
    tests/microbenchmarks/init_time_nobuild.bin \
    tests/microbenchmarks/nb_allreduce.bin \
    tests/microbenchmarks/nb_barrier.bin \
    tests/microbenchmarks/ping_pong.bin \
    tests/microbenchmarks/read_bw.bin \
    tests/microbenchmarks/read_notify_bw.bin \
    tests/microbenchmarks/write_bw.bin \
    tests/microbenchmarks/write_lat.bin \
    tests/microbenchmarks/write_notify_bw.bin \
    tests/microbenchmarks/write_notify_lat.bin

tests_microbenchmarks_allreduce_bin_SOURCES =              tests/microbenchmarks/allreduce.c          $(auxMICSRCS)
tests_microbenchmarks_barrier_bin_SOURCES =                tests/microbenchmarks/barrier.c            $(auxMICSRCS)
tests_microbenchmarks_init_time_bin_SOURCES =              tests/microbenchmarks/init_time.c          $(auxMICSRCS)
tests_microbenchmarks_init_time_nobuild_bin_SOURCES =      tests/microbenchmarks/init_time_nobuild.c  $(auxMICSRCS)
tests_microbenchmarks_nb_allreduce_bin_SOURCES =           tests/microbenchmarks/nb_allreduce.c       $(auxMICSRCS)
tests_microbenchmarks_nb_barrier_bin_SOURCES =             tests/microbenchmarks/nb_barrier.c         $(auxMICSRCS)
tests_microbenchmarks_ping_pong_bin_SOURCES =              tests/microbenchmarks/ping_pong.c          $(auxMICSRCS)
tests_microbenchmarks_read_bw_bin_SOURCES =                tests/microbenchmarks/read_bw.c            $(auxMICSRCS)
tests_microbenchmarks_read_notify_bw_bin_SOURCES =         tests/microbenchmarks/read_notify_bw.c     $(auxMICSRCS)
tests_microbenchmarks_write_bw_bin_SOURCES =               tests/microbenchmarks/write_bw.c           $(auxMICSRCS)
tests_microbenchmarks_write_lat_bin_SOURCES =              tests/microbenchmarks/write_lat.c          $(auxMICSRCS)
tests_microbenchmarks_write_notify_bw_bin_SOURCES =        tests/microbenchmarks/write_notify_bw.c    $(auxMICSRCS)
tests_microbenchmarks_write_notify_lat_bin_SOURCES =       tests/microbenchmarks/write_notify_lat.c   $(auxMICSRCS)

noinst_PROGRAMS += $(micro_tests)
