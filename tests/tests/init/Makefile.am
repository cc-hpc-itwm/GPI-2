include ../Make.inc

## INIT
init_tests = ../../bin/proc_init.bin \
	../../bin/proc_init_timeout.bin \
	../../bin/cmd_line_args.bin \
	../../bin/kill_procs.bin \
	../../bin/print_to.bin \
	../../bin/null_ptrs.bin \
	../../bin/numa_check.bin \
	../../bin/strong_sym.bin \
	../../bin/local_rank.bin \
	../../bin/initialized.bin

______bin_proc_init_bin_SOURCES = proc_init.c $(auxSRCS)
______bin_proc_init_timeout_bin_SOURCES = proc_init_timeout.c $(auxSRCS)
______bin_cmd_line_args_bin_SOURCES = cmd_line_args.c $(auxSRCS)
______bin_kill_procs_bin_SOURCES = kill_procs.c $(auxSRCS)
______bin_print_to_bin_SOURCES = print_to.c $(auxSRCS)
______bin_null_ptrs_bin_SOURCES = null_ptrs.c $(auxSRCS)
______bin_numa_check_bin_SOURCES = numa_check.c $(auxSRCS)
______bin_strong_sym_bin_SOURCES = strong_sym.c $(auxSRCS)
______bin_local_rank_bin_SOURCES = local_rank.c $(auxSRCS)
______bin_initialized_bin_SOURCES = initialized.c $(auxSRCS)

noinst_PROGRAMS = $(init_tests)
