include ../Make.inc

## COMM
comm_tests = \
	../../bin/all-to-all.bin \
	../../bin/all-to-rank0.bin \
	../../bin/comm_limits.bin \
	../../bin/queue_full.bin \
	../../bin/read_nsizes.bin \
	../../bin/read_smalls.bin \
	../../bin/read_write.bin \
	../../bin/small_writes_notify.bin \
	../../bin/strings.bin \
	../../bin/write.bin \
	../../bin/write_all.bin \
	../../bin/write_all_mtt.bin \
	../../bin/write_all_nsizes.bin \
	../../bin/write_all_nsizes_mtt.bin \
	../../bin/write_all_nsizes_nobuild.bin \
	../../bin/comm_write_m_to_1.bin \
	../../bin/write_right_left.bin \
	../../bin/write_simple.bin \
	../../bin/write_timeout.bin \
	../../bin/z4k_pressure.bin \
	../../bin/z4k_pressure_mtt.bin

______bin_all_to_all_bin_SOURCES = all-to-all.c $(auxSRCS)
______bin_all_to_rank0_bin_SOURCES = all-to-rank0.c $(auxSRCS)
______bin_comm_limits_bin_SOURCES = comm_limits.c $(auxSRCS)
______bin_queue_full_bin_SOURCES = queue_full.c $(auxSRCS)
______bin_read_nsizes_bin_SOURCES = read_nsizes.c $(auxSRCS)
______bin_read_smalls_bin_SOURCES = read_smalls.c $(auxSRCS)
______bin_read_write_bin_SOURCES = read_write.c $(auxSRCS)
______bin_small_writes_notify_bin_SOURCES = small_writes_notify.c $(auxSRCS)
______bin_strings_bin_SOURCES = strings.c $(auxSRCS)
______bin_write_bin_SOURCES = write.c $(auxSRCS)
______bin_write_all_bin_SOURCES = write_all.c $(auxSRCS)
______bin_write_all_mtt_bin_SOURCES = write_all_mtt.c $(auxSRCS)
______bin_write_all_nsizes_bin_SOURCES = write_all_nsizes.c $(auxSRCS)
______bin_write_all_nsizes_mtt_bin_SOURCES = write_all_nsizes_mtt.c $(auxSRCS)
______bin_write_all_nsizes_nobuild_bin_SOURCES = write_all_nsizes_nobuild.c $(auxSRCS)
______bin_comm_write_m_to_1_bin_SOURCES = comm_write_m_to_1.c $(auxSRCS)
______bin_write_right_left_bin_SOURCES = write_right_left.c $(auxSRCS)
______bin_write_simple_bin_SOURCES = write_simple.c $(auxSRCS)
______bin_write_timeout_bin_SOURCES = write_timeout.c $(auxSRCS)
______bin_z4k_pressure_bin_SOURCES = z4k_pressure.c $(auxSRCS)
______bin_z4k_pressure_mtt_bin_SOURCES = z4k_pressure_mtt.c $(auxSRCS)

noinst_PROGRAMS = $(comm_tests)
