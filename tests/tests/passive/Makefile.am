include ../Make.inc

## PASSIVE
passive_tests = ../../bin/passive.bin \
      ../../bin/passive_limits.bin \
      ../../bin/passive_limits_config.bin \
      ../../bin/passive_n.bin \
      ../../bin/passive_nbytes.bin \
      ../../bin/passive_own.bin \
      ../../bin/passive_own_block.bin \
      ../../bin/passive_recvtimeout.bin \
      ../../bin/passive_tout_test.bin   \
      ../../bin/passive_queue_max_config.bin


______bin_passive_bin_SOURCES = passive.c $(auxSRCS)
______bin_passive_limits_bin_SOURCES = passive_limits.c $(auxSRCS)
______bin_passive_limits_config_bin_SOURCES = passive_limits_config.c $(auxSRCS)
______bin_passive_n_bin_SOURCES = passive_n.c $(auxSRCS)
______bin_passive_nbytes_bin_SOURCES = passive_nbytes.c $(auxSRCS)
______bin_passive_own_bin_SOURCES = passive_own.c $(auxSRCS)
______bin_passive_own_block_bin_SOURCES = passive_own_block.c $(auxSRCS)
______bin_passive_recvtimeout_bin_SOURCES = passive_recvtimeout.c $(auxSRCS)
______bin_passive_tout_test_bin_SOURCES = passive_tout_test.c $(auxSRCS)
______bin_passive_queue_max_config_bin_SOURCES= passive_queue_max_config.c $(auxSRCS)

noinst_PROGRAMS = $(passive_tests)
