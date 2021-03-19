include ../Make.inc

## CONFIG
config_tests = ../../bin/set_config.bin \
	../../bin/config_netinfo.bin ../../bin/config_usernet.bin

______bin_set_config_bin_SOURCES = set_config.c $(auxSRCS)
______bin_config_netinfo_bin_SOURCES = config_netinfo.c $(auxSRCS)
______bin_config_usernet_bin_SOURCES = config_usernet.c $(auxSRCS)

test_pkglibexec_PROGRAMS = $(config_tests)
