include ../Make.inc

## CONFIG
config_tests = ../../bin/set_config.bin \
	../../bin/config_netinfo.bin \
	../../bin/print_config.bin ##Not include before?

______bin_set_config_bin_SOURCES = set_config.c $(auxSRCS)
______bin_config_netinfo_bin_SOURCES = config_netinfo.c $(auxSRCS)
______bin_print_config_bin_SOURCES = print_config.c $(auxSRCS)

noinst_PROGRAMS = $(config_tests)
