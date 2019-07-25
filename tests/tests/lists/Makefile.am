include ../Make.inc

## LISTS
lists_tests = ../../bin/list_limit.bin \
      ../../bin/list_read.bin \
      ../../bin/list_read_notify.bin \
      ../../bin/list_write.bin \
      ../../bin/list_write_notify.bin

______bin_list_limit_bin_SOURCES = list_limit.c $(auxSRCS)
______bin_list_read_bin_SOURCES = list_read.c $(auxSRCS)
______bin_list_read_notify_bin_SOURCES = list_read_notify.c $(auxSRCS)
______bin_list_write_bin_SOURCES = list_write.c $(auxSRCS)
______bin_list_write_notify_bin_SOURCES = list_write_notify.c $(auxSRCS)

noinst_PROGRAMS = $(lists_tests)
