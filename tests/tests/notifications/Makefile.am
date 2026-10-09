include ../Make.inc

## NOTIFICATIONS
notifications_tests = ../../bin/not_zero_wait.bin \
      ../../bin/notify_after_delete.bin \
      ../../bin/notify_all.bin \
      ../../bin/notify_different_segs.bin \
      ../../bin/notify_last.bin \
      ../../bin/notify_many_senders.bin \
      ../../bin/notify_null.bin \
      ../../bin/notify_waitsome_mt.bin \
      ../../bin/notify_waitsome_timeout.bin \
      ../../bin/read_notify.bin \
      ../../bin/write_m_to_1.bin \
      ../../bin/write_notify.bin \
      ../../bin/write_notify_outstanding.bin \
      ../../bin/write_notify_same_notf_diff_value.bin \
      ../../bin/write_notify_zero_size.bin \
      ../../bin/read_notify_zero_size.bin \
      ../../bin/max_notifications_config.bin \
      ../../bin/write_notify_ordering.bin \
      ../../bin/write_notify_visibility.bin \
      ../../bin/read_notify_visibility.bin
#      ../../bin/notification_value.bin

______bin_not_zero_wait_bin_SOURCES = not_zero_wait.c $(auxSRCS)
______bin_notify_after_delete_bin_SOURCES = notify_after_delete.c $(auxSRCS)
______bin_notify_all_bin_SOURCES = notify_all.c $(auxSRCS)
______bin_notify_different_segs_bin_SOURCES = notify_different_segs.c $(auxSRCS)
______bin_notify_last_bin_SOURCES = notify_last.c $(auxSRCS)
______bin_notify_many_senders_bin_SOURCES = notify_many_senders.c $(auxSRCS)
______bin_notify_null_bin_SOURCES = notify_null.c $(auxSRCS)
______bin_notify_waitsome_mt_bin_SOURCES = notify_waitsome_mt.c $(auxSRCS)
______bin_notify_waitsome_timeout_bin_SOURCES = notify_waitsome_timeout.c $(auxSRCS)
______bin_read_notify_bin_SOURCES = read_notify.c $(auxSRCS)
______bin_write_m_to_1_bin_SOURCES = write_m_to_1.c $(auxSRCS)
______bin_write_notify_bin_SOURCES = write_notify.c $(auxSRCS)
______bin_write_notify_outstanding_bin_SOURCES = write_notify_outstanding.c $(auxSRCS)
______bin_write_notify_same_notf_diff_value_bin_SOURCES = write_notify_same_notf_diff_value.c $(auxSRCS)
______bin_write_notify_zero_size_bin_SOURCES = write_notify_zero_size.c $(auxSRCS)
______bin_read_notify_zero_size_bin_SOURCES = read_notify_zero_size.c $(auxSRCS)
______bin_max_notifications_config_bin_SOURCES = max_notifications_config.c $(auxSRCS)
______bin_write_notify_ordering_bin_SOURCES = write_notify_ordering.c $(auxSRCS)
______bin_write_notify_visibility_bin_SOURCES = write_notify_visibility.c $(auxSRCS)
______bin_read_notify_visibility_bin_SOURCES = read_notify_visibility.c $(auxSRCS)

#______bin_notification_value_bin_SOURCES = notification_value.cpp $(auxSRCS)
#______bin_notification_value_bin_LDFLAGS = -std=c++11

test_pkglibexec_PROGRAMS = $(notifications_tests)
