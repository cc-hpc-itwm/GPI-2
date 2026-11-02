#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nranks;
  ASSERT (gaspi_proc_num (&nranks));
  ASSERT (gaspi_proc_rank (&rank));

  const gaspi_segment_id_t seg0 = 0;
  const gaspi_segment_id_t seg1 = 1;

  ASSERT (gaspi_segment_create (seg0,
                                1024,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_MEM_UNINITIALIZED));
  ASSERT (gaspi_segment_create (seg1,
                                1024,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_MEM_UNINITIALIZED));

  gaspi_number_t notif_num;
  ASSERT (gaspi_notification_num (&notif_num));

  gaspi_number_t n_notifications = notif_num < 1024 ? notif_num : 1024;
  if (n_notifications == 0)
  {
    TSUITE_SKIP ("no notifications");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  gaspi_number_t queue_max;
  ASSERT (gaspi_queue_size_max (&queue_max));

  gaspi_rank_t const target_rank = (rank + 1) % nranks;

  for (gaspi_number_t n = 0; n < n_notifications; n++)
  {
    gaspi_number_t queue_size;
    ASSERT (gaspi_queue_size (0, &queue_size));
    if (queue_size > queue_max - 2)
    {
      ASSERT (gaspi_wait (0, GASPI_BLOCK));
    }

    ASSERT (gaspi_notify (seg0,
                          target_rank,
                          (gaspi_notification_id_t) n,
                          1,
                          0,
                          GASPI_BLOCK));
    ASSERT (gaspi_notify (seg1,
                          target_rank,
                          (gaspi_notification_id_t) n,
                          2,
                          0,
                          GASPI_BLOCK));
  }

  for (gaspi_number_t n = 0; n < n_notifications; n++)
  {
    gaspi_notification_id_t id;
    gaspi_notification_t notification_val;

    ASSERT (gaspi_notify_waitsome (seg0,
                                   0,
                                   n_notifications,
                                   &id,
                                   GASPI_BLOCK));
    ASSERT (gaspi_notify_reset (seg0, id, &notification_val));
    assert (notification_val == 1);

    ASSERT (gaspi_notify_waitsome (seg1,
                                   0,
                                   n_notifications,
                                   &id,
                                   GASPI_BLOCK));
    ASSERT (gaspi_notify_reset (seg1, id, &notification_val));
    assert (notification_val == 2);
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}