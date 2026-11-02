#include <stdlib.h>

#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nranks;
  ASSERT (gaspi_proc_num (&nranks));
  ASSERT (gaspi_proc_rank (&rank));

  ASSERT (gaspi_segment_create (0,
                                sizeof (char),
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_MEM_UNINITIALIZED));

  if (nranks < 3)
  {
    TSUITE_SKIP ("needs 3 ranks");
    ASSERT (gaspi_segment_delete (0));
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  gaspi_number_t notif_num;
  gaspi_number_t queue_max;
  ASSERT (gaspi_notification_num (&notif_num));
  ASSERT (gaspi_queue_size_max (&queue_max));

  gaspi_number_t notif_per_sender = notif_num / (nranks - 1);
  gaspi_number_t queue_budget = queue_max > 1 ? queue_max / 2 : 1;
  if (notif_per_sender > queue_budget)
  {
    notif_per_sender = queue_budget;
  }

  if (notif_per_sender == 0)
  {
    TSUITE_SKIP ("too few notifications per sender");
    ASSERT (gaspi_segment_delete (0));
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  gaspi_number_t total_notifications = notif_per_sender * (nranks - 1);

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  if (rank != 0)
  {
    for (gaspi_number_t n = 0; n < notif_per_sender; n++)
    {
      gaspi_number_t queue_size;
      ASSERT (gaspi_queue_size (0, &queue_size));
      if (queue_size > queue_max - 1)
      {
        ASSERT (gaspi_wait (0, GASPI_BLOCK));
      }

      ASSERT (gaspi_notify (0,
                            0,
                            (gaspi_notification_id_t) ((rank - 1) * notif_per_sender + n),
                            rank,
                            0,
                            GASPI_BLOCK));
    }

    ASSERT (gaspi_wait (0, GASPI_BLOCK));
  }
  else
  {
    gaspi_number_t *const counts =
      (gaspi_number_t *) calloc (nranks, sizeof (gaspi_number_t));
    assert (counts != NULL);

    for (gaspi_number_t n = 0; n < total_notifications; n++)
    {
      gaspi_notification_id_t id;
      gaspi_notification_t notification_val;

      ASSERT (gaspi_notify_waitsome (0,
                                     0,
                                     total_notifications,
                                     &id,
                                     GASPI_BLOCK));
      ASSERT (gaspi_notify_reset (0, id, &notification_val));

      assert (notification_val > 0);
      assert (notification_val < nranks);
      counts[notification_val]++;
    }

    for (gaspi_rank_t sender = 1; sender < nranks; sender++)
    {
      assert (counts[sender] == notif_per_sender);
    }

    free (counts);
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_segment_delete (0));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}