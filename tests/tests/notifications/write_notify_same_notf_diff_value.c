#include <test_utils.h>

int
main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t rank, nranks;
  ASSERT (gaspi_proc_num (&nranks));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_number_t notif_num;
  ASSERT (gaspi_notification_num (&notif_num));

  gaspi_number_t n_notifications = notif_num < 1024 ? notif_num : 1024;
  if (n_notifications == 0)
  {
    TSUITE_SKIP ("no notifications");
    ASSERT (gaspi_proc_term (GASPI_BLOCK));
    return EXIT_SUCCESS;
  }

  const gaspi_size_t total_transfer_size =
    n_notifications * sizeof (gaspi_rank_t);
  const gaspi_size_t seg_size = 2 * total_transfer_size;
  const gaspi_segment_id_t seg_id = 0;
  const gaspi_rank_t producer_rank = nranks > 1 ? 1 : 0;

  ASSERT (gaspi_segment_create (seg_id,
                                seg_size,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_MEM_UNINITIALIZED));

  gaspi_pointer_t seg_ptr_void;
  ASSERT (gaspi_segment_ptr (seg_id, &seg_ptr_void));

  gaspi_rank_t *const mem = (gaspi_rank_t *) seg_ptr_void;

  for (gaspi_size_t i = 0; i < seg_size / sizeof (gaspi_rank_t); i++)
  {
    mem[i] = rank;
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  gaspi_number_t queue_size;
  gaspi_number_t queue_max;
  ASSERT (gaspi_queue_size_max (&queue_max));

  gaspi_offset_t local_offset = 0;
  gaspi_offset_t remote_offset = total_transfer_size;

  if (rank == producer_rank)
  {
    for (gaspi_number_t n = 0; n < n_notifications; n++)
    {
      for (gaspi_rank_t target = 0; target < nranks; target++)
      {
        ASSERT (gaspi_queue_size (0, &queue_size));
        if (queue_size > queue_max - 1)
        {
          ASSERT (gaspi_wait (0, GASPI_BLOCK));
        }

        ASSERT (gaspi_write_notify (seg_id, local_offset, target,
                                    seg_id, remote_offset,
                                    sizeof (gaspi_rank_t),
                                    (gaspi_notification_id_t) n,
                                    target + 1,
                                    0,
                                    GASPI_BLOCK));
      }

      local_offset += sizeof (gaspi_rank_t);
      remote_offset += sizeof (gaspi_rank_t);
    }
  }

  for (gaspi_number_t n = 0; n < n_notifications; n++)
  {
    gaspi_notification_id_t id;
    gaspi_notification_t notification_val;

    ASSERT (gaspi_notify_waitsome (seg_id,
                                   0,
                                   n_notifications,
                                   &id,
                                   GASPI_BLOCK));
    ASSERT (gaspi_notify_reset (seg_id, id, &notification_val));

    assert (notification_val == rank + 1);
  }

  for (gaspi_size_t i = 0; i < total_transfer_size / sizeof (gaspi_rank_t); i++)
  {
    assert (mem[i] == rank);
  }

  for (gaspi_size_t i = total_transfer_size / sizeof (gaspi_rank_t);
       i < 2 * total_transfer_size / sizeof (gaspi_rank_t);
       i++)
  {
    assert (mem[i] == producer_rank);
  }

  ASSERT (gaspi_wait (0, GASPI_BLOCK));
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}