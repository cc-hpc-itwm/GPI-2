#include <test_utils.h>

/* This tests checks the a user defined configuration of the maximum *
 number of allowed notifications:
 -) Zero notifications is allowed
 -) Set to the double of the default value (65536)
 -) Sending (waiting) for notification with id's larger than the
 above value leads to a GASPI_ERR_INV_NOTIF_ID error*/

int main ()
{
  gaspi_config_t default_conf;

  ASSERT (gaspi_config_get (&default_conf));

  gaspi_notification_id_t max_notification = default_conf.notification_num;

  gaspi_notification_id_t zero_notifications = 0;
  default_conf.notification_num = zero_notifications;
  ASSERT (gaspi_config_set(default_conf));

  gaspi_notification_id_t double_max_notification = 2 * max_notification;
  gaspi_notification_id_t triple_max_notification = 3 * max_notification;

  default_conf.notification_num = double_max_notification;
  ASSERT (gaspi_config_set (default_conf));

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t num, rank;
  ASSERT (gaspi_proc_num (&num));
  ASSERT (gaspi_proc_rank (&rank));

  ASSERT (gaspi_segment_create (0, sizeof(char), GASPI_GROUP_ALL,
                                GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  gaspi_number_t queue_size;
  gaspi_number_t queue_max;

  ASSERT (gaspi_queue_size_max (&queue_max));

  int res;
  gaspi_notification_t val;
  gaspi_notification_id_t my_id;


  if (rank == 0)
  {
    for (gaspi_notification_id_t i = 1; i <= triple_max_notification;++i)
    {
      res = gaspi_notify (0, 1, i, 1, 0, GASPI_BLOCK);
      if (i >= default_conf.notification_num)
      {
        assert (res == GASPI_ERR_INV_NOTIF_ID);
      }
      else
      {
        ASSERT (gaspi_queue_size (0, &queue_size));
        if (queue_size > queue_max - 1)
          ASSERT (gaspi_wait (0, GASPI_BLOCK));

        ASSERT (gaspi_notify (0, 1, i, 1, 0, GASPI_BLOCK));
      }
    }
  }
  else
  {
    for (gaspi_notification_id_t i=1; i<=triple_max_notification; ++i)
    {
      res = gaspi_notify_waitsome (0, i, 1, &my_id, GASPI_BLOCK);
      if (i >= default_conf.notification_num)
      {
        assert (res == GASPI_ERR_INV_NOTIF_ID);
      }
      else
      {
        ASSERT (res);
        ASSERT (gaspi_notify_reset (0, my_id, &val));
      }
    }
  }

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_segment_delete (0));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));
}
