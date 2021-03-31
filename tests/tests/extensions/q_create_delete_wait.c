#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

#include <GASPI_Ext.h>

/* Test creation, deletion and waiting on queues */

int
main (int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_number_t queue_number, queue_counter, q_max;
  gaspi_notification_id_t id;

  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  gaspi_queue_id_t new_queue;

  ASSERT (gaspi_queue_max (&q_max));
  ASSERT (gaspi_queue_num (&queue_number));

  /* Create maximum number of queues */
  for (queue_counter = queue_number; queue_counter < q_max; queue_counter++)
  {
    ASSERT (gaspi_queue_create (&new_queue, GASPI_BLOCK));
    assert (new_queue == queue_counter);

    ASSERT (gaspi_queue_num (&queue_number));
    assert (queue_number == queue_counter + 1);
  }

  /* Can not go above the limit */
  EXPECT_FAIL (gaspi_queue_create (&new_queue, GASPI_BLOCK));

  /* Delete the above created queues */
  for (id = queue_number - 1; id > 0; id--)
  {
    ASSERT (gaspi_queue_delete (id));

    ASSERT (gaspi_queue_num (&queue_number));

    /* Wait on a deleted queue must fail */
    EXPECT_FAIL (gaspi_wait(id, GASPI_BLOCK));

    assert (queue_number == id);
  }

  /* Must be have at least a queue */
  EXPECT_FAIL (gaspi_queue_delete (id - 1));

  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  return EXIT_SUCCESS;
}
