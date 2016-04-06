#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>

#include <GASPI_Ext.h>
int main(int argc, char *argv[])
{
  gaspi_rank_t rank, nprocs;
  gaspi_number_t queue_number;
  gaspi_notification_id_t id;
  
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  const gaspi_rank_t right = (rank + nprocs + 1) % nprocs;

  ASSERT(gaspi_queue_num(&queue_number));

  gaspi_queue_id_t new_queue;
  ASSERT ( gaspi_queue_create(&new_queue, GASPI_BLOCK) );
  assert( new_queue == (queue_number) );

  ASSERT(gaspi_queue_delete(new_queue));

  ASSERT ( gaspi_queue_create(&new_queue, GASPI_BLOCK) );
  assert( new_queue == (queue_number) );

  for ( queue_number = new_queue + 1; queue_number < GASPI_MAX_QP; queue_number++)
    {
      ASSERT ( gaspi_queue_create(&new_queue, GASPI_BLOCK) );
      assert( new_queue == queue_number);
    }
  EXPECT_FAIL( gaspi_queue_create(&new_queue, GASPI_BLOCK) );

  /* use queues */
  ASSERT ( gaspi_segment_create(0, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED) );

  for (new_queue = 0; new_queue < GASPI_MAX_QP; new_queue++)
    {
      ASSERT(gaspi_write(0, 0, right,
			 0, 0, 8,
			 new_queue, GASPI_BLOCK));

      ASSERT(gaspi_notify(0, right, 0, 1, new_queue, GASPI_BLOCK));
      ASSERT(gaspi_notify_waitsome(0, 0, 1, &id, GASPI_BLOCK));
      ASSERT(gaspi_wait(new_queue, GASPI_BLOCK));

      ASSERT(gaspi_read(0, 0, right,
			0, 0, 8,
			new_queue, GASPI_BLOCK));

      ASSERT(gaspi_wait(new_queue, GASPI_BLOCK));
    }

  ASSERT(gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
