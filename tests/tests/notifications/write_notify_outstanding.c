#include <test_utils.h>

/* Test to exhaust notifications on all ranks while keep the
   communication outstanding as long it fits the queue. */

int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);

  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  gaspi_rank_t rank, nprocs, i;
  const  gaspi_segment_id_t seg_id = 0;
  const gaspi_offset_t offset = 0;
  const gaspi_size_t transfer_size = 8192;

  gaspi_number_t queue_size;
  gaspi_number_t queue_max;
  ASSERT (gaspi_queue_size_max(&queue_max));

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));

  if( nprocs < 2 )
    {
      return EXIT_SUCCESS;
    }

  ASSERT (gaspi_segment_create(seg_id, nprocs * 2 * transfer_size, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));

  gaspi_number_t max_notifications;
  ASSERT(gaspi_notification_num(&max_notifications));

  gaspi_number_t avail_notifications = max_notifications / nprocs;

  max_notifications = avail_notifications * nprocs;

  gaspi_pointer_t _vptr;
  ASSERT (gaspi_segment_ptr(0, &_vptr));

  int *mem = (int *) _vptr;

  for(i = 0; i < nprocs; i++)
    {
      mem[i] = (int) rank;
    }

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));

  for(i = 0; i < nprocs; i++)
    {
      gaspi_notification_id_t not;
      for(not = 0; not < avail_notifications; not++)
	{
	  ASSERT (gaspi_queue_size(0, &queue_size));
	  if( queue_size > queue_max - 1 )
	    {
	      ASSERT (gaspi_wait(0, GASPI_BLOCK));
	    }

	  gaspi_notification_id_t the_notification = (gaspi_notification_id_t) (rank * avail_notifications + not);

	  ASSERT( gaspi_write_notify( seg_id, offset, i,
				      seg_id, offset, transfer_size,
				      the_notification, 1,
				      0, GASPI_BLOCK));
	}
    }

  gaspi_notification_id_t n = 0;

  do
    {
      gaspi_notification_id_t id;
      ASSERT (gaspi_notify_waitsome(seg_id, 0, max_notifications - 1 , &id, GASPI_BLOCK));

      gaspi_notification_t notification_val;
      ASSERT( gaspi_notify_reset(seg_id, id, &notification_val));

      assert(notification_val == 1);
      n++;
    }
  while(n < max_notifications - 1);

  ASSERT(gaspi_wait(0, GASPI_BLOCK));

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
