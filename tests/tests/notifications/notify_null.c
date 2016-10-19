#include <test_utils.h>

/* Test the usage of a NULL pointer for the notification value */
int
main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);
  
  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  
  gaspi_notification_id_t notif = 0;
  gaspi_number_t notif_num,n;
  gaspi_rank_t rank, nprocs, i;
  const  gaspi_segment_id_t seg_id = 0;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));
  
  ASSERT (gaspi_segment_create(seg_id, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  
  ASSERT( gaspi_notification_num(&notif_num));
  
  if(rank == 0)
    {
      gaspi_number_t queue_size;
      gaspi_number_t queue_max;
      ASSERT (gaspi_queue_size_max(&queue_max));

      for(n = 0; n < notif_num; n++)
	{
	  for(i = 1; i < nprocs; i++)
	    {
	      ASSERT (gaspi_queue_size(0, &queue_size));
	      if(queue_size > queue_max - 1)
		ASSERT (gaspi_wait(0, GASPI_BLOCK));

	      notif = (gaspi_notification_id_t) n;
	      ASSERT (gaspi_notify( seg_id, i, notif, 1, 0, GASPI_BLOCK));
	    }
	}
    }
  else
    {
      do
	{
	  gaspi_notification_id_t id;
	  ASSERT (gaspi_notify_waitsome(seg_id, 0, notif_num, &id, GASPI_BLOCK));

	  ASSERT( gaspi_notify_reset(seg_id, id, NULL));

	  n++;
	}
      while(n < notif_num);
    }
  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
  
  return EXIT_SUCCESS;
}
