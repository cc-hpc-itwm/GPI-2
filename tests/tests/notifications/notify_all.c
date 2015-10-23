#include <stdio.h>
#include <stdlib.h>

#include <test_utils.h>


int main(int argc, char *argv[])
{
  TSUITE_INIT(argc, argv);
  
  ASSERT (gaspi_proc_init(GASPI_BLOCK));
  
  gaspi_notification_id_t n=0;
  gaspi_number_t notif_num;
  gaspi_rank_t rank, nprocs, i;
  const  gaspi_segment_id_t seg_id = 0;

  ASSERT(gaspi_proc_num(&nprocs));
  ASSERT (gaspi_proc_rank(&rank));
  
  ASSERT (gaspi_segment_create(seg_id, 1024, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED));
  
  ASSERT( gaspi_notification_num(&notif_num));
  gaspi_printf("max num notifications %u\n", notif_num);
  
  gaspi_number_t queue_size;
  gaspi_number_t queue_max;
  ASSERT (gaspi_queue_size_max(&queue_max));
  
  for(n = 0; n < notif_num; n++)
    {      
      for(i = 0; i < nprocs; i++)
	{
	  ASSERT (gaspi_queue_size(0, &queue_size));
	  if(queue_size > queue_max - 1)
	    ASSERT (gaspi_wait(0, GASPI_BLOCK));
	  
	  ASSERT (gaspi_notify( seg_id, i, n, i+1, 0, GASPI_BLOCK));
	}
    }

  n=0; //re-start counter

  do
    {
      gaspi_notification_id_t id;
      ASSERT (gaspi_notify_waitsome(seg_id, 0, notif_num, &id, GASPI_BLOCK));
      
      gaspi_notification_t notification_val;
      ASSERT( gaspi_notify_reset(seg_id, id, &notification_val));
      assert(notification_val == rank + 1);
      n++;
    }
  while(n < notif_num);
  ASSERT(gaspi_wait(0, GASPI_BLOCK));
  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term(GASPI_BLOCK));
  
  return EXIT_SUCCESS;
}
