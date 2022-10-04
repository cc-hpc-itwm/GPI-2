#include <pthread.h>
#include <test_utils.h>

typedef struct 
{
  gaspi_segment_id_t segment_id;
  int *p_recv;
} thread_data_t;

gaspi_rank_t right_neighbor_rank() 
{
  gaspi_rank_t my_proc;
  ASSERT (gaspi_proc_rank (&my_proc));
  
  gaspi_rank_t nproc;
  ASSERT (gaspi_proc_num (&nproc));

  return ((my_proc + nproc + 1) % nproc);
}

gaspi_return_t wait_for_notification (gaspi_rank_t rank_id,
                                      gaspi_segment_id_t segment_id,
                                      gaspi_notification_id_t notification_id)
{
  gaspi_notification_id_t first_id;
  
  if (gaspi_notify_waitsome (segment_id,
                             notification_id,
                             1,
                             &first_id,
                             20000 ) == GASPI_TIMEOUT) 
  {
    return GASPI_TIMEOUT;
  }
  
  if (notification_id != first_id)
  {
    return GASPI_ERROR;
  }
  
  gaspi_notification_t old_notification_val;
  ASSERT (gaspi_notify_reset (segment_id,
                              notification_id,
                              &old_notification_val));
  return GASPI_SUCCESS;
}

gaspi_offset_t pointer_to_offset (void const * const pointer,
                                  gaspi_segment_id_t segment_id)
{
  gaspi_pointer_t segment_ptr = NULL;
  ASSERT (gaspi_segment_ptr (segment_id, &segment_ptr));

  gaspi_offset_t offset;
  offset = (gaspi_offset_t) (pointer) - (gaspi_offset_t) (segment_ptr);

  assert (offset >= 0);

  return offset;
}

static void *passive_recv (void *tdata)
{
  int FINISHED = 0;
  gaspi_return_t ret;
  thread_data_t *info = (thread_data_t *) tdata;
  gaspi_segment_id_t segment_id = info->segment_id;
  int *p_recv = info->p_recv;

  while (!FINISHED)
  {
    gaspi_rank_t  sender_rank = 0;

    ASSERT (gaspi_passive_receive (segment_id,
                                  pointer_to_offset (p_recv, segment_id),
                                  &sender_rank,
                                  sizeof (int),
                                  GASPI_BLOCK ));

    if (*p_recv == __INT_MAX__) 
    {
      FINISHED = 1;
    }
    else 
    {
      while (GASPI_QUEUE_FULL == (ret = gaspi_notify (segment_id,
                                                      sender_rank,
                                                      *p_recv,
                                                      1,
                                                      0,
                                                      GASPI_BLOCK))) 
      {
        ASSERT (gaspi_wait (0, GASPI_BLOCK));
      }
      assert (ret == GASPI_SUCCESS);
    }
  }
  pthread_exit (NULL);
}

int main (int argc, char *argv[])
{
  TSUITE_INIT (argc, argv);

  ASSERT (gaspi_proc_init (GASPI_BLOCK));

  gaspi_rank_t my_proc;
  ASSERT (gaspi_proc_rank (&my_proc));

  gaspi_segment_id_t segment_id = 0;
  gaspi_size_t segment_size = 1024;
  ASSERT (gaspi_segment_create (segment_id,
                                segment_size,
                                GASPI_GROUP_ALL,
                                GASPI_BLOCK,
                                GASPI_MEM_INITIALIZED));

  gaspi_pointer_t ptr = NULL;
  int *p_send = NULL, *p_recv = NULL;
  ASSERT (gaspi_segment_ptr (segment_id, &ptr));
  p_send = (int *) ptr;
  p_recv = p_send + 1;

  thread_data_t tdata;
  pthread_t tid;
  tdata.segment_id = segment_id;
  tdata.p_recv = p_recv;
  ASSERT (pthread_create (&tid, NULL, &passive_recv, (void *) &tdata));

  int num_send = 30, iter = 10;
  gaspi_rank_t rank = right_neighbor_rank();
  gaspi_return_t ret = GASPI_SUCCESS;
  for (int i = 0; i < iter; i++)
  {
    for (int j = 0; j < num_send; j++)
    {
      *p_send = i * num_send + j;
      
      ASSERT (gaspi_passive_send (segment_id,
                                  pointer_to_offset (p_send, segment_id),
                                  rank,
                                  sizeof (int),
                                  GASPI_BLOCK));
    }
    for (int j = 0; j < num_send; j++) 
    {
      if (GASPI_SUCCESS != (ret = wait_for_notification (rank,
                                                         segment_id,
                                                         i * num_send + j)))
      {
        goto clean_exit;
      }
    }
  }

ASSERT (gaspi_barrier (GASPI_GROUP_ALL,GASPI_BLOCK));

clean_exit:
  *p_send = __INT_MAX__;
  ASSERT (gaspi_passive_send (segment_id,
                             pointer_to_offset (p_send, segment_id),
                             my_proc,
                             sizeof (int),
                             GASPI_BLOCK ));
  ASSERT (pthread_join (tid, NULL));
  ASSERT (gaspi_segment_delete (segment_id));
  ASSERT (gaspi_barrier (GASPI_GROUP_ALL, GASPI_BLOCK));
  ASSERT (gaspi_proc_term (GASPI_BLOCK));

  if (ret != GASPI_SUCCESS)
  {
    return (EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}
