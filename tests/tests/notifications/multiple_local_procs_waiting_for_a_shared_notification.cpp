#include <GASPI.h>
#include <GASPI_Ext.h>

#include <assert.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <stdlib.h>

#include "allocate_shared_memory_for_segments.h"

#define SUCCESS_OR_DIE(f, args...)              \
  do                                            \
  {                                             \
    gaspi_return_t const r = f (args);          \
                                                \
    if (r != GASPI_SUCCESS)                     \
    {                                           \
      gaspi_printf ( "Error[%s:%i]: %s\n"       \
                   , __FILE__                   \
                   , __LINE__                   \
                   , gaspi_error_str (r)        \
                   );                           \
                                                \
      exit (-1);                      \
    }                                           \
  } while (0)


static void wait_for_queue_entries
  ( gaspi_queue_id_t* queue
  , gaspi_number_t n_requested_entries
  )
{
  gaspi_number_t queue_size_max;
  gaspi_number_t queue_size;
  gaspi_number_t queue_num;

  SUCCESS_OR_DIE (gaspi_queue_size_max, &queue_size_max);
  SUCCESS_OR_DIE (gaspi_queue_size, *queue, &queue_size);
  SUCCESS_OR_DIE (gaspi_queue_num, &queue_num);

  if (! (queue_size + n_requested_entries <= queue_size_max))
  {
    *queue = (*queue + 1) % queue_num;
    SUCCESS_OR_DIE (gaspi_wait, *queue, GASPI_BLOCK);
  }
}

int main(int argc, char* argv[])
{
  gaspi_rank_t gpi_rank, n_gaspi_procs;
  SUCCESS_OR_DIE (gaspi_proc_init, GASPI_BLOCK);
  SUCCESS_OR_DIE (gaspi_proc_rank, &gpi_rank);
  SUCCESS_OR_DIE (gaspi_proc_num, &n_gaspi_procs);

  gaspi_rank_t gpi_local_rank;
  SUCCESS_OR_DIE (gaspi_proc_local_rank, &gpi_local_rank);

  gaspi_rank_t n_local_procs;
  SUCCESS_OR_DIE (gaspi_proc_local_num, &n_local_procs);

  int* ptr_to_shared_data;
  int shm_fd_data = -1;
  unsigned long shm_data_size = n_local_procs * sizeof(int);

  SUCCESS_OR_DIE (get_ptr_to_shared_data, (void**)&ptr_to_shared_data, shm_data_size,  &shm_fd_data, "/shared_mem_for_data");

  unsigned int *ptr_cnt_notified_procs;
  int shm_fd_cnt = -1;
  SUCCESS_OR_DIE (get_ptr_to_shared_data, (void**)&ptr_cnt_notified_procs, shm_data_size,  &shm_fd_cnt, "/shared_mem_for_counter");
  *ptr_cnt_notified_procs = 0;

  gaspi_segment_id_t segment_id = 0;
  SUCCESS_OR_DIE ( gaspi_segment_use
                 , segment_id
                 , ptr_to_shared_data
                 , n_local_procs * sizeof (int)
                 , GASPI_GROUP_ALL
                 , GASPI_BLOCK
                 , GASPI_NODE_LOCAL
                 );

  gaspi_pointer_t gaspi_ptr;
  SUCCESS_OR_DIE (gaspi_segment_ptr, segment_id, &gaspi_ptr);

  ptr_to_shared_data[gpi_local_rank] = gpi_rank;
  SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_queue_id_t queue = 0;
  wait_for_queue_entries (&queue, n_gaspi_procs);

  gaspi_notification_id_t notification_id (1000);

  sem_t* mutex_cnt_notified_procs;
  const char SEM_NAME[] = "semaphore_cnt";
  mutex_cnt_notified_procs = sem_open (SEM_NAME, O_CREAT, 0644, 1);
  if (mutex_cnt_notified_procs == SEM_FAILED)
  {
    perror ("unable to create semaphore");
    sem_unlink (SEM_NAME);
    exit(-1);
  }

  // the local process 0 on node0 sends the content of the buffer shared between the local processes
  if (gpi_rank == 0)
  {
    SUCCESS_OR_DIE
      ( gaspi_write_notify
      , segment_id
      , 0
      , n_local_procs // local process 0 on remote node 1
      , segment_id
      , 0
      , n_local_procs * sizeof (int)
      , notification_id
      , (gaspi_notification_t) (gpi_rank + 1)
      , queue
      , GASPI_BLOCK
      );
  }

  gaspi_notification_id_t received_notification;

  // any process on remote node 1 should see the notification
  if (gpi_rank >= n_local_procs && gpi_rank < 2 * n_local_procs - 1)
  {
    SUCCESS_OR_DIE
      ( gaspi_notify_waitsome
      , segment_id
      , notification_id
      , 1
      , &received_notification
      , GASPI_BLOCK
      );

    sem_wait (mutex_cnt_notified_procs);
    (*ptr_cnt_notified_procs)++;
    sem_post (mutex_cnt_notified_procs);

    // when all the processes on node 1 have received the notification, reset it
    if (*ptr_cnt_notified_procs == n_local_procs)
    {
      gaspi_notification_t val;
      SUCCESS_OR_DIE
        ( gaspi_notify_reset
        , segment_id
        , notification_id
        , &val
        );
    }

    if (ptr_to_shared_data[gpi_local_rank] != gpi_rank - n_local_procs)
    {
      throw std::runtime_error ("Incorrect value stored into the shared buffer portion corresponding to my rank");
    }
  }

  sem_close (mutex_cnt_notified_procs);
  sem_unlink (SEM_NAME);
  SUCCESS_OR_DIE (gaspi_proc_term, GASPI_BLOCK);

  free_shared_data (ptr_to_shared_data, shm_data_size, shm_fd_data, "/shared_mem_for_data");
  free_shared_data (ptr_cnt_notified_procs, sizeof (unsigned int), shm_fd_cnt, "/shared_mem_for_counter");

  return 0;
}
