#include <assert.h>
#include <GASPI.h>
#include <GASPI_Ext.h>

#include <assert.h>
#include <fcntl.h>
#include <memory.h>
#include <cstdlib>
#include <sys/shm.h>
#include <sys/mman.h>
#include <stdexcept>
#include <unistd.h>

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


static void
  wait_for_queue_entries ( gaspi_queue_id_t* queue
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

gaspi_return_t get_ptr_to_shared_data
  (void** ptr_ptr_to_shared_data, unsigned long size, int* shm_fd)
{
  gaspi_rank_t gaspi_local_rank;
  SUCCESS_OR_DIE (gaspi_proc_local_rank, &gaspi_local_rank);

  const char* const name = "/shared_data";
  if (gaspi_local_rank == 0)
  {
    *shm_fd = shm_open (name, O_CREAT | O_RDWR, 0666);
    if (*shm_fd == -1)
     {
       return GASPI_ERROR;
     }

     ftruncate (*shm_fd, size);

     *ptr_ptr_to_shared_data = mmap ( 0
                                    , size
                                    , PROT_READ | PROT_WRITE
                                    , MAP_SHARED
                                    , *shm_fd
                                    , 0
                                    );

     if (*ptr_ptr_to_shared_data == MAP_FAILED)
     {
       close (*shm_fd);
       shm_unlink (name);
       return GASPI_ERROR;
     }

     SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK) ;
  }
  else
  {
    SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK) ;

    *shm_fd = shm_open (name, O_RDWR, 0666);
    if (*shm_fd == -1)
    {
      return GASPI_ERROR;
    }

    *ptr_ptr_to_shared_data = mmap ( 0
                                   , size
                                   , PROT_READ | PROT_WRITE
                                   , MAP_SHARED
                                   , *shm_fd
                                   , 0
                                   );

    if (*ptr_ptr_to_shared_data == MAP_FAILED)
    {
      close (*shm_fd);
      shm_unlink (name);
      return GASPI_ERROR;
    }
  }

  return GASPI_SUCCESS;
}

gaspi_return_t free_shared_data
  (void* ptr_to_shared_data, int size, int shm_fd)
{
  if (munmap (ptr_to_shared_data, size) == -1)
  {
    return GASPI_ERROR;
  }

  if (close (shm_fd) == -1)
  {
    return GASPI_ERROR;
  }

  shm_unlink ("/shared_data");

  return GASPI_SUCCESS;
}

int main(int argc, char* argv[])
{
  gaspi_rank_t my_gaspi_rank, n_gaspi_procs;
  SUCCESS_OR_DIE (gaspi_proc_init, GASPI_BLOCK);
  SUCCESS_OR_DIE (gaspi_proc_rank, &my_gaspi_rank);
  SUCCESS_OR_DIE (gaspi_proc_num, &n_gaspi_procs);

  gaspi_rank_t gpi_local_rank;
  SUCCESS_OR_DIE (gaspi_proc_local_rank, &gpi_local_rank);

  gaspi_rank_t n_local_procs;
  SUCCESS_OR_DIE (gaspi_proc_local_num, &n_local_procs);

  int* ptr_to_shared_data;
  int shm_fd = -1;
  unsigned long shm_data_size = n_local_procs * sizeof(int);

  SUCCESS_OR_DIE (get_ptr_to_shared_data, (void**)&ptr_to_shared_data, shm_data_size,  &shm_fd);

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

  //store some value into the shared segment and notify the others
  //the receivers will add the received values to their value
  ptr_to_shared_data[gpi_local_rank] = 100 + my_gaspi_rank;
  SUCCESS_OR_DIE (gaspi_barrier, GASPI_GROUP_ALL, GASPI_BLOCK);

  gaspi_queue_id_t queue = 0;
  wait_for_queue_entries (&queue, n_gaspi_procs);

  if (my_gaspi_rank == 0)
  {
    for (int k=0; k < n_local_procs; k++)
    {
      int rank = n_local_procs + k;
      SUCCESS_OR_DIE
        ( gaspi_write_notify
        , segment_id
        , 0
        , n_local_procs // local process 0 on remote node 1
        , segment_id
        , k * sizeof (int)
        , sizeof (int)
        , (gaspi_notification_id_t) (rank)
        , (gaspi_notification_t) (my_gaspi_rank + 1)
        , queue
        , GASPI_BLOCK
        );
    }
  }

  // any process on remote node 1 should see the notification (set to its rank)
  if (my_gaspi_rank >= n_local_procs && my_gaspi_rank < 2*n_local_procs - 1)
  {
    gaspi_notification_id_t received_notification;
    SUCCESS_OR_DIE
      ( gaspi_notify_waitsome
      , segment_id
      , my_gaspi_rank
      , 1
      , &received_notification
      , GASPI_BLOCK
      );

    gaspi_notification_t value;
    SUCCESS_OR_DIE
      ( gaspi_notify_reset
      , segment_id
      , received_notification
      , &value
      );

    if (ptr_to_shared_data[gpi_local_rank] != 100)
    {
      throw std::runtime_error ("Incorrect value stored into the shared buffer portion corresponding to my rank");
    }
  }

  SUCCESS_OR_DIE (gaspi_proc_term, GASPI_BLOCK);

  free_shared_data (ptr_to_shared_data, shm_data_size, shm_fd);

  return 0;
}
